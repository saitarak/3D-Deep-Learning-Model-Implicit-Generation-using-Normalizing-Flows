"""Train Real NVP on SHAPENET.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
summary_writer = SummaryWriter('logs_new')

from models.real_nvp import RealNVP, RealNVPLoss
from tqdm import tqdm

import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import sys
sys.path.insert(0,'../ndf/configs')
import config_loader as cfg_loader

cfg = cfg_loader.get_config()
net_ndf = model.NDF()
net_ndf = net_ndf.to('cuda')

train_dataset = voxelized_data.VoxelizedDataset('train',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=0,
                                          sample_distribution=[0.01, 0.49, 0.5],
                                          sample_sigmas=[0.08, 0.02, 0.003]) #cfg.sample_std_dev)
test_dataset = voxelized_data.VoxelizedDataset('test',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=1,
                                          num_sample_points=cfg.num_sample_points_generation,
                                          num_workers=0,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)
train_data_loader = train_dataset.get_loader()
test_data_loader = test_dataset.get_loader()


def main(args):
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'

    # Model
    print('Building model..')
    net = RealNVP(num_scales=2, in_channels=497, mid_channels=128, num_blocks=3)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)
        
    def load_ndf():
        net_ndf = model.NDF()
        net_ndf = net_ndf.to('cuda')
        checkpoint_ndf = torch.load('/checkpoints.tar')
        with torch.no_grad():
            net_ndf.load_state_dict(checkpoint_ndf['model_state_dict'])
            optimizer_ndf = optim.Adam(net_ndf.parameters(), lr= 1e-4)
            optimizer_ndf.load_state_dict(checkpoint_ndf['optimizer_state_dict'])
        print('ndf_loaded')    
        return net_ndf
        
    start_epoch = 0
    
    with torch.no_grad():
        net_ndf = load_ndf()
    
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net_ndf, net, train_data_loader, device, optimizer, loss_fn, args.max_grad_norm)
        test(epoch, net_ndf, net, test_data_loader, device, loss_fn, args.num_samples)

def train(epoch, net_ndf, net, train_data_loader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(train_data_loader.dataset)) as progress_bar:
        for x in train_data_loader:
            inputs = x.get('inputs').to(device)
            x = net_ndf.encoder(inputs)
                       
            optimizer.zero_grad()
            z, sldj, ldj = net(x, reverse=False) 
            
            loss = loss_fn(z, sldj, ldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))
    print('loss_meter average:',loss_meter.avg)
    summary_writer.add_scalar('logs_new/train', loss_meter.avg, epoch)


def sample(net, net_ndf, batch_size, device, samples):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """    
    z = torch.randn((batch_size, 497, 8, 8, 8), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    x = net_ndf.decoder(samples, x)
    return x


def test(epoch, net_ndf, net, test_data_loader, device, loss_fn, num_samples):
    global best_loss
    net_ndf.eval()
    net.eval()
    loss_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(test_data_loader.dataset)) as progress_bar:
            for x in test_data_loader:
                inputs = x.get('inputs').to(device)
                x = net_ndf.encoder(inputs)  
                        
                z, sldj, ldj = net(x, reverse=False)

                loss = loss_fn(z, sldj, ldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    # Save checkpoint
    print('loss_meter average:',loss_meter.avg)
    summary_writer.add_scalar('logs_new/test', loss_meter.avg, epoch)
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
            }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg
        print('best_loss:',best_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on SHAPENET')

    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=1, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = 0

    main(parser.parse_args())