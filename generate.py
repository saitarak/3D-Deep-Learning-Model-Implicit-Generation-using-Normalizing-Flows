import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models.generation import Generator
from sklearn.utils import resample
import open3d
import torch
import os
import trimesh
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '../ndf/configs')
import config_loader as cfg_loader

cfg = cfg_loader.get_config()

device = torch.device("cuda")
net = model.NDF()

dataset = voxelized_data.VoxelizedDataset('test',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=1,
                                          num_sample_points=cfg.num_sample_points_generation,
                                          num_workers=30,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)

gen = Generator(net, cfg.exp_name, device=device)

#For checkpoint used is NDF model
out_path = 'experiments/{}/evaluation/'.format(cfg.exp_name)


def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)
    
    src_sample_pc_list = []
    pc_src_list_uni = []
    tar_sample_pc_list = []
    pc_tar_list_uni = []

    for i, data in tqdm(enumerate(loader)):
 
        path = os.path.normpath(data['path'][0])
        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])
        obj_file_path = path + '/model_normalized.obj'

        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            #continue
        else:
            os.makedirs(export_path)

        for num_steps in [7]:
            point_cloud, duration = gen.generate_point_cloud(data, num_steps)
            
            #source point cloud voxel and uniform sampling method
            point_cloud_pcd = open3d.geometry.PointCloud()
            point_cloud_pcd.points = open3d.utility.Vector3dVector(point_cloud)
            source_pcd_uni = point_cloud_pcd.uniform_down_sample(every_k_points=488)
            source_pcd_uni_array = np.asarray(source_pcd_uni.points) 
            pc_src_list_uni.append(source_pcd_uni_array)
            
            #Source point cloud resample method
            source_pcd = resample(point_cloud,n_samples=2048,replace=False)
            src_sample_pc_list.append(source_pcd)
            
            #Target point cloud voxel and uniform sampling method
            mesh = open3d.io.read_triangle_mesh(obj_file_path)
            n_pts = point_cloud.shape[0]
            pcd = mesh.sample_points_uniformly(n_pts) 
            target_pcd_uni = pcd.uniform_down_sample(every_k_points=488) 
            target_pcd_uni_array = np.asarray(target_pcd_uni.points)  
            pc_tar_list_uni.append(target_pcd_uni_array)     
            
            #Taregt point cloud resample method
            pcd_array = np.asarray(pcd.points)
            target_pcd = resample(pcd_array,n_samples=2048,replace=False)
            tar_sample_pc_list.append(target_pcd)
            np.savez(export_path + 'dense_point_cloud_{}'.format(num_steps), point_cloud=point_cloud, duration=duration)
            trimesh.Trimesh(vertices=point_cloud, faces=[]).export(
                export_path + 'dense_point_cloud_{}.off'.format(num_steps))
            
    pc_src_array_uni = np.array(pc_src_list_uni)
    src_sample_pc_array = np.array(src_sample_pc_list)
    pc_tar_array_uni = np.array(pc_tar_list_uni)
    tar_sample_pc_array = np.array(tar_sample_pc_list)
    np.save('/work/ws-tmp/g051382-NDF_task/ndf/Numpy_files_ndf_new/pc_src_array_uni1', pc_src_array_uni)
    np.save('/work/ws-tmp/g051382-NDF_task/ndf/Numpy_files_ndf_new/src_sample_pc_array1', src_sample_pc_array)
    np.save('/work/ws-tmp/g051382-NDF_task/ndf/Numpy_files_ndf_new/pc_tar_array_uni1', pc_tar_array_uni)
    np.save('/work/ws-tmp/g051382-NDF_task/ndf/Numpy_files_ndf_new/tar_sample_pc_array1', tar_sample_pc_array)

gen_iterator(out_path, dataset, gen)