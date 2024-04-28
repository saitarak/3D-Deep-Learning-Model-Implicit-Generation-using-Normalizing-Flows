import torch
import torch.nn.functional as F


def squeeze_2x2(x, reverse=False, alt_order=False):
    """For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
    reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.
    Adapted from:
        https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py
    See Also:
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        reverse (bool): Whether to do a reverse squeeze (unsqueeze).
        alt_order (bool): Whether to use alternate ordering.
    """
    block_size = 2
    if alt_order:
        n, c, h, w, d = x.size()

        if reverse:
            if c % 8 != 0:
                raise ValueError('Number of channels must be divisible by 8, got {}.'.format(c))
            c //= 8
        else:
            if h % 2 != 0:
                raise ValueError('Height must be divisible by 2, got {}.'.format(h))
            if w % 2 != 0:
                raise ValueError('Width must be divisible by 2, got {}.'.format(w))
            if d % 2 != 0:
                raise ValueError('Width must be divisible by 2, got {}.'.format(w))
        # Defines permutation of input channels (shape is (4, 1, 2, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        squeeze_matrix = torch.unsqueeze(squeeze_matrix, dim=-1)
        squeeze_matrix = squeeze_matrix.expand(-1, -1, -1, -1, 2)
        batch_size = 2
        squeeze_matrix = torch.cat([squeeze_matrix] * batch_size, dim=0)
        perm_weight = torch.zeros((8 * c, c, 2, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 8, (c_idx + 1) * 8)
            #print('slice_0:',slice_0)
            slice_1 = slice(c_idx, c_idx + 1)
            #print('slice_1:',slice_1)
            perm_weight[slice_0, slice_1, :, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 8 for c_idx in range(c)]
                                        + [c_idx * 8 + 1 for c_idx in range(c)]
                                        + [c_idx * 8 + 2 for c_idx in range(c)]
                                        + [c_idx * 8 + 3 for c_idx in range(c)]
                                        + [c_idx * 8 + 4 for c_idx in range(c)]
                                        + [c_idx * 8 + 5 for c_idx in range(c)]
                                        + [c_idx * 8 + 6 for c_idx in range(c)]
                                        + [c_idx * 8 + 7 for c_idx in range(c)])
        perm_weight = perm_weight[shuffle_channels, :, :, :, :]

        if reverse:
            x = F.conv_transpose3d(x, perm_weight, stride=2)
        else:
            x = F.conv3d(x, perm_weight, stride=2)
    else:
        b, c, h, w, d = x.size()
        x = x.permute(0, 2, 3, 4, 1)

        if reverse:
            if c % 8 != 0:
                raise ValueError('Number of channels {} is not divisible by 8'.format(c))
            x = x.view(b, h, w, d, c // 8, 2, 2, 2)
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
            x = x.contiguous().view(b, 2 * h, 2 * w, 2*d, c // 8)
        else:
            if h % 2 != 0 or w % 2 != 0 or d % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, d//2, 2, c)
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
            x = x.contiguous().view(b, h // 2, w // 2, d//2, c * 8)

        x = x.permute(0, 4, 3, 1, 2)

    return x


def checkerboard_mask(height, width, depth, reverse=False, dtype=torch.float32,
                      device=None, requires_grad=False):
    """Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.
    Args:
        height (int): Number of rows in the mask.
        width (int): Number of columns in the mask.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.
    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width, depth).
    """
    checkerboard = [[[((((i % 2) + j) % 2) + k) % 2 for k in range(depth)] for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, 1, height, width, depth) for broadcasting with tensors of shape (B, C, H, W, D)
    mask = mask.view(1, 1, height, width, depth)

    return mask