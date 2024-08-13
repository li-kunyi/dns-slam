import math
import yaml
import os
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

import matplotlib.pyplot as plt

from mathutils import Matrix


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def setup_dist_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        return 0, 1  # Single GPU run
        
    dist.init_process_group(backend="nccl")
    print(f'Initialized process {local_rank} / {world_size}')
    torch.cuda.set_device(local_rank)

    setup_dist_print(local_rank == 0)
    return local_rank, world_size


__LOG10 = math.log(10)


def mse2psnr(x):
    return -10.*torch.log(x)/__LOG10


def setup_dist_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def using_dist():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not using_dist():
        return 1
    return dist.get_world_size()


def get_rank():
    if not using_dist():
        return 0
    return dist.get_rank()


def gather_all(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return tensor_list


def reduce_dict(input_dict, average=True):
    """
    Reduces the values in input_dict across processes, when distributed computation is used.
    In all processes, the dicts should have the same keys mapping to tensors of identical shape.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys] 

    if average:
        op = dist.ReduceOp.AVG
    else:
        op = dist.ReduceOp.SUM

    for value in values:
        dist.all_reduce(value, op=op)

    reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


class LrScheduler():
    """ Implements a learning rate schedule with warum up and decay """
    def __init__(self, peak_lr=5e-4, peak_it=5, decay_rate=0.5, decay_it=20):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:  # Warmup period
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, R, T, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).to(device)
        T = torch.from_numpy(T).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * R, -1)
    rays_o = T.expand(rays_d.shape)
    return rays_o, rays_d

def select_uv(i, j, n, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    channel = color.shape[-1]
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    color = color.reshape(-1, channel)
    color = color[indices]  # (n,3)
    return i, j, color

def get_sample_uv(H0, H1, W0, W1, n, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    i = i.t()  # transpose
    j = j.t()
    i, j, color = select_uv(i, j, n, color, device=device)
    return i, j, color


def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, R, T, color, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    i, j, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, R, T, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_color


def select_by_class(i, j, n, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    label = color[:, :, -1]
    classes = torch.unique(label, sorted=True)
    n_class = classes.numel()
    n_k = n // n_class
    indices = []
    for id in range(n_class):
        if id == 0:
            m = n - n_k * (n_class - 1)
        else:
            m = n_k
        indices_class = torch.nonzero(torch.eq(label.reshape(-1), classes[id])).squeeze()
        k = indices_class.numel()
        if k == 1:
            indices.append(indices_class.repeat(m))
        else:
            idx = torch.randint(k, (m,), device=device)
            indices.append(indices_class[idx])

    indices = torch.cat(indices, dim=-1)
    channel = color.shape[-1]
    i = i.reshape(-1)
    j = j.reshape(-1)
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    color = color.reshape(-1, channel)
    color = color[indices]  # (n,3)
    return i, j, color

def get_sample_by_class(H0, H1, W0, W1, n, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    i = i.t()  # transpose
    j = j.t()
    i, j, color = select_by_class(i, j, n, color, device=device)
    return i, j, color

def get_samples_by_class(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, R, T, color, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    i, j, sample_color = get_sample_by_class(
        H0, H1, W0, W1, n, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, R, T, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_color


def get_samples_by_uniq_class(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, R, T, color, class_dict, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    i = i.t()  # transpose
    j = j.t()

    label = color[:, :, -1]
    classes = class_dict
    n_class = len(classes)
    n_k = n // n_class
    indices = []
    for id in range(n_class):
        if id == 0:
            m = n - n_k * (n_class - 1)
        else:
            m = n_k
        indices_class = torch.nonzero(torch.eq(label.reshape(-1), classes[id])).squeeze()
        k = indices_class.numel()
        if k == 1:
            indices.append(indices_class.repeat(m))
        elif k > 1:
            idx = torch.randint(k, (m,), device=device)
            indices.append(indices_class[idx])

    indices = torch.cat(indices, dim=-1)
    channel = color.shape[-1]
    i = i.reshape(-1)
    j = j.reshape(-1)
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    color = color.reshape(-1, channel)
    color = color[indices]  # (n,3)

    rays_o, rays_d = get_rays_from_uv(i, j, R, T, fx, fy, cx, cy, device)
    return rays_o, rays_d, color


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT

def get_rotation_from_quad(quad):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(quad.shape)
    if N == 1:
        quad = quad.unsqueeze(0)
    R = quad2rotation(quad)
    if N == 1:
        R = R[0]
    return R


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def get_quad_from_camera(RT):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    quad = np.concatenate([quad], 0)
    quad = torch.from_numpy(quad).float()
    if gpu_id != -1:
        quad = quad.to(gpu_id)
    return quad

def raw2nerf_color(raw, z_vals, rays_d, occupancy=True, device='cuda:0'):
    """
    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.
    """
    def raw2alpha(raw, dists): return 1. - torch.exp(-F.relu(raw)*dists)
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(z_vals.device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :3]
    if occupancy:
        alpha = torch.sigmoid(10*raw[..., -1])
    else:
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(z_vals.device).float(), 
                                                       (1.-alpha + 1e-10).float()], -1).float(), -1)[..., :-1]
    weights = weights / weights.sum(dim=-1)[:, None]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=-1)  # (N_rays)
    return depth_map, depth_var, rgb_map, weights


def get_all_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)  # [H, W, 3]
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def sample_along_rays(gt_depth, n_samples, n_surface, far_bb, device):
    # [N, C]
    gt_depth = gt_depth.reshape(-1, 1)  # [n_pixels, 1]
    
    gt_none_zero_mask = gt_depth > 0
    gt_none_zero = gt_depth[gt_none_zero_mask]
    gt_none_zero = gt_none_zero.unsqueeze(-1)

    # for thoe with validate depth, sample near the surface
    gt_depth_surface = gt_none_zero.repeat(1, n_surface)  # [n_pixels, n_samples//2]
    t_vals_surface = torch.rand(n_surface).to(device)
    if not torch.any(t_vals_surface == 0.5):
        t_vals_surface[n_surface//2+1] = 0.5
    z_vals_surface_depth_none_zero = 0.95 * gt_depth_surface * (1.-t_vals_surface) + 1.05 * gt_depth_surface * (t_vals_surface)
    z_vals_near_surface = torch.zeros(gt_depth.shape[0], n_surface).to(device)
    gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
    z_vals_near_surface[gt_none_zero_mask, :] = z_vals_surface_depth_none_zero

    # for those with zero depth, random sample along the space
    near = 0.001
    far = torch.max(gt_depth)
    t_vals_surface = torch.rand(n_surface).to(device)
    z_vals_surface_depth_zero = near * (1.-t_vals_surface) + far * (t_vals_surface)
    z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)
    z_vals_near_surface[~gt_none_zero_mask, :] = z_vals_surface_depth_zero

    # none surface
    if n_samples > 0:
        gt_depth_samples = gt_depth.repeat(1, n_samples)  # [n_pixels, n_samples]
        near = gt_depth_samples * 0.001
        far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
        z_vals = near * (1.-t_vals) + far * (t_vals)

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_near_surface], -1), -1)  # [n_pixels, n_samples]
    else:
        z_vals, _ = torch.sort(z_vals_near_surface, -1)

    return z_vals.float()

def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p
    

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def compute_depth_var(depth):
    return torch.sum(torch.square(depth - torch.mean(depth))) / (depth.numel())


def feature_searching(pts, features, H, W):
    N = features.shape[0]
    code_pts = []
    for i in range(N):
        h = pts[i,:,1].clamp(0, H - 1)
        w = pts[i,:,0].clamp(0, W - 1)
        ft = features[i]
        code_pts.append(ft[:, h, w])
    code_pts = torch.stack(code_pts, dim=0)
    
    return code_pts


def feature_matching(H, W, K, pts_, refer_w2c, features, merge_fn):
    features = F.interpolate(features, size=[H, W], mode='bilinear', align_corners=True)
    ones = torch.ones(pts_.shape[0], 1).to(pts_.device)
    pts = torch.cat((pts_, ones), dim=-1)

    pts_img_view = torch.matmul(refer_w2c, pts.permute(1, 0)) 
    pts_img_view[:, 1, :] *= -1
    pts_img_view[:, 2, :] *= -1
    proj_depth = pts_img_view[:, 2, :]
    pts_img_view = torch.matmul(K[None, :, :], pts_img_view[:, :3, :])

    uv_img = torch.zeros_like(pts_img_view[:, :2, :])
    uv_img = pts_img_view[:, :2, :] / (pts_img_view[:, 2:3, :] + 1e-5)

    uv_img = torch.round(uv_img.permute(0, 2, 1))

    mask_x = (uv_img[:, :, 0] > 0) * (uv_img[:, :, 0] < W - 1)
    mask_y = (uv_img[:, :, 1] > 0) * (uv_img[:, :, 1] < H - 1)
    mask_depth = proj_depth > 0
    mask = mask_x * mask_y * mask_depth
    uv_img = uv_img * mask[:, :, None]
    uv_img_ = (uv_img).to(torch.int64)  #[batch_size, num_points, 2]

    # # [batch_size, channels, height, width]
    code_pts = feature_searching(uv_img_, features, H, W)
    code_pts = code_pts.permute(0, 2, 1)   # [n_refer, n_point, C] if the point cannot macth to the reference frame, use the zero code

    refer_c2w = torch.inverse(refer_w2c)
    refer_view = refer_c2w[:, :3, 2]
    refer_o = refer_c2w[:, :3, 3]
    refer_p = pts_[None, :, :] - refer_o[:, None, :]
    code_pts = code_pts* mask[:, :, None]
    code_pts = merge_fn(refer_p, refer_o, code_pts)

    return code_pts  # [n_pts, c]


def fig_plot(idx, out_dir, gt_color, est_color, gt_depth, est_depth, gt_label, est_label):
    color_residual = np.abs(gt_color - est_color)
    #color_residual[gt_depth_np == 0.0] = 0.0
    depth_residual = np.abs(gt_depth - est_depth)
    #depth_residual[gt_depth_np == 0.0] = 0.0
    label_residual = np.abs(gt_label - est_label)

    fig, axs = plt.subplots(3, 3)
    fig.tight_layout()
    max_depth = np.max(gt_depth)
    axs[0, 0].imshow(gt_depth, cmap="plasma", vmin=0, vmax=max_depth)
    axs[0, 0].set_title('Input Depth')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].imshow(est_depth, cmap="plasma", vmin=0, vmax=max_depth)
    axs[0, 1].set_title('Generated Depth')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
    axs[0, 2].set_title('Depth Residual')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    
    gt_color = np.clip(gt_color, 0, 1)
    axs[1, 0].imshow(gt_color, cmap="plasma")
    axs[1, 0].set_title('Input RGB')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    est_color = np.clip(est_color, 0, 1)
    axs[1, 1].imshow(est_color, cmap="plasma")
    axs[1, 1].set_title('Generated RGB')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    color_residual = np.clip(color_residual, 0, 1)
    axs[1, 2].imshow(color_residual, cmap="plasma")
    axs[1, 2].set_title('RGB Residual')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    max_label = 101
    axs[2, 0].imshow(gt_label, cmap="plasma", vmin=0, vmax=max_label)
    axs[2, 0].set_title('Input Label')
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    axs[2, 1].imshow(est_label, cmap="plasma", vmin=0, vmax=max_label)
    axs[2, 1].set_title('Generated Label')
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])
    axs[2, 2].imshow(label_residual, cmap="plasma", vmin=0, vmax=max_label)
    axs[2, 2].set_title('Label Residual')
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])


    plt.subplots_adjust(wspace=0, hspace=0.2)
    file_path = out_dir
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(
        f'{out_dir}/{idx:05d}.jpg', bbox_inches='tight', pad_inches=0.3, dpi=300)
    plt.clf()
    plt.close()

    print(f'{idx}th rerendering image saved.')


def coordinates(voxel_dim, device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def approx_occ(x, sigma):
    gaussian_part = 0.5 * torch.exp(-0.5 * (x / sigma) ** 2)
    return gaussian_part


def get_opacity_loss(z_vals, depth, occ, truncation=0.2, sigma=0.05):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        dpeth: torch.Tensor, (Bs, 1)
        predicted_sdf: torch.Tensor, (Bs, N_samples)
        truncation: float
    Return:
        fs_loss: torch.Tensor, (1,)
        opacity_loss: torch.Tensor, (1,)
        eikonal_loss: torch.Tensor, (1,)
    '''
    bs, n_sample = z_vals.shape
    depth = depth.unsqueeze(-1)
    occ = torch.sigmoid(10*occ).reshape(bs, n_sample)

    # before truncation
    front_mask = torch.where(z_vals < (depth - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (depth + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(depth > 0.0, torch.ones_like(depth), torch.zeros_like(depth))
    # Valid sdf regionn
    opacity_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    if torch.count_nonzero(front_mask) > 0 and torch.count_nonzero(opacity_mask) > 0:
        fs_loss = ((occ * front_mask * depth_mask) ** 2).mean()
        pesudo_occ = approx_occ((z_vals - depth), sigma=sigma)
        opacity_loss = ((occ * opacity_mask - pesudo_occ * opacity_mask) ** 2).mean()
    else:
        fs_loss = torch.tensor(0)
        opacity_loss = torch.tensor(0)

    return fs_loss, opacity_loss