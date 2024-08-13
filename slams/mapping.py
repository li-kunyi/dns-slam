import os
import time, datetime
import math
import copy
import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.nn import functional as F
import tinycudann as tcnn
from models.encoder import ResNet

from utils.common import get_camera_from_tensor, get_quad_from_camera, get_rotation_from_quad, get_opacity_loss
from utils.common import get_all_rays, get_samples, get_samples_by_class, get_samples_by_uniq_class, sample_along_rays
from utils.common import mse2psnr, raw2nerf_color, fig_plot, feature_matching, get_tensor_from_camera, coordinates


class Mapper(object):
    def __init__(self, cfg, slam, device):
        self.cfg = cfg
        self.scene = slam.scene

        self.device = device
        self.low_gpu_mem = slam.low_gpu_mem

        self.verbose = slam.verbose
        self.out_dir = slam.out_dir

        self.scale = slam.scale
        self.bound = slam.bound.to(self.device)

        self.encoder = ResNet().to(self.device)
        self.decoder = slam.decoder

        self.pe_dim = slam.decoder.pe_dim
        self.grid_dim = slam.decoder.grid_dim

        self.pts_dim = cfg['model']['pts_dim']
        self.hidden_dim = cfg['model']['hidden_dim']
        self.pixel_dim = cfg['model']['pixel_dim']

        self.fine_decoders = {}

        self.checkpoint = slam.checkpoint

        self.lr = slam.lr
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        
        self.front_idx = slam.front_idx
        self.back_idx = slam.back_idx

        self.estimate_c2w_list = slam.estimate_c2w_list
        self.gt_c2w_list = slam.gt_c2w_list
        self.first_frame_optimized = slam.first_frame_optimized

        self.keyframe_dict = []
        self.keyframe_list = []

        self.back_dataset = slam.back_dataset 
        self.n_img = len(self.back_dataset)
        self.n_class = self.back_dataset.n_class
        self.label2class_dict = self.back_dataset.label2class_dict
        self.class2label_dict = self.back_dataset.class2label_dict

        self.n_joint_optimize_frames = cfg['mapping']['n_joint_optimize_frames']
        self.n_refer_frames = cfg['mapping']['n_refer_frames']
        self.n_pixels = cfg['mapping']['n_pixels']
        self.n_iters = cfg['mapping']['n_iters']
        self.n_pts_batch = cfg['mapping']['n_pts_batch']
        self.optimize_every_frame_flag = slam.optimize_every_frame

        self.n_surface_ray = cfg['training']['n_surface_ray']
        self.n_samples_ray = cfg['training']['n_samples_ray']

        self.H = slam.H
        self.W = slam.W
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy

        self.K = torch.tensor([[self.fx,  0,     self.cx],
                               [0,     self.fy,  self.cy],
                               [0,        0,        1   ]]).to(self.device)

        self.sync_method = cfg['sync_method']

        self.choose_keyframe_every = cfg['mapping']['choose_keyframe_every']
        self.optimize_every_n_frames = cfg['mapping']['optimize_every_n_frames']
        self.vis_every = cfg['mapping']['vis_every']
        self.mesh_every = cfg['mapping']['mesh_every']
        self.checkpoint_every = cfg['mapping']['checkpoint_every']
        
        self.lambda_p = cfg['training']['lambda_color']
        self.lambda_d = cfg['training']['lambda_depth']
        self.lambda_l = cfg['training']['lambda_label']
        self.lambda_sm = cfg['training']['lambda_smooth']
        self.lambda_fs = cfg['training']['lambda_fs']
        self.lambda_opacity = cfg['training']['lambda_opacity']

        self.ignore = False
        self.seperate_LR = cfg['seperate_LR']
        self.start_optimize_idx = cfg['mapping']['start_optimize_idx']
        self.exist_decoders = {}

        self.mesher = slam.mesher


    def compute_photometric_loss(self, gt_color, pred_color):
        loss = ((gt_color - pred_color) ** 2).mean()
        return loss
    
    def compute_depth_loss(self, gt_depth, pred_depth):
        mask = gt_depth > 0
        loss = (torch.abs(gt_depth[mask] - pred_depth[mask])).mean()
        return loss
    
    def compute_label_loss(self, gt_label, pred_logits):
        loss = F.cross_entropy(pred_logits, gt_label)
        return loss

    def compute_latent_loss(self, coarse, fine):
        # loss = (torch.abs(coarse - fine)).mean()
        loss = ((coarse - fine) ** 2).mean()
        return loss


    def smoothness(self, sample_points=64, voxel_size=0.1, margin=0.05):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bound[:, 1] - self.bound[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bound[:, 1]-self.bound[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bound[:, 0] + offset

        pts = (pts - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])

        pts_shape = pts.shape

        pts = pts.reshape(-1, 3)

        pe, grid_pts = self.decoder.pe_fn(pts)

        coarse_latents = self.decoder.coarse_fn(pe, features=grid_pts)
        occ = coarse_latents[:, 0:1]
        occ = occ.reshape(pts_shape[:3], 1)
        tv_x = torch.pow(occ[1:,...]-occ[:-1,...], 2).sum()
        tv_y = torch.pow(occ[:,1:,...]-occ[:,:-1,...], 2).sum()
        tv_z = torch.pow(occ[:,:,1:,...]-occ[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss

    def random_select(self, n, k):
        """
        Random select k values from 0..n.

        """
        if n == 0:
            return [0]
        else:
            return list(np.random.choice(np.array(range(n)), size=k, replace=True))

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, 
                                   keyframe_dict, k, N_samples=16, pixels=100, th=0.0):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        gt_depth = gt_depth.unsqueeze(-1)
        rays_o, rays_d, samples = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w[:3, :3], c2w[:3, -1], 
            torch.cat((gt_color, gt_depth), -1), device)

        gt_color = samples[:, :3]
        gt_depth = samples[:, -1]
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate([vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 10
            mask = (uv[:, 0] < W-edge) * (uv[:, 0] > edge) * \
                   (uv[:, 1] < H-edge) * (uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append({'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id'] for dic in list_keyframe 
                                  if dic['percent_inside'] > th]
        selected_keyframe_list = list(np.random.permutation(np.array(
                                      selected_keyframe_list))[:k])

        return selected_keyframe_list
    
    
    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool_)
            return mask
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak-ray_o
        dist = torch.sum(dist*dist, axis=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask
    
    def choose_refer_frames(self, refer_frame_idx):

        gt_color_list = []
        est_c2w_list = []
        for refer_id in refer_frame_idx:
            gt_color = self.keyframe_dict[refer_id]['gt_color'].to(self.device)
            est_c2w = self.keyframe_dict[refer_id]['est_c2w'].to(self.device)

            gt_color_list.append(gt_color.float())
            est_c2w_list.append(est_c2w.float())
        
        gt_color_list = torch.stack(gt_color_list, 0).to(self.device)
        est_c2w_list = torch.stack(est_c2w_list, 0).to(self.device)

        return gt_color_list, est_c2w_list
    
    
    def set_target_refer_frames(self, cur_gt_color, cur_gt_depth, 
                                cur_gt_label, cur_gt_c2w, cur_c2w):
        target_gt_color_list = []
        target_gt_depth_list = []
        target_gt_label_list = []
        target_gt_c2w_list = []
        target_est_c2w_list = []
        target_labels = []
        target_frame_idx = []

        refer_gt_color_list = []
        refer_est_c2w_list = []
        refer_idx_list = []

        # number of optimized frames
        num_frames = self.n_joint_optimize_frames-2 if len(self.keyframe_list) > self.n_joint_optimize_frames-2 else len(self.keyframe_list)
        
        # id of optimized frames
        if (len(self.keyframe_list) < 2):
            target_frame_idx = []
        else:
            if self.mapping_mode == 'global':        
                target_frame_idx = self.random_select(
                    len(self.keyframe_list[:-1]), num_frames)
            elif self.mapping_mode == 'overlap':
                target_frame_idx = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, 
                    self.keyframe_dict[:-1], num_frames, th=0.05)
        # uniqueness
        if len(self.keyframe_list) > 1:    
            target_frame_idx = target_frame_idx + [len(self.keyframe_list)-1]
            target_frame_idx = list(set(target_frame_idx))
            target_frame_idx = [x for x in target_frame_idx if x != 0]
            target_frame_idx.sort()

        # [-1] represent current frame
        target_frame_idx += [-1]
        self.n_target_frame = len(target_frame_idx)

        # print('Now have', len(self.keyframe_list), 'keyframes')
        # print('Selected target frames:', target_frame_idx)

        # for each optimized frame, get data
        for target_id in target_frame_idx:
            if target_id != -1:
                target_gt_color = self.keyframe_dict[target_id]['gt_color']
                target_gt_color = target_gt_color.to(self.device)
                target_gt_depth = self.keyframe_dict[target_id]['gt_depth']
                target_gt_depth = target_gt_depth.to(self.device)
                target_gt_label = self.keyframe_dict[target_id]['gt_label']
                target_gt_label = target_gt_label.to(self.device)
                target_gt_c2w = self.keyframe_dict[target_id]['gt_c2w']
                target_gt_c2w = target_gt_c2w.to(self.device)
                target_est_c2w = self.keyframe_dict[target_id]['est_c2w']
                target_est_c2w = target_est_c2w.to(self.device)
            else:
                target_gt_color = cur_gt_color
                target_gt_depth = cur_gt_depth
                target_gt_label = cur_gt_label
                target_gt_c2w = cur_gt_c2w
                target_est_c2w = cur_c2w

            labels = torch.unique(target_gt_label, sorted=True)
            target_labels.append(labels)

            target_gt_color_list.append(target_gt_color.float())
            target_gt_depth_list.append(target_gt_depth.float())
            target_gt_label_list.append(target_gt_label)
            target_gt_c2w_list.append(target_gt_c2w.float())
            target_est_c2w_list.append(target_est_c2w.float())

            # for each target frame, find two reference image
            if target_id == -1:
                first = 0 if len(self.keyframe_list) - 2 < 0 else len(self.keyframe_list) - 2
                second = 0 if len(self.keyframe_list) - 1 < 0 else len(self.keyframe_list) - 1
            elif target_id == len(self.keyframe_list)-1: 
                first = 0 if len(self.keyframe_list) - 3 < 0 else len(self.keyframe_list) - 3
                second = 0 if len(self.keyframe_list) - 2 < 0 else len(self.keyframe_list) - 2
            else: 
                first = 0 if target_id - 1 < 0 else target_id - 1
                second = target_id + 1

            refer_frame_idx = [first, second]
            # print('Target frame idx:', target_id, 'Refer frame idx:', refer_frame_idx)

            # for each reference frame, get data
            refer_gt_color, refer_est_c2w = self.choose_refer_frames(refer_frame_idx)
            refer_frame_idx += [-1]
            refer_gt_color_list.append(torch.cat((refer_gt_color, target_gt_color.unsqueeze(0)), 0))
            refer_est_c2w_list.append(torch.cat((refer_est_c2w, target_est_c2w.unsqueeze(0)), 0))

            # refer_gt_color_list.append(refer_gt_color)
            # refer_est_c2w_list.append(refer_est_c2w)
            refer_idx_list.append(refer_frame_idx)

        target_frames = {'gt_color': torch.stack(target_gt_color_list, 0).to(self.device),  # [B, H, W, 3]
                         'gt_depth': torch.stack(target_gt_depth_list, 0).to(self.device),
                         'gt_label': torch.stack(target_gt_label_list, 0).to(self.device),
                         'gt_c2w': torch.stack(target_gt_c2w_list, 0).to(self.device),
                         'est_c2w': torch.stack(target_est_c2w_list, 0).to(self.device),
                         'label_dict': torch.unique(torch.cat(target_labels, 0), sorted=True).cpu().int().numpy(),
                         'kf_idx': target_frame_idx}

        refer_frames = {'gt_color': torch.stack(refer_gt_color_list, 0).to(self.device),  # [B, N, H, W, 3]
                        'est_c2w': torch.stack(refer_est_c2w_list, 0).to(self.device),
                        'kf_idx': refer_idx_list}  # kf_idx: which kf is this reference frame from

        return target_frame_idx, target_frames, refer_frames

    def set_optimizer(self, target_frames):
        # nerf of one class needs to be optimized only if it is in current frames 
        label_dict = target_frames['label_dict']

        net_para_list = []
        net_para_list += list(self.decoder.parameters())

        for obj_id in label_dict:
            net_para_list += list(self.fine_decoders[obj_id].parameters())

        # seperate LR for quaterion and translation vector
        quad_list = []
        T_list = []
        for frame in range(self.n_target_frame):
            c2w = target_frames['est_c2w'][frame]
            quad = get_quad_from_camera(c2w).clone().detach()
            T = c2w[:3, 3].clone().detach()

            # the oldest frame(0) should be fixed to avoid drifting
            if (self.n_target_frame == 1 or frame != 0) and self.is_BA == True:
                quad = Variable(quad.to(self.device), requires_grad=True)
                T = Variable(T.to(self.device), requires_grad=True)

            quad_list.append(quad.to(self.device))
            T_list.append(T.to(self.device))

        optimizer = torch.optim.Adam([{'params': net_para_list, 'lr': 0},
                                          {'params': quad_list, 'lr': 0},
                                          {'params': T_list, 'lr': 0}])

        return optimizer, quad_list, T_list
    
    
    def get_target_samples(self, target_frames, quad_list, T_list, refer_frames=None, features=None):
        gt_color_sp_list = []
        gt_depth_sp_list = []
        gt_label_sp_list = []
        rays_o_sp_list = []
        rays_d_sp_list = []
        pts_sp_list = []
        z_vals_sp_list = []
        mask_list = []
        code_list = []

        target_idx = target_frames['kf_idx']

        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                 [1, 4])).type(torch.float32).to(self.device)
        n_pixels = self.n_pixels // self.n_target_frame
        for i in range(self.n_target_frame):
            gt_color = target_frames['gt_color'][i]
            gt_depth = target_frames['gt_depth'][i]
            gt_label = target_frames['gt_label'][i]

            R = get_rotation_from_quad(quad_list[i])
            T = T_list[i]
            cur_c2w = torch.cat([torch.cat((R, T[:, None]), -1), bottom], dim=0)
            
            # unified sample along the whole image
            rays_o_sp1, rays_d_sp1, samples1 = get_samples(
                0, self.H, 0, self.W, n_pixels // 3 * 2, self.H, self.W, 
                self.fx, self.fy, self.cx, self.cy, R, T,
                torch.cat((gt_color, gt_depth.unsqueeze(-1), gt_label.unsqueeze(-1)), -1), 
                self.device)  # [n_pixels, C]
            
            # sample by class
            rays_o_sp2, rays_d_sp2, samples2 = get_samples_by_class(
                0, self.H, 0, self.W, n_pixels // 3, self.H, self.W, 
                self.fx, self.fy, self.cx, self.cy, R, T,
                torch.cat((gt_color, gt_depth.unsqueeze(-1), gt_label.unsqueeze(-1)), -1), 
                self.device)  # [n_pixels, C]
            
            rays_o_sp = torch.cat((rays_o_sp1, rays_o_sp2), dim=0)
            rays_d_sp = torch.cat((rays_d_sp1, rays_d_sp2), dim=0)
            samples = torch.cat((samples1, samples2), dim=0)
            
            gt_color_sp = samples[:, :3]  # [n_pixels, 3]
            gt_depth_sp = samples[:, 3]  # [n_pixels]
            gt_label_sp = samples[:, 4]

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = rays_o_sp.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_sp.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(self.device) -
                    det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = far_bb >= gt_depth_sp
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

            # sample along the rays
            z_vals_sp = sample_along_rays(gt_depth_sp, self.n_samples_ray, self.n_surface_ray, far_bb, self.device)  # [n_pixels, n_samples]
            pts_sp = rays_o_sp[:, None, :] + rays_d_sp[:, None, :] * z_vals_sp[:, :, None]

            refer_idx = refer_frames['kf_idx'][i]
            refer_w2c = []
            for refer_id in refer_idx:
                if refer_id == -1:
                    refer_c2w = cur_c2w.clone().detach()
                elif refer_id in target_idx:  # if reference frame is also one of target frame, use pose from optimizer
                    target_id = target_idx.index(refer_id)
                    R_ = get_rotation_from_quad(quad_list[target_id])
                    T_ = T_list[target_id]
                    refer_c2w = torch.cat([torch.cat((R_, T_[:, None]), -1), bottom], dim=0).clone().detach()
                else:  # else use fixed pose
                    refer_c2w = refer_frames['est_c2w'][i][refer_idx.index(refer_id)].clone().detach()

                refer_w2c.append(torch.inverse(refer_c2w))

            refer_w2c = torch.stack(refer_w2c, 0)
            code = feature_matching(
                self.H, self.W, self.K, pts_sp.flatten(0, 1), refer_w2c, features[i], self.decoder.merge)
            code = code.reshape(pts_sp.shape[0], pts_sp.shape[1], -1)

            front_mask = torch.where(z_vals_sp < (gt_depth_sp[:, None] * 0.95), torch.ones_like(z_vals_sp), torch.zeros_like(z_vals_sp))
            back_mask = torch.where(z_vals_sp > (gt_depth_sp[:, None] * 1.05), torch.ones_like(z_vals_sp), torch.zeros_like(z_vals_sp))
            depth_mask = torch.where(gt_depth_sp[:, None] > 0.0, torch.ones_like(gt_depth_sp[:, None]), torch.zeros_like(gt_depth_sp[:, None]))
            trunc_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask
            code = code * trunc_mask[..., None]

            gt_color_sp_list.append(gt_color_sp.float())
            gt_depth_sp_list.append(gt_depth_sp.float())
            gt_label_sp_list.append(gt_label_sp.to(torch.int64))
            rays_o_sp_list.append(rays_o_sp.float())
            rays_d_sp_list.append(rays_d_sp.float())
            pts_sp_list.append(pts_sp.float())
            z_vals_sp_list.append(z_vals_sp.float())
            mask_list.append(inside_mask)
            code_list.append(code)

        gt_color_batch = torch.cat(gt_color_sp_list, 0)  # [N, 3]
        gt_depth_batch = torch.cat(gt_depth_sp_list, 0)  # [N]
        gt_label_batch = torch.cat(gt_label_sp_list, 0)  # [N, n_class]
        rays_o_batch = torch.cat(rays_o_sp_list, 0)  # [N, 3]
        rays_d_batch = torch.cat(rays_d_sp_list, 0)  # [N, 3]
        pts_batch = torch.cat(pts_sp_list, 0)  # [N, n_samples, 3]
        z_vals_batch = torch.cat(z_vals_sp_list, 0)
        mask_batch = torch.cat(mask_list, 0).cpu().numpy()
        code_batch = torch.cat(code_list, 0)

        samples_all = {'gt_color': gt_color_batch[mask_batch],  # [N, 3]
                       'gt_depth': gt_depth_batch[mask_batch],  # [N]
                       'gt_label': gt_label_batch[mask_batch],  # [N, n_class]
                       'rays_o': rays_o_batch[mask_batch],  # [N, 3]
                       'rays_d': rays_d_batch[mask_batch],  # [N, 3]
                       'pts': pts_batch[mask_batch],  # [N, n_samples, 3]
                       'z_vals': z_vals_batch[mask_batch],
                       'features': code_batch[mask_batch]}
        
        return samples_all
    
    def fine_fn(self, pes, classes=None, features=None):
        n_class = torch.unique(classes, sorted=True).cpu().int().numpy()
        latents = torch.zeros(pes.shape[0], self.hidden_dim+1, device=pes.device)
        for i in n_class:
            if i not in self.fine_decoders.keys():
                raise ValueError('Fine decoders does NOT have class', i)
            index = (classes[:] == i).cpu().numpy()
            if index.sum() > 1:
                pe = pes[index, :]
                feature = features[index, :]
                latents[index, :] = self.fine_decoders[i](torch.cat((pe, feature), -1)).float()
        return latents
    
    def renderer(self, samples):
        pts = samples['pts']  # [B*n_pts, n_sample, 3]
        n_pts, n_samples, _  = pts.shape
        pts = pts.flatten(0, 1)  # [B*N*n_samples, C]
        
        pts = (pts - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])

        rays_d = samples['rays_d']  # [B*n_pts, 3]
        z_vals = samples['z_vals']  # [B*n_pts, n_sample]
        gt_label = samples['gt_label']
        gt_label = gt_label.repeat(1, n_samples).flatten(0, 1)

        # reference feature
        pixel_pts = samples['features']
        pixel_pts = pixel_pts.flatten(0, 1)

        pe, grid_pts = self.decoder.pe_fn(pts)

        coarse_latents = self.decoder.coarse_fn(pe, features=grid_pts)
        fine_latents = self.fine_fn(pe, classes=gt_label, features=grid_pts)

        color_pts, logits_pts = self.decoder.out_fn(pe, torch.cat((fine_latents[:, 1:], pixel_pts), -1))

        values_pts = torch.cat((color_pts, fine_latents[:, 0:1]), -1)

        values_pts = values_pts.reshape(n_pts, n_samples, -1)
        logits_pts = logits_pts.reshape(n_pts, n_samples, -1)

        pred_depth, pred_depth_var, pred_color, weights = raw2nerf_color(
            values_pts, z_vals, rays_d, device=self.device)
        pred_logits = torch.sum(weights[..., None] * logits_pts, -2)

        return pred_color, pred_depth, pred_depth_var, pred_logits, fine_latents, coarse_latents
    

    def frame_vis(self, idx, cur_gt_color, cur_gt_depth, 
        cur_gt_label, cur_gt_c2w, cur_c2w):
        self.is_BA = False
        with torch.no_grad():
            H, W, _ = cur_gt_color.shape

            _, _, refer_frames = self.set_target_refer_frames(
                cur_gt_color, cur_gt_depth, cur_gt_label, cur_gt_c2w, cur_c2w)

            rays_o, rays_d = get_all_rays(self.H, self.W, 
                                self.fx, self.fy, self.cx, self.cy, cur_c2w, self.device)
            rays_o = rays_o.flatten(0, 1)  # [N, 3]
            rays_d = rays_d.flatten(0, 1)
            cur_gt_label = cur_gt_label.flatten(0, 1)

            with torch.no_grad():
                det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(self.device) -
                    det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01
            
            z_vals = sample_along_rays(cur_gt_depth.flatten(0, 1), self.n_samples_ray, self.n_surface_ray,  
                                       far_bb, self.device)  # [n_pixels, n_samples]
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]

            features = self.encoder(refer_frames['gt_color'][-1, ...].unsqueeze(0))  # [1, n_class, C]

            n_pts = pts.shape[0]
            n_split = math.ceil(n_pts / self.n_pts_batch)        
            pred_color = []
            pred_depth = []  
            pred_label = []
            refer_c2w = refer_frames['est_c2w'][-1]
            w2c = torch.inverse(refer_c2w)
            for i in range(n_split):
                start = i * self.n_pts_batch
                end = min((i + 1) * self.n_pts_batch, n_pts) 

                pts_sp = pts[start:end, :, :]
                code = feature_matching(
                    self.H, self.W, self.K, pts_sp.flatten(0, 1), w2c, features[0], self.decoder.merge)
                code = code.reshape(pts_sp.shape[0], pts_sp.shape[1], -1)

                samples = {'rays_o': rays_o[start:end, :],  # [N, 3]
                        'rays_d': rays_d[start:end, :],  # [N, 3]
                        'gt_label': cur_gt_label[start:end],
                        'pts': pts_sp,
                        'z_vals': z_vals[start:end, :],
                        'features': code}  # [N, n_samples]
                
                color, depth, _, label, _, _ = self.renderer(samples)  # [N, n_samples, C], [N, n_samples, num_slots]
                pred_color.append(color)
                pred_depth.append(depth)
                pred_label.append(label)

            pred_color = torch.cat(pred_color, dim = 0)
            pred_depth = torch.cat(pred_depth, dim = 0)
            pred_label = torch.cat(pred_label, dim = 0)
            pred_label = torch.argmax(pred_label, dim=-1)

            def map_function(value):
                return self.class2label_dict.get(value, value)  # .get(value, default)
            
            v_map_function = np.vectorize(map_function)

            pred_color = pred_color.reshape(H, W, -1)
            pred_depth = pred_depth.reshape(H, W)
            pred_label = pred_label.reshape(H, W)

            gt_color_np = cur_gt_color.cpu().numpy()
            gt_depth_np = cur_gt_depth.cpu().numpy()
            gt_label_np = cur_gt_label.reshape(H, W).cpu().numpy()
            color_np = pred_color.detach().cpu().numpy()
            depth_np = pred_depth.detach().cpu().numpy()
            label_np = pred_label.detach().cpu().numpy()

            label_np = v_map_function(label_np)
            gt_label_np = v_map_function(gt_label_np)

            fig_plot(idx, 
                     self.out_dir, 
                     gt_color_np, color_np, 
                     gt_depth_np, depth_np,
                     gt_label_np, label_np)

    
    def set_decoder(self, target_frames):
        # for new frame, first check class id, if does not exist in current network, add a new one
        label_dict = target_frames['label_dict']
        new_decoders_list = []

        for obj_id in label_dict:
            if obj_id not in self.class2label_dict.keys():
                raise ValueError('Unknown semantic class', obj_id, 'Existed class:', self.class2label_dict.keys())
            
            if obj_id not in self.fine_decoders.keys():  # init new obj nerf decoder
                obj_nerf = tcnn.Network(n_input_dims=self.pe_dim + self.grid_dim,
                                    n_output_dims=self.hidden_dim + 1,
                                    network_config={
                                        "otype": "CutlassMLP",
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": self.hidden_dim,
                                        "n_hidden_layers": 1})
                print('Adding decoder:', obj_id)
                # obj_nerf = copy.deepcopy(self.decoder.coarse_fn).to(self.device)

                self.fine_decoders.update({obj_id: obj_nerf})
                self.exist_decoders.update({obj_id: 1})
            else:
                self.exist_decoders[obj_id] += 1

            if self.exist_decoders[obj_id] <= 4:
                new_decoders_list.append(obj_id)
        
        min_obj = min(self.exist_decoders, key=self.exist_decoders.get)
        if min_obj not in new_decoders_list and self.exist_decoders[min_obj] < 10:
            self.exist_decoders[min_obj] += 1
            new_decoders_list.append(min_obj)

        return new_decoders_list
    

    def decoder_init(self, decoder_idx, gt_color, gt_depth, 
        gt_label, gt_c2w, cur_c2w):
        
        # ResNet18 are used as encoder and fixed all the time
        features = self.encoder(gt_color.unsqueeze(0).unsqueeze(0)).clone().detach()

        # only update grid and fine decoder
        net_para_list = []
        net_para_list += list(self.decoder.parameters())
        for obj_id in decoder_idx:
            net_para_list += list(self.fine_decoders[obj_id].parameters())

        optimizer = torch.optim.Adam([{'params': net_para_list, 'lr': self.lr}])

        for iter_ in range(100):
            optimizer.zero_grad()

            # unified sample along the whole image
            rays_o_sp, rays_d_sp, samples = get_samples_by_uniq_class(
                0, self.H, 0, self.W, 300, self.H, self.W, 
                self.fx, self.fy, self.cx, self.cy, cur_c2w[:3, :3], cur_c2w[:3, 3],
                torch.cat((gt_color, gt_depth.unsqueeze(-1), gt_label.unsqueeze(-1)), -1), decoder_idx,
                self.device)  # [n_pixels, C]
            
            gt_color_sp = samples[:, :3]
            gt_depth_sp = samples[:, 3]
            gt_label_sp = samples[:, 4]

            with torch.no_grad():
                det_rays_o = rays_o_sp.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_sp.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(self.device) -
                    det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = far_bb >= gt_depth_sp
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

            # sample along the rays
            z_vals_sp = sample_along_rays(gt_depth_sp, self.n_samples_ray, self.n_surface_ray, far_bb, self.device)  # [n_pixels, n_samples]
            pts_sp = rays_o_sp[:, None, :] + rays_d_sp[:, None, :] * z_vals_sp[:, :, None]

            w2c = torch.inverse(cur_c2w).unsqueeze(0).clone().detach()
            code = feature_matching(
                self.H, self.W, self.K, pts_sp.flatten(0, 1), w2c, features[0], self.decoder.merge)
            code = code.reshape(pts_sp.shape[0], pts_sp.shape[1], -1)

            depth_mask = gt_depth_sp > 0.01
            mask = depth_mask * inside_mask #* mask_bp
            
            samples = {'gt_color': gt_color_sp.float(),  # [N, 3]
                        'gt_depth': gt_depth_sp.float(),  # [N]
                        'gt_label': gt_label_sp.to(torch.int64),
                        'rays_o': rays_o_sp.float(),  # [N, 3]
                        'rays_d': rays_d_sp.float(),  # [N, 3]
                        'pts': pts_sp.float(),  # [N, n_samples, 3]
                        'z_vals': z_vals_sp.float(),
                        'mask': mask.cpu().numpy(),
                        'features': code}
            
            pred_color, pred_depth, pred_depth_var, pred_logits, fine_latents, _ = self.renderer(samples)

            d_loss = self.compute_depth_loss(samples['gt_depth'], pred_depth)
            p_loss = self.compute_photometric_loss(samples['gt_color'], pred_color)
            l_loss = self.compute_label_loss(samples['gt_label'], pred_logits)
            smooth_loss = self.smoothness(sample_points=self.cfg['training']['smooth_pts'])
            fs_loss, opacity_loss = get_opacity_loss(samples['z_vals'], samples['gt_depth'], fine_latents[..., -1], self.cfg['training']['opacity_sigma']) 

            loss = self.lambda_p * p_loss + self.lambda_d * d_loss + self.lambda_l * l_loss +\
                 self.lambda_sm * smooth_loss + self.lambda_fs * fs_loss + self.lambda_opacity * opacity_loss
            
            loss.backward(retain_graph=False)
            optimizer.step()


    def optimize(self, n_iters, cur_idx, cur_gt_color, cur_gt_depth, 
        cur_gt_label, cur_gt_c2w, cur_c2w):
        
        target_frame_idx, target_frames, refer_frames = self.set_target_refer_frames(
            cur_gt_color, cur_gt_depth, cur_gt_label, cur_gt_c2w, cur_c2w)
        
        # ResNet18 is used as encoder and fixed all the time
        features = self.encoder(refer_frames['gt_color']).clone().detach()

        if cur_idx >= self.start_optimize_idx:
            self.is_BA = True
        else: 
            self.is_BA = False

        # incremental class-wise nerf adding and initialization
        new_decoder_idx = self.set_decoder(target_frames)
        if self.first_frame_optimized[0] == 1 and len(new_decoder_idx) > 0 and cur_idx > 50:
            cur_class = torch.unique(cur_gt_label)
            new_decoder_idx_ = new_decoder_idx.copy()
            # in case that new decoder does NOT in current frame
            for id_ in new_decoder_idx_:
                if id_ not in cur_class:
                    new_decoder_idx.remove(id_)
            if len(new_decoder_idx) > 0:
                print('New decoders initializing:', new_decoder_idx)
                self.decoder_init(new_decoder_idx, cur_gt_color, cur_gt_depth, 
                                  cur_gt_label, cur_gt_c2w, cur_c2w)

        # set optimizaer
        optimizer, quad_list, T_list = self.set_optimizer(target_frames)

        net_lr = self.lr
        cam_q_lr = self.BA_cam_lr * self.is_BA
        cam_T_lr = self.BA_cam_lr * self.is_BA
        lambda_p = self.lambda_p
        lambda_d = self.lambda_d
        lambda_l = self.lambda_l

        optimizer.param_groups[0]['lr'] = net_lr
        optimizer.param_groups[1]['lr'] = cam_q_lr
        optimizer.param_groups[2]['lr'] = cam_T_lr

        for iter_ in range(n_iters):           
            optimizer.zero_grad()
            # get samples from images and the sample along the rays
            samples = self.get_target_samples(target_frames, quad_list, T_list, 
                                              refer_frames=refer_frames, features=features)
            
            # render rgb-d values from nerfs
            pred_color, pred_depth, _, pred_logits, fine_latents, coarse_latents = self.renderer(samples)
            
            # loss
            d_loss = self.compute_depth_loss(samples['gt_depth'], pred_depth)
            p_loss = self.compute_photometric_loss(samples['gt_color'], pred_color)
            l_loss = self.compute_label_loss(samples['gt_label'], pred_logits)
            lt_loss = self.compute_latent_loss(coarse_latents, fine_latents)
            smooth_loss = self.smoothness(sample_points=self.cfg['training']['smooth_pts'])


            fs_loss, opacity_loss = get_opacity_loss(samples['z_vals'], samples['gt_depth'], fine_latents[..., -1], self.cfg['training']['opacity_sigma']) 
            
            if len(new_decoder_idx) > 0:
                if iter_ > n_iters // 2:
                    lambda_lt = 10
                else:
                    lambda_lt = 0
            else:
                lambda_lt = 10

            loss = lambda_p * p_loss + lambda_d * d_loss + lambda_l * l_loss + lambda_lt * lt_loss +\
                   self.lambda_sm * smooth_loss + self.lambda_fs * fs_loss + self.lambda_opacity * opacity_loss
            
            loss.backward(retain_graph=False)
            optimizer.step()
        
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                 [1, 4])).type(torch.float32).to(self.device)
        
        loss_camera_tensor = 0
        if self.is_BA:
            # updated camera poses in key_frame_list
            for i in range(1, len(target_frame_idx)-1):
                keyframe_id = target_frame_idx[i]

                camera_tensor = torch.cat((quad_list[i], T_list[i]), 0).clone().detach()
                c2w = get_camera_from_tensor(camera_tensor)
                c2w = torch.cat([c2w, bottom], dim=0)

                self.keyframe_dict[keyframe_id]['est_c2w'] = c2w.clone()
                target_frames['est_c2w'][i] = c2w.clone()

                # calculate all camera tensors error
                gt_camera_tensor = get_tensor_from_camera(self.keyframe_dict[keyframe_id]['gt_c2w']).to(self.device)
                loss_camera_tensor += torch.abs(gt_camera_tensor - camera_tensor).mean().item()
        
        camera_tensor = torch.cat((quad_list[-1], T_list[-1]), 0).clone().detach()
        c2w = get_camera_from_tensor(camera_tensor.detach())
        c2w = torch.cat([c2w, bottom], dim=0)
        cur_c2w = c2w.clone()
        target_frames['est_c2w'][-1] = cur_c2w.clone()

        gt_camera_tensor = get_tensor_from_camera(cur_gt_c2w).to(self.device)
        loss_camera_tensor = (loss_camera_tensor + \
            torch.abs(gt_camera_tensor-camera_tensor).mean().item()) /(len(target_frame_idx))
        
        fine_loss_dict = {'loss_camera_tensor': loss_camera_tensor,
                          'p_loss': p_loss,
                          'd_loss': d_loss,
                          'l_loss': l_loss,
                          'lt_loss': lt_loss,
                          'smooth_loss': smooth_loss}
        
        return cur_c2w, fine_loss_dict


    def run(self):
        '''
            Mapping process(Back End)

            First frame: first update the coarse and fine nerf in mapping, then start tracking
            Follow frames: first estimates coarse pose from tracking, the coarse pose will be refined
            together with fine nerf. Then refined pose will be used to update the coarse nerf.

            Coarse nerf: single nerf network with feature grid as conditional information
            Fine nerf: class-wise nerf networks with both feature grid and pixel features from reference frame
        '''
        scene = self.scene

        if self.verbose:
            print(Fore.YELLOW + "Current Scene: " + Style.RESET_ALL,  scene)
        cfg = self.cfg

        # frame 0 is used as the baseline of the coordinate
        idx, gt_color, gt_depth, gt_label, gt_c2w = self.back_dataset[0]
        self.gt_c2w_list[0] = gt_c2w.cpu()
        self.estimate_c2w_list[0] = gt_c2w.cpu()

        self.keyframe_list.append(idx)
        self.keyframe_dict.append({'gt_color': gt_color.cpu(),
                                   'gt_depth': gt_depth.cpu(), 
                                   'gt_label': gt_label, 
                                   'gt_c2w': gt_c2w.cpu(),
                                   'est_c2w': gt_c2w.cpu()})
        _, _, _, _, gt_c2w_1 = self.back_dataset[1]
        self.gt_c2w_list[1] = gt_c2w_1.cpu()
        self.estimate_c2w_list[1] = gt_c2w_1.cpu()

        # process start
        is_init = True
        prev_idx = 0
        time_elapsed_all = 0
        while (1):
            while True:
                # get current frame id from front end
                idx = self.front_idx[0].clone()

                # synchronization method selection
                if idx == self.n_img-1:
                    break
                if self.sync_method == 'strict':
                    if idx <= 1 or (idx % self.optimize_every_n_frames == 0 and idx > prev_idx):
                        break
                elif self.sync_method == 'loose':
                    if idx <= 1 or idx >= prev_idx + self.optimize_every_n_frames // 2:
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1)

            prev_idx = idx

            # current frame from tracking(front end) [H, W, C]
            _, gt_color, gt_depth, gt_label, gt_c2w = self.back_dataset[idx]
            cur_c2w = self.estimate_c2w_list[idx].to(self.device)

            if self.first_frame_optimized[0] == 1 and idx == 1:  # to prevent repetitive optimize frame 1
                self.ignore = True
            else:
                self.ignore = False

            if self.ignore == False:
                print(Fore.GREEN + "BACK END: Frame: " + Style.RESET_ALL, idx.item())

                if is_init:
                    outer_joint_iters = 1
                    n_iters = cfg['mapping']['n_iters_first']
                    print('Initialzing...')
                else:
                    outer_joint_iters = 2
                    n_iters = cfg['mapping']['n_iters']
                        
                iters = n_iters // outer_joint_iters

                t0 = time.perf_counter()
                for outer_joint_iter in range(outer_joint_iters):
                    if outer_joint_iter % 2 == 0:
                        self.mapping_mode = 'overlap'
                    else:
                        self.mapping_mode = 'global'

                    # main process
                    cur_c2w, loss = self.optimize(iters, idx, gt_color, gt_depth, gt_label, gt_c2w, cur_c2w)

                time_elapsed = time.perf_counter() - t0
                time_elapsed_all += time_elapsed

                # update the refined pose in estimated camera pose list
                if self.is_BA:
                    self.estimate_c2w_list[idx] = cur_c2w.cpu()

                # information print and save
                if self.verbose:
                    p_loss = loss['p_loss']
                    d_loss = loss['d_loss']
                    l_loss = loss['l_loss']
                    lt_loss = loss['lt_loss']
                    smooth_loss = loss['smooth_loss']
                    loss_camera_tensor = loss['loss_camera_tensor']
                    print(f'BACK END: rgb loss: {p_loss.item():.4f} ' +
                            f'psnr: {mse2psnr(p_loss).item():.4f} ' + 
                            f'depth loss: {d_loss.item():.4f} ' +
                            f'label loss: {l_loss.item():.4f} ' +
                            f'coarse2fine loss: {lt_loss.item():.4f} ' +
                            f'smooth: {smooth_loss.item():.4f}' +
                            f'| ATE: {loss_camera_tensor:.6f}\n' +
                            f'time elapsed: total {time_elapsed_all:.2f} current {time_elapsed:.2f}')
                    file = open(f'{self.out_dir}/output_back_fine.txt', 'a')
                    file.write(f'{scene} Current frame: {idx.item()} ' +
                               f'BACK END: rgh loss: {p_loss.item():.4f} ' +
                               f'psnr: {mse2psnr(p_loss).item():.4f} ' + 
                               f'depth loss: {d_loss.item():.4f} ' +
                               f'label loss: {l_loss.item():.4f} ' +
                               f'coarse2fine loss: {lt_loss.item():.4f} ' +
                               f'ATE: {loss_camera_tensor:.6f} ' +
                               f'time elapsed: total {time_elapsed_all} current {time_elapsed}\n')
                    file.close()

                # visiualization                    
                if (idx % self.vis_every == 0 or idx <= 1) and (not self.ignore) \
                   and (self.vis_every > 0) and self.verbose:
                    print('Visualizing...')
                    self.frame_vis(idx, gt_color, gt_depth, gt_label, 
                                   gt_c2w.clone(), cur_c2w)
                    
                # add new keyframe
                if (idx % self.choose_keyframe_every == 0 or (idx == self.n_img-2)) and \
                   (idx not in self.keyframe_list):
                    self.keyframe_list.append(idx.item())
                    self.keyframe_dict.append({'gt_color': gt_color.cpu(),
                                            'gt_depth': gt_depth.cpu(), 
                                            'gt_label': gt_label.cpu(), 
                                            'gt_c2w': gt_c2w.cpu(),
                                            'est_c2w': cur_c2w.cpu()})

                # meshing
                if (idx % self.mesh_every == 0 and idx > 0) and (not self.ignore) \
                   and (self.mesh_every > 0) and self.verbose:
                    print('Meshing...')
                    self.mesher.get_mesh(self.out_dir,
                                             self.encoder, 
                                             self.decoder, 
                                             self.fine_decoders,
                                             self.keyframe_dict, 
                                             self.estimate_c2w_list, 
                                             idx, 
                                             self.device, 
                                             color=self.cfg['meshing']['color'],
                                             label=self.cfg['meshing']['label'],
                                             element=self.cfg['meshing']['element'],
                                             clean_mesh=self.cfg['meshing']['clean_mesh'],
                                             get_mask_use_all_frames=self.cfg['meshing']['get_mask_use_all_frames'])


            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            # initialization finish, start tracking(front end)
            is_init = False
            self.first_frame_optimized[0] = 1
            self.back_idx[0] = idx

            # checkpoint
            checkpoint_kwargs = {'scene': scene,
                                 'idx': idx, 
                                 'fine_decoders': self.fine_decoders,
                                 'keyframe_dict': self.keyframe_dict,
                                 'keyframe_list': self.keyframe_list,
                                 'estimate_c2w_list': self.estimate_c2w_list,
                                 'gt_c2w_list': self.gt_c2w_list}
            
            if (self.checkpoint_every > 0 and (idx % self.checkpoint_every) == 0) and idx > 1:
                self.checkpoint.save(f'model_{idx.item()}.pt', **checkpoint_kwargs)
                if self.verbose:
                    print('Checkpoint saved.')                
            
            # for the last frame, save and quit
            if idx == self.n_img-1:
                if self.verbose:
                    print('Visualizing...')
                self.frame_vis(idx, gt_color, gt_depth, gt_label, gt_c2w.clone(), cur_c2w)
                
                if self.verbose:
                    num_decoder_params = sum(p.numel() for p in self.decoder.parameters())
                    for i in self.fine_decoders.keys():
                        num_decoder_params += sum(p.numel() for p in self.fine_decoders[i].parameters())
                    print(f'Number of decoders parameters: {num_decoder_params}')

                self.checkpoint.save('model.pt', **checkpoint_kwargs)
                print('Model saved, process end.')
                break