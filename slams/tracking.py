import time
import numpy as np
import copy
import torch
from torch.nn import functional as F
import torch.nn as nn
from colorama import Fore, Style
from torch.autograd import Variable

from models.encoder import ResNet

from tqdm import tqdm

from utils.common import get_camera_from_tensor, get_tensor_from_camera, get_quad_from_camera, get_rotation_from_quad
from utils.common import get_samples, sample_along_rays, feature_matching
from utils.common import mse2psnr, raw2nerf_color


class Tracker(object):
    def __init__(self, cfg, slam, device):
        self.cfg = slam.cfg

        self.device = device
        self.low_gpu_mem = slam.low_gpu_mem

        self.verbose = slam.verbose
        self.out_dir = slam.out_dir
        self.bound = slam.bound.to(self.device)

        self.encoder = ResNet().to(self.device)
        self.shared_decoder = slam.decoder

        self.seperate_LR = cfg['seperate_LR']
        self.cam_lr = cfg['tracking']['cam_lr']

        self.front_idx = slam.front_idx
        self.back_idx = slam.back_idx
        self.prev_back_idx = -1

        self.estimate_c2w_list = slam.estimate_c2w_list
        self.gt_c2w_list = slam.gt_c2w_list

        self.front_dataset = slam.front_dataset
        self.front_loader = slam.front_loader
        
        self.H = slam.H
        self.W = slam.W
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy

        self.K = torch.tensor([[self.fx,  0,     self.cx],
                               [0,     self.fy,  self.cy],
                               [0,        0,        1   ]]).to(self.device)

        self.optimize_every_n_frames = cfg['mapping']['optimize_every_n_frames']
        self.const_speed_assumption = cfg['const_speed_assumption']

        self.sync_method = cfg['sync_method']
        self.use_gt_camera = cfg['use_gt_camera']

        self.n_iters = cfg['tracking']['n_iters']
        self.n_pixels = cfg['tracking']['n_pixels']

        self.n_surface_ray = cfg['training']['n_surface_ray']
        self.n_samples_ray = cfg['training']['n_samples_ray']

        self.lambda_p = cfg['training']['lambda_color']
        self.lambda_d = cfg['training']['lambda_depth']
        self.lambda_l = cfg['training']['lambda_label']

        self.optimize_every_frame_flag = slam.optimize_every_frame


    def update_para_from_mapping(self):
        if self.back_idx[0] != self.prev_back_idx:
            if self.verbose:
                print('FRONT END: update the parameters from Back End')
            # only clone coarse nerf from mapping(back end)
            self.decoder = copy.deepcopy(self.shared_decoder).to(self.device)
            
            self.prev_back_idx = self.back_idx[0].clone()

    def compute_photometric_loss(self, gt_color, pred_color, mask):
        loss = ((gt_color[mask, :] - pred_color[mask, :])**2).mean()
        return loss

    def compute_depth_loss(self, gt_depth, pred_depth, pred_depth_var, mask):
        loss = (torch.abs(gt_depth-pred_depth) /
                torch.sqrt(pred_depth_var+1e-10))[mask].mean()
        return loss
    
    def compute_label_loss(self, gt_label, pred_logits, mask):
        loss = F.cross_entropy(pred_logits[mask, :], gt_label[mask])
        return loss
    
    def random_select(self, n, k):
        """
        Random select k values from 0..n.

        """
        if n == 0:
            return [0]
        else:
            return list(np.random.choice(np.array(range(n)), size=k, replace=False))

    def set_optimizer(self, c2w, idx):
        lr = self.cam_lr

        quad = get_quad_from_camera(c2w).clone().detach()
        quad = Variable(quad.to(self.device), requires_grad=True)
        cam_para_list_quad = [quad]

        T = c2w[:3, 3].clone().detach()
        T = Variable(T.to(self.device), requires_grad=True)
        cam_para_list_T = [T]

        if self.seperate_LR:
            optimizer = torch.optim.Adam([{'params': cam_para_list_T, 'lr': lr*0.2},
                                          {'params': cam_para_list_quad, 'lr': lr}])
        else:
            optimizer = torch.optim.Adam([{'params': cam_para_list_T, 'lr': lr},
                                          {'params': cam_para_list_quad, 'lr': lr}])

        return optimizer, quad, T
    
    def get_target_samples(self, cur_frames, refer_frames, features):
        gt_color = cur_frames['gt_color']
        gt_depth = cur_frames['gt_depth']
        gt_label = cur_frames['gt_label']

        R = get_rotation_from_quad(cur_frames['est_quad'])
        T = cur_frames['est_T']

        # unified sample along the whole image
        rays_o_sp, rays_d_sp, samples = get_samples(
            20, self.H-20, 20, self.W-20, self.n_pixels, self.H, self.W, 
            self.fx, self.fy, self.cx, self.cy, R, T,
            torch.cat((gt_color, gt_depth.unsqueeze(-1), gt_label.unsqueeze(-1)), -1), 
            self.device)  # [n_pixels, C]
        
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

        w2c = refer_frames['est_w2c'].clone().detach()
        code = feature_matching(
            self.H, self.W, self.K, pts_sp.flatten(0, 1), w2c, features[0], self.decoder.merge)
        code = code.reshape(pts_sp.shape[0], pts_sp.shape[1], -1)

        front_mask = torch.where(z_vals_sp < (gt_depth_sp[:, None] * 0.95), torch.ones_like(z_vals_sp), torch.zeros_like(z_vals_sp))
        back_mask = torch.where(z_vals_sp > (gt_depth_sp[:, None] * 1.05), torch.ones_like(z_vals_sp), torch.zeros_like(z_vals_sp))
        depth_mask = torch.where(gt_depth_sp[:, None] > 0.0, torch.ones_like(gt_depth_sp[:, None]), torch.zeros_like(gt_depth_sp[:, None]))
        trunc_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask
        code = code * trunc_mask[..., None]


        depth_mask = gt_depth_sp > 0.01
        mask = depth_mask * inside_mask
        
        samples_all = {'gt_color': gt_color_sp.float(),  # [N, 3]
                       'gt_depth': gt_depth_sp.float(),  # [N]
                       'gt_label': gt_label_sp.to(torch.int64),
                       'rays_o': rays_o_sp.float(),  # [N, 3]
                       'rays_d': rays_d_sp.float(),  # [N, 3]
                       'pts': pts_sp.float(),  # [N, n_samples, 3]
                       'z_vals': z_vals_sp.float(),
                       'mask': mask.cpu().numpy(),
                       'features': code}
        return samples_all
    
    def renderer(self, samples):
        pts = samples['pts'].flatten(0, 1)  # [n_pts, n_sample, 3]
        pts = (pts - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])

        rays_d = samples['rays_d']  # [n_pts, 3]
        z_vals = samples['z_vals']  # [n_pts, n_sample]
        n_pts, n_samples  = z_vals.shape

        # prepare features
        pixel_pts = samples['features']
        pixel_pts = pixel_pts.flatten(0, 1)

        pe, grid_pts = self.decoder.pe_fn(pts)

        latents = self.decoder.coarse_fn(pe, features=grid_pts)
        color_pts, logits_pts = self.decoder.out_fn(pe, torch.cat((latents[:, 1:], pixel_pts),-1))

        values_pts = torch.cat((color_pts, latents[:, 0:1]), -1)

        values_pts = values_pts.reshape(n_pts, n_samples, -1)
        logits_pts = logits_pts.reshape(n_pts, n_samples, -1)

        pred_depth, pred_depth_var, pred_color, weights = raw2nerf_color(
            values_pts, z_vals, rays_d, device=self.device)
        pred_logits = torch.sum(weights[..., None] * logits_pts, -2)  # (N_rays, 3)

        return pred_color, pred_depth, pred_depth_var, pred_logits
    
    def pose_init(self, idx):
        pre_c2w = self.estimate_c2w_list[idx-1].to(self.device)

        if self.const_speed_assumption and idx > 2:
            pre_c2w = pre_c2w.float()
            delta = pre_c2w@self.estimate_c2w_list[idx-2].to(self.device).float().inverse()
            estimated_new_cam_c2w = delta@pre_c2w
        else:
            estimated_new_cam_c2w = pre_c2w
        
        est_c2w = estimated_new_cam_c2w.detach()
        return est_c2w

    def run(self):
        '''
            Tracking process(Front End)

            Tracking will start only after first frame being optimized in mapping
            Coarse nerf: single nerf network with feature grid as conditional information

            Tracking will fix the coarse nerf and only updates camera pose.
            While sampling, only use points that have been seen by previous frames
        '''
        device = self.device
        time_elapsed_all = 0.

        if self.verbose:
            pbar = self.front_loader
        else:
            pbar = tqdm(self.front_loader)

        for idx, gt_color, gt_depth, gt_label, gt_c2w in pbar:  # [B, H, W, C]
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx.squeeze(0)
            gt_depth = gt_depth.squeeze(0)
            gt_color = gt_color.squeeze(0)
            gt_label = gt_label.squeeze(0)
            gt_c2w = gt_c2w.squeeze(0)  # [b,4,4]->[4,4]

            optimize_every_n_frames = self.optimize_every_n_frames
            # synchronization method selection
            if self.sync_method == 'strict':
                if idx > 2 and (idx % optimize_every_n_frames == 1 or \
                optimize_every_n_frames == 1):
                    while self.back_idx[0] != idx-1:
                        time.sleep(0.1)
            elif self.sync_method == 'loose':
                while self.back_idx[0] < (idx - optimize_every_n_frames - \
                optimize_every_n_frames // 2):
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                pass

            # for the first frame or choose to use gt pose, skip
            if idx <= 1 or self.use_gt_camera:
                c2w = gt_c2w
                pre_color = gt_color.clone().detach()
                refer_color = pre_color
                if idx == 1:
                    refer_c2w = self.estimate_c2w_list[idx]
                    refer_w2c = torch.inverse(refer_c2w).to(self.device)
            else:
                print(Fore.MAGENTA + "FRONT END: Frame: " + Style.RESET_ALL, idx.item())
                self.update_para_from_mapping()

                t0 = time.perf_counter()
                current_min_loss = 10000000000. 
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)      
                                            
                # initialize estimated pose based on previous frame and movement 
                if idx - 1 % optimize_every_n_frames == 0 or optimize_every_n_frames == 1:
                    refer_color = pre_color
                    refer_c2w = self.estimate_c2w_list[idx - 1]
                    refer_w2c = torch.inverse(refer_c2w).to(self.device)

                refer_frames = {'gt_color': torch.stack((refer_color, gt_color), 0).to(self.device)}

                with torch.no_grad():
                    features = self.encoder(refer_frames['gt_color'].unsqueeze(0))

                # set optimizer: in tracking, only update camera pose
                est_c2w = self.pose_init(idx)
                camera_init = get_tensor_from_camera(est_c2w).to(device) 
                gt_camera_tensor = get_tensor_from_camera(gt_c2w).to(device) 
                camera_error_init = torch.abs(gt_camera_tensor- \
                                            camera_init).mean().item()
                optimizer, quad, T = self.set_optimizer(est_c2w, idx)

                target_frames = {'gt_color': gt_color,  # [H, W, 3]
                                'gt_depth': gt_depth,
                                'gt_label': gt_label,
                                'gt_c2w': gt_c2w,
                                'est_quad': quad,
                                'est_T': T}

                for iter in range(self.n_iters):
                    optimizer.zero_grad()

                    R = get_rotation_from_quad(quad)
                    cur_c2w = torch.cat([torch.cat((R, T[:, None]), -1), bottom], dim=0)
                    cur_w2c = torch.inverse(cur_c2w).to(self.device)
                    refer_frames['est_w2c'] = torch.stack((refer_w2c, cur_w2c), 0).to(self.device)

                    # quad will be updated for each iteration
                    samples = self.get_target_samples(target_frames, refer_frames, features)    

                    pred_color, pred_depth, pred_depth_var, pred_logits = self.renderer(samples) 

                    p_loss = self.compute_photometric_loss(samples['gt_color'], pred_color, samples['mask'])
                    d_loss = self.compute_depth_loss(samples['gt_depth'], pred_depth, pred_depth_var, samples['mask'])
                    l_loss = self.compute_label_loss(samples['gt_label'], pred_logits, samples['mask'])
                    loss = self.lambda_p * p_loss + self.lambda_d * d_loss + self.lambda_l * l_loss

                    if loss < current_min_loss:
                        with torch.no_grad():
                            current_min_loss = loss
                            current_min_p_loss = p_loss
                            current_min_d_loss = d_loss
                            candidate_cam_tensor = torch.cat((quad, T), 0)

                    loss.backward(retain_graph=False)
                    optimizer.step()
                    optimizer.zero_grad()
                
                time_elapsed = time.perf_counter() - t0
                time_elapsed_all += time_elapsed
                
                # turn 3*4 matrix to 4*4              
                c2w = get_camera_from_tensor(candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
                
                loss_camera_tensor = torch.abs(gt_camera_tensor- \
                                            candidate_cam_tensor).mean().item()    

                pre_color = gt_color.clone().detach()

                if self.verbose:
                    print(f'FRONT END: rgb loss: {current_min_p_loss.item():.4f} ' +
                            f'psnr: {mse2psnr(current_min_p_loss).item():.4f} ' +
                            f'depth loss: {current_min_d_loss.item():.4f} ' +
                            f'| ATE: {camera_error_init:.6f} --> {loss_camera_tensor:.6f}\n' +
                            f'time elapsed: total {time_elapsed_all:.2f} current {time_elapsed:.2f}')
                    file = open(f'{self.out_dir}/output_front.txt', 'a')
                    file.write(f'Current frame: {idx.item()} ' +
                            f'FRONT END: rgb loss: {current_min_p_loss.item():.4f} ' +
                            f'psnr: {mse2psnr(current_min_p_loss).item():.4f} ' + 
                            f'depth loss: {current_min_d_loss.item():.4f} ' +
                            f'ATE before: {camera_error_init:.6f} after: {loss_camera_tensor:.6f} ' +
                            f'time elapsed: total {time_elapsed_all} current {time_elapsed}\n')
                    file.close()   

            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            
            if idx > 1:
                self.front_idx[0] = idx

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

