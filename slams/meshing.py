import os
import torch
import numpy as np

import numpy as np
import open3d as o3d
import skimage
import torch
import torch.nn.functional as F
import trimesh
from packaging import version
import cv2
import math

from utils.common import get_all_rays, sample_along_rays, raw2nerf_color

class Mesher(object):
    def __init__(self, cfg, slam,
                 ray_batch_size=2048, 
                 device='cuda:0'):

        self.device = device
        self.out_dir = slam.out_dir
        self.verbose = slam.verbose

        self.points_batch_size = cfg['meshing']['points_batch_size']
        self.ray_batch_size = ray_batch_size

        self.scale = cfg['scale']
        self.resolution = cfg['meshing']['resolution']
        self.level_set = cfg['meshing']['level_set']
        self.clean_mesh_bound_scale = cfg['meshing']['clean_mesh_bound_scale']
        self.remove_small_geometry_threshold = cfg['meshing']['remove_small_geometry_threshold']
        self.get_largest_components = cfg['meshing']['get_largest_components']
        self.depth_test = cfg['meshing']['depth_test']
        self.use_est_depth = cfg['meshing']['use_est_depth']
        
        self.bound = slam.bound.clone().to(self.device)
        self.marching_cubes_bound = torch.from_numpy(
            np.array(cfg['back_end']['marching_cubes_bound']) * self.scale)

        self.H = slam.H
        self.W = slam.W
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.K = torch.tensor([[self.fx,  0,     self.cx],
                               [0,     self.fy,  self.cy],
                               [0,        0,        1   ]]).to(self.device)

        self.pixel_dim = cfg['model']['pixel_dim']
        self.hidden_dim = cfg['model']['hidden_dim']

        self.n_surface_ray = cfg['training']['n_surface_ray']
        self.n_samples_ray = cfg['training']['n_samples_ray']

        if not slam.v_map_function == None:
            self.v_map_function = slam.v_map_function


    def depth_render(self, cur_gt_depth, cur_c2w, decoder):
        H, W = cur_gt_depth.shape
        depth_flat = cur_gt_depth.flatten(0, 1)
        zero_mask = depth_flat <= 0

        if torch.sum(zero_mask) > 1:
            rays_o, rays_d = get_all_rays(self.H, self.W, 
                                self.fx, self.fy, self.cx, self.cy, cur_c2w, self.device)
            rays_o = rays_o.flatten(0, 1)  # [N, 3]
            rays_o = rays_o[zero_mask]
            rays_d = rays_d.flatten(0, 1)
            rays_d = rays_d[zero_mask]

            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(self.device) -
                det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01
            
            z_vals = sample_along_rays(depth_flat[zero_mask], self.n_samples_ray, self.n_surface_ray,  
                                        far_bb, self.device) 
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]

            n_pts = pts.shape[0]
            n_split = math.ceil(n_pts / 5000)        
            pred_depth = []  
            for i in range(n_split):
                start = i * 5000
                end = min((i + 1) * 5000, n_pts) 

                pts_sp = pts[start:end, :, :]
                rays_d_sp = rays_d[start:end, :]
                z_sp = z_vals[start:end, :]
                
                n_sp, n_samples, _  = pts_sp.shape
                pts_sp = pts_sp.flatten(0, 1)  # [B*N*n_samples, C]
                pts_sp = (pts_sp - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])

                pe, grid_pts, _ = decoder.pe_fn(pts_sp)

                coarse_latents = decoder.coarse_fn(pe, features=grid_pts)

                values_pts = torch.zeros(coarse_latents.shape[0], 4).to(self.device)
                values_pts[:, -1] = coarse_latents[:, 0]

                values_pts = values_pts.reshape(n_sp, n_samples, -1)

                depth, _, _, _ = raw2nerf_color(
                    values_pts, z_sp, rays_d_sp, device=self.device)
                
                pred_depth.append(depth)

            pred_depth = torch.cat(pred_depth, dim = 0)
        
            depth_flat[zero_mask] = pred_depth

        return depth_flat.reshape(H, W)



    def point_masks(self, input_points, keyframe_dict, estimate_c2w_list, 
                    idx, device, decoders, get_mask_use_all_frames=False):
        """
        Split the input points into seen, unseen, and forcast,
        according to the estimated camera pose and depth image.

        Args:
            input_points (tensor): input points.
            keyframe_dict (list): list of keyframe info dictionary.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current frame index.
            device (str): device name to compute on.

        Returns:
            seen_mask (tensor): the mask for seen area.
            forecast_mask (tensor): the mask for forecast area.
            unseen_mask (tensor): the mask for unseen area.
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if not isinstance(input_points, torch.Tensor):
            input_points = torch.from_numpy(input_points)
        input_points = input_points.clone().detach()
        seen_mask_list = []
        forecast_mask_list = []
        unseen_mask_list = []
        
        for i in range(math.ceil(input_points.shape[0] / self.points_batch_size)):
            start = i * self.points_batch_size
            end = min((i + 1) * self.points_batch_size, input_points.shape[0])

            points = input_points[start:end, :].to(device).float()

            # should divide the points into three parts, seen and forecast and unseen
            # seen: union of all the points in the viewing frustum of keyframes
            # forecast: union of all the points in the extended edge of the viewing frustum of keyframes
            # unseen: all the other points

            seen_mask = torch.zeros((points.shape[0])).bool().to(device)
            forecast_mask = torch.zeros((points.shape[0])).bool().to(device)
            if get_mask_use_all_frames:
                for i in range(0, idx + 1, 1):
                    c2w = estimate_c2w_list[i].cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    w2c = torch.from_numpy(w2c).to(device).float()
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()  # (N, 4)
                    # (N, 4, 1)=(4,4)*(N, 4, 1)
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)

                    K = torch.from_numpy(
                        np.array([[fx, .0, cx], [.0, fy, cy],
                                  [.0, .0, 1.0]]).reshape(3, 3)).to(device)
                    cam_cord[:, 0] *= -1
                    uv = K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    # forecast
                    cur_mask_forecast = cur_mask_forecast.reshape(-1)
                    # seen
                    cur_mask_seen = cur_mask_seen.reshape(-1)

                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast
            else:
                for keyframe in keyframe_dict:
                    c2w = keyframe['est_c2w'].to(self.device)

                    w2c = torch.inverse(c2w).to(device).float()
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]

                    cam_cord[:, 0] *= -1
                    uv = self.K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    if self.depth_test:
                        if self.use_est_depth:
                            gt_depth = self.depth_render(keyframe['gt_depth'], c2w, decoders)
                        else:
                            gt_depth = keyframe['gt_depth']

                        gt_depth = gt_depth.to(device).reshape(1, 1, H, W)
                        vgrid = uv.reshape(1, 1, -1, 2)
                        # normalized to [-1, 1]
                        vgrid[..., 0] = (vgrid[..., 0] / (W-1) * 2.0 - 1.0)
                        vgrid[..., 1] = (vgrid[..., 1] / (H-1) * 2.0 - 1.0)
                        depth_sample = F.grid_sample(
                            gt_depth, vgrid, padding_mode='zeros', align_corners=True)
                        depth_sample = depth_sample.reshape(-1)
                        max_depth = torch.max(depth_sample)
                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                                                        2].reshape(-1)
                        cur_mask_forecast[cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth
                        # seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - cam_cord[cur_mask_seen, 2].reshape(-1)

                        cur_mask_seen[cur_mask_seen.clone()] &= \
                            (proj_depth_seen < depth_sample[cur_mask_seen] + 0.1) \
                            & (depth_sample[cur_mask_seen] - 2.5 < proj_depth_seen)
                    else:
                        max_depth = torch.max(keyframe['gt_depth'])*1.2

                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                                                        2].reshape(-1)
                        cur_mask_forecast[
                            cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth

                        # seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - \
                            cam_cord[cur_mask_seen, 2].reshape(-1)
                        cur_mask_seen[cur_mask_seen.clone(
                        )] &= proj_depth_seen < max_depth

                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast

            forecast_mask &= ~seen_mask
            unseen_mask = ~(seen_mask | forecast_mask)

            seen_mask = seen_mask.cpu().numpy()
            forecast_mask = forecast_mask.cpu().numpy()
            unseen_mask = unseen_mask.cpu().numpy()

            seen_mask_list.append(seen_mask)
            forecast_mask_list.append(forecast_mask)
            unseen_mask_list.append(unseen_mask)

        seen_mask = np.concatenate(seen_mask_list, axis=0)
        forecast_mask = np.concatenate(forecast_mask_list, axis=0)
        unseen_mask = np.concatenate(unseen_mask_list, axis=0)

        return seen_mask, forecast_mask, unseen_mask


    def get_2d_feature(self, points, keyframe_dict, device, color, encoder=None, decoders=None):
        """
        Split the input points into seen, unseen, and forcast,
        according to the estimated camera pose and depth image.

        Args:
            input_points (tensor): input points.
            keyframe_dict (list): list of keyframe info dictionary.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current frame index.
            device (str): device name to compute on.

        Returns:
            seen_mask (tensor): the mask for seen area.
            forecast_mask (tensor): the mask for forecast area.
            unseen_mask (tensor): the mask for unseen area.
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        pixel_pts = torch.zeros(points.shape[0], self.hidden_dim).to(self.device)
        label_pts = torch.zeros(points.shape[0]).to(self.device)
        count_pts = torch.zeros(points.shape[0]).to(self.device)

        for keyframe in keyframe_dict:
            c2w = keyframe['est_c2w'].to(self.device)
            color = keyframe['gt_color'].to(self.device)
            label = keyframe['gt_label'].to(self.device)
            depth = keyframe['gt_depth'].to(self.device)

            w2c = torch.inverse(c2w).to(device).float()
            ones = torch.ones_like(
                points[:, 0]).reshape(-1, 1).to(device)
            homo_points = torch.cat([points, ones], dim=1).reshape(
                -1, 4, 1).to(device).float()
            cam_cord_homo = w2c @ homo_points
            cam_cord = cam_cord_homo[:, :3]

            cam_cord[:, 0] *= -1
            uv = self.K.float() @ cam_cord.float()
            z = uv[:, -1:] + 1e-8
            uv = uv[:, :2] / z
            uv = uv.float()
            edge = 0
            cur_mask_seen = (uv[:, 0] < W - edge) & (
                uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
            cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)
            cur_mask_seen = cur_mask_seen.reshape(-1)

            uv_ = uv[cur_mask_seen, :, 0]
            p = points[cur_mask_seen, :]
            if uv_.numel() != 0:
                uv_ = torch.round(uv_).to(torch.int64)
                uv_[:, 0] = uv_[:, 0].clamp(0, W - 1)
                uv_[:, 1] = uv_[:, 1].clamp(0, H - 1)

                label_seen = label[uv_[:, 1], uv_[:, 0]]
                depth_seen = depth[uv_[:, 1], uv_[:, 0]]

                depth_proj = -z[cur_mask_seen].squeeze()

                front_mask = torch.where(depth_proj < (depth_seen * 0.95), torch.ones_like(depth_seen), torch.zeros_like(depth_seen))
                back_mask = torch.where(depth_proj > (depth_seen * 1.05), torch.ones_like(depth_seen), torch.zeros_like(depth_seen))
                trunc_mask = (1.0 - front_mask) * (1.0 - back_mask)

                features = encoder(color.unsqueeze(0).unsqueeze(0))
                features = F.interpolate(features[0], size=[self.H, self.W], mode='bilinear', align_corners=True)
                ft = features[0, :, uv_[:, 1], uv_[:, 0]].permute(1, 0).unsqueeze(0)

                # refer_view = c2w[:3, 2].unsqueeze(0)
                refer_o = c2w[:3, 3].unsqueeze(0)
                refer_p = p[None, :, :].clone() - refer_o[:, None, :]

                code_pts = decoders.merge(refer_p, refer_o, ft)
                code_pts = code_pts * trunc_mask[..., None]

                count_ = torch.ones_like(trunc_mask) * trunc_mask

                count_pts[cur_mask_seen] += count_
                pixel_pts[cur_mask_seen, :] += code_pts.float()
                label_pts[cur_mask_seen] = label_seen.float()

        pixel_pts[count_pts > 0, :] = pixel_pts[count_pts > 0, :] / count_pts[count_pts > 0, None]

        return pixel_pts, label_pts
    

    def get_bound_from_frames(self, keyframe_dict, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_dict:
            c2w = keyframe['est_c2w'].cpu().numpy()
            # convert to open3d camera pose
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            depth = keyframe['gt_depth'].cpu().numpy()
            color = keyframe['gt_color'].cpu().numpy()

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.clean_mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.clean_mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh
    
    def fine_fn(self, fine_decoders, pes, classes=None, features=None):
        n_class = torch.unique(classes, sorted=True).cpu().int().numpy()
        latents = torch.zeros(pes.shape[0], self.hidden_dim+1, device=pes.device)
        for i in n_class:
            if i not in fine_decoders.keys():
                raise ValueError('Fine decoders does NOT have class', i)
            index = (classes[:] == i).cpu().numpy()
            if index.sum() > 1:
                pe = pes[index, :]
                feature = features[index, :]
                latents[index, :] = fine_decoders[i](torch.cat((pe, feature), -1)).float()
        return latents


    def eval_points(self, pts, pixel_pts=None, gt_label_pts=None, decoders=None, fine_decoders=None, stage='fine', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): point coordinates.
            decoders (nn.module decoders): decoders.
            c (dicts, optional): feature grids. Defaults to None.
            stage (str, optional): query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        bound = self.bound

        # mask for points out of bound
        mask_x = (pts[:, 0] < bound[0, 1]) & (pts[:, 0] > bound[0, 0])
        mask_y = (pts[:, 1] < bound[1, 1]) & (pts[:, 1] > bound[1, 0])
        mask_z = (pts[:, 2] < bound[2, 1]) & (pts[:, 2] > bound[2, 0])
        mask = mask_x & mask_y & mask_z

        p = (pts.clone() - bound[:, 0]) / (bound[:, 1] - bound[:, 0])

        pe, grid_pts = decoders.pe_fn(p)

        if stage == 'coarse':
            latents = decoders.coarse_fn(pe, features=grid_pts)
            color_pts, logits_pts = decoders.out_fn(pe, torch.cat((latents[:, 1:], pixel_pts), -1))
            values_pts = torch.cat((color_pts, latents[:, 0:1]), -1)
            values_pts[..., -1] = values_pts[..., -1]+ 0.0

            values_pts[~mask, 3] = -100
            label_pts = None
        else:
            latents = self.fine_fn(fine_decoders, pe, classes=gt_label_pts, features=grid_pts)
            color_pts, logits_pts = decoders.out_fn(pe, torch.cat((latents[:, 1:], pixel_pts), -1))
            values_pts = torch.cat((color_pts, latents[:, 0:1]), -1)
            
            values_pts[..., -1] = values_pts[..., -1] + 0.0

            values_pts[~mask, 3] = -100

            label_pts = torch.argmax(logits_pts, dim=-1)
            label_pts[~mask] = -1

        if False:
            if stage == 'coarse':
                latents = decoders.coarse_fn(pe, features=grid_pts)
                color_pts, logits_pts = decoders.out_fn(pe, torch.cat((latents[:, 1:], pixel_pts), -1))
                values_pts = torch.cat((color_pts, latents[:, 0:1]), -1)
                values_pts = values_pts.squeeze(0)

                values_pts[~mask, 3] = 100
                label_pts = None
            else:
                latents = decoders.coarse_fn(pe, features=grid_pts)
                _, logits_pts = decoders.out_fn(pe, torch.cat((latents[:, 1:], pixel_pts), -1))
                label_pts = torch.argmax(logits_pts, dim=-1)

                latents = self.fine_fn(fine_decoders, pe, classes=label_pts, features=grid_pts)
                color_pts, logits_pts = decoders.out_fn(pe, torch.cat((latents[:, 1:], pixel_pts), -1))
                values_pts = torch.cat((color_pts, latents[:, 0:1]), -1)
                values_pts = values_pts.squeeze(0)

                values_pts[~mask, 3] = 100

                label_pts = torch.argmax(logits_pts, dim=-1)
                label_pts[~mask] = -1

        return values_pts, label_pts


    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound

        padding = 0.05
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding,
                        resolution)
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding,
                        resolution)
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding,
                        resolution)

        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

        return {"grid_points": grid_points, "xyz": [x, y, z]}


    def get_mesh(self,
                 mesh_out_file,
                 encoder,
                 decoders,
                 fine_decoders,
                 keyframe_dict,
                 estimate_c2w_list,
                 idx,
                 device='cuda:0',
                 show_forecast=False,
                 color=True,
                 label=False,
                 element=False,
                 clean_mesh=True,
                 get_mask_use_all_frames=False):
        """
        Extract mesh from scene representation and save mesh to file.

        Args:
            mesh_out_file (str): output mesh filename.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
            keyframe_dict (list):  list of keyframe info.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current processed camera ID.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.
            show_forecast (bool, optional): show forecast. Defaults to False.
            color (bool, optional): whether to extract colored mesh. Defaults to True.
            clean_mesh (bool, optional): whether to clean the output mesh 
                                        (remove outliers outside the convexhull and small geometry noise). 
                                        Defaults to True.
            get_mask_use_all_frames (bool, optional): 
                whether to use all frames or just keyframes when getting the seen/unseen mask. Defaults to False.
        """
        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)  # [resolution^3, 3], flattened
            points = grid['grid_points']
            points = points.to(device)

            if show_forecast:
                # instaed of predicting the whole space (which is feasible but with bad result),
                # only predict a part of unseen aera: forecast
                seen_mask, forecast_mask, unseen_mask = self.point_masks(
                    points, keyframe_dict, estimate_c2w_list, idx, device, decoders,
                    get_mask_use_all_frames=get_mask_use_all_frames)

                # predict the density or occpuancy for each point
                # use fine decoder to deal with seen points
                seen_points = points[seen_mask]
                z_seen = []
                for i in range(math.ceil(seen_points.shape[0] / self.points_batch_size)):
                    start = i * self.points_batch_size
                    end = min((i + 1) * self.points_batch_size, seen_points.shape[0])
                    pts = seen_points[start:end, :]
                    pixel_pts, label_pts = self.get_2d_feature(pts, keyframe_dict, device, color, encoder=encoder, decoders=decoders)
                    values, _ = self.eval_points(pts, pixel_pts=pixel_pts, gt_label_pts=label_pts, 
                                                decoders=decoders, fine_decoders=fine_decoders,
                                                stage='fine', device=device)
                    z_seen.append(values[:, 3].cpu().numpy())
                z_seen = np.concatenate(z_seen, axis=0)

                # use coarse decoder to deal with forecast and unseen points
                forecast_points = points[forecast_mask]
                z_forecast = []
                for i in range(math.ceil(forecast_points.shape[0] / self.points_batch_size)):
                    start = i * self.points_batch_size
                    end = min((i + 1) * self.points_batch_size, forecast_points.shape[0])
                    pts = forecast_points[start:end, :]
                    pixel_pts, label_pts = self.get_2d_feature(pts, keyframe_dict, device, color, encoder=encoder, decoders=decoders)
                    values, _ = self.eval_points(pts, pixel_pts=pixel_pts,
                                                decoders=decoders, fine_decoders=fine_decoders,
                                                stage='coarse', device=device)
                    z_forecast.append(values[:, 3].cpu().numpy())
                z_forecast = np.concatenate(z_forecast, axis=0)
                # z_forecast += 0.2

                all_z = np.zeros(points.shape[0])
                all_z[seen_mask] = z_seen
                all_z[forecast_mask] = z_forecast
                all_z[unseen_mask] = -100
            else:
                seen_points = points
                all_z = []
                all_labels = []
                for i in range(math.ceil(seen_points.shape[0] / self.points_batch_size)):
                    start = i * self.points_batch_size
                    end = min((i + 1) * self.points_batch_size, seen_points.shape[0])
                    pts = seen_points[start:end, :]
                    pixel_pts, label_pts = self.get_2d_feature(pts, keyframe_dict, device, color, encoder=encoder, decoders=decoders)
                    values, labels = self.eval_points(pts, pixel_pts=pixel_pts, gt_label_pts=label_pts, 
                                                decoders=decoders, fine_decoders=fine_decoders,
                                                stage='fine', device=device)
                    all_z.append(values[:, 3].cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                all_z = np.concatenate(all_z, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

            all_z = all_z.astype(np.float32)

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=all_z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                grid['xyz'][1][2] - grid['xyz'][1][1],
                                grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=all_z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                grid['xyz'][1][2] - grid['xyz'][1][1],
                                grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print(
                    'marching_cubes error. Possibly no surface extracted from the level set.'
                )
                return

            # convert back to world coordinates
            vertices = verts + np.array(
                [grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            # remove excess spots
            if clean_mesh:
                if show_forecast:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=faces,
                                        process=False)
                    mesh_bound = self.get_bound_from_frames(
                        keyframe_dict, self.scale)
                    contain_mask = []
                    for i, pnts in enumerate(np.array_split(points, self.points_batch_size, axis=0)):
                        contain_mask.append(mesh_bound.contains(pnts))
                    contain_mask = np.concatenate(contain_mask, axis=0)
                    not_contain_mask = ~contain_mask
                    face_mask = not_contain_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)
                else:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=faces,
                                        process=False)
                    seen_mask, forecast_mask, unseen_mask = self.point_masks(
                        points, keyframe_dict, estimate_c2w_list, idx, device, decoders,
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    unseen_mask = ~seen_mask
                    face_mask = unseen_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)

                # get connected components
                components = mesh.split(only_watertight=False)
                if self.get_largest_components:
                    areas = np.array([c.area for c in components], dtype=np.float)
                    mesh = components[areas.argmax()]
                else:
                    new_components = []
                    for comp in components:
                        if comp.area > self.remove_small_geometry_threshold * self.scale * self.scale:
                            new_components.append(comp)
                    mesh = trimesh.util.concatenate(new_components)
                vertices = mesh.vertices
                faces = mesh.faces

            if color:
                # color is extracted by passing the coordinates of mesh vertices through the network
                points = torch.from_numpy(vertices).to(self.device).float()
                z = []
                for i in range(math.ceil(points.shape[0] / self.points_batch_size)):
                    start = i * self.points_batch_size
                    end = min((i + 1) * self.points_batch_size, points.shape[0])
                    pts = points[start:end, :]
                    pixel_pts, label_pts = self.get_2d_feature(pts, keyframe_dict, device, color, encoder=encoder, decoders=decoders)
                    values, labels = self.eval_points(pts, pixel_pts=pixel_pts, gt_label_pts=label_pts, 
                                                    decoders=decoders, fine_decoders=fine_decoders,
                                                    stage='fine', device=device)
                    z.append(torch.cat((values, labels.unsqueeze(-1)), -1).cpu().numpy())
                z = np.concatenate(z, axis=0)
                z = z.astype(np.float32)

                vertex_colors = z[:, :3]
                vertex_colors = np.clip(vertex_colors, 0, 1) * 255
                vertex_colors = vertex_colors.astype(np.uint8)

                # cyan color for forecast region
                if show_forecast:
                    seen_mask, forecast_mask, unseen_mask = self.point_masks(
                        vertices, keyframe_dict, estimate_c2w_list, idx, device, decoders,
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    vertex_colors[forecast_mask, 0] = 0
                    vertex_colors[forecast_mask, 1] = 255
                    vertex_colors[forecast_mask, 2] = 255

            else:
                vertex_colors = None

            vertex_labels = z[:, -1]
            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh.fill_holes()
            mesh.export(f'{mesh_out_file}/mesh_{idx}.ply')
            if self.verbose:
                print('Saved mesh at', f'{mesh_out_file}/mesh_{idx}.ply')

            if label and self.v_map_function is not None:
                vertex_labels_ = self.v_map_function(vertex_labels)
                vertex_label_vis = np.zeros([vertex_labels.shape[0], 3])
                for i in range(3):
                    vertex_label_vis[..., i] = vertex_labels_[i]

                mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_label_vis)
                mesh.export(f'{mesh_out_file}/mesh_{idx}_semantic.ply')
                if self.verbose:
                    print('Saved label mesh at', f'{mesh_out_file}/mesh_{idx}_semantic.ply')

            if element:
                element_id = np.unique(vertex_labels)
                for e_id in element_id:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices,
                                            faces=faces,
                                            process=False)
                    label_mask = vertex_labels == e_id
                    unseen_mask = ~label_mask
                    face_mask = unseen_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)

                    vert = mesh.vertices
                    fa = mesh.faces

                    if color:
                        points = torch.from_numpy(vert).to(self.device).float()
                        z = []
                        for i in range(math.ceil(points.shape[0] / self.points_batch_size)):
                            start = i * self.points_batch_size
                            end = min((i + 1) * self.points_batch_size, points.shape[0])
                            pts = points[start:end, :]
                            pixel_pts, label_pts = self.get_2d_feature(pts, keyframe_dict, device, color, encoder=encoder, decoders=decoders)
                            values, labels = self.eval_points(pts, pixel_pts=pixel_pts, gt_label_pts=label_pts, 
                                                            decoders=decoders, fine_decoders=fine_decoders,
                                                            stage='fine', device=device)
                            z.append(torch.cat((values, labels.unsqueeze(-1)), -1).cpu().numpy())
                        z = np.concatenate(z, axis=0)
                        z = z.astype(np.float32)

                        vertex_colors = z[:, :3]
                        vertex_colors = np.clip(vertex_colors, 0, 1) * 255
                        vertex_colors = vertex_colors.astype(np.uint8)

                    else:
                        vertex_colors = None

                    mesh = trimesh.Trimesh(vert, fa, vertex_colors=vertex_colors)
                    mesh.export(f'{mesh_out_file}/mesh_{idx}_part_{int(e_id)}.ply')
                    if self.verbose:
                        print('Saved mesh at', f'{mesh_out_file}/mesh_{idx}_part_{int(e_id)}.ply')

