import glob
import os
import csv
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_dataset(cfg, input_folder, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, input_folder, scale, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, input_folder, scale, device='cuda:0'
                 ):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.semantic = True

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        self.input_folder = input_folder

        self.crop_edge = cfg['cam']['crop_edge']

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        if self.name == 'replica':
            color_path = f'{self.input_folder}/rgb/rgb_{index}.png'
            depth_path = f'{self.input_folder}/depth/depth_{index}.png'
            label_path = f'{self.input_folder}/semantic_class/semantic_class_{index}.png'
        elif self.name == 'scannet':
            color_path = f'{self.input_folder}/color/{index}.jpg'
            depth_path = f'{self.input_folder}/depth/{index}.png'
            label_path = f'{self.input_folder}/label-filt/{index}.png'

        if self.semantic == True:
            label_data = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            #depth_data = cv2.imdecode(np.fromfile(depth_path, dtype=np.uint16), -1)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data.astype(np.float32) / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data)*self.scale

        if self.semantic == True:
            label_data = label_data.astype(np.float32)
            label_data = cv2.resize(label_data, (W, H), interpolation=cv2.INTER_NEAREST)
            label_data = self.v_map_function(label_data)
            label_data = torch.from_numpy(label_data)

        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()
            if self.semantic == True:
                label_data = F.interpolate(
                    label_data[None, None], self.crop_size, mode='nearest')[0, 0]

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            if self.semantic == True:
                label_data = label_data[edge:-edge, edge:-edge]
    
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        
        if self.semantic == True:
            return index, color_data.to(self.device), depth_data.to(self.device), label_data.to(self.device), pose.to(self.device)
        else:
            return index, color_data.to(self.device), depth_data.to(self.device), {}, pose.to(self.device)



class ScanNet(BaseDataset):
    def __init__(self, cfg, input_folder, scale, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, input_folder, scale, device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        
        self.input_folder = os.path.join(self.input_folder)#, 'frames')

        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))

        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        
        self.label_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'label-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.semantic = True

        tsv_file_path = self.input_folder + '/scannetv2-labels.combined.tsv'

        self.id_map = {}

        with open(tsv_file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            header = next(reader)
            for row in reader:
                self.id_map[int(row[0])] = int(row[4])

        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

        self.cal_class_n()

        self.v_map_function = np.vectorize(self.map_function)

    def map_function(self, value):
            value_nyu = self.id_map.get(value, 0)
            return self.label2class_dict.get(value_nyu, 0)  # .get(value, default)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    def cal_class_n(self):
        self.label2class_dict = {}
        self.class2label_dict = {}
        self.n_class = 0
        for i in range(0, self.n_img, 5):
            label_path = f'{self.input_folder}/label-filt/{i}.png'
            label_data = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            unique_labels = set(label_data.flatten())
            for label in unique_labels:
                label_nyu = self.id_map.get(label, 0)
                if label_nyu not in self.label2class_dict.keys():
                    self.label2class_dict.update({label_nyu: self.n_class})
                    self.class2label_dict.update({self.n_class: label_nyu})
                    self.n_class += 1
        
        print('NYU label to class dict:')
        print(self.label2class_dict)


class Replica(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, args, scale, device)

        self.H, self.W= cfg['cam']['H'], cfg['cam']['W']
        
        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0

        self.color_paths = sorted(glob.glob(f'{self.input_folder}/rgb/rgb_*.png'))
        self.depth_paths = sorted(glob.glob(f'{self.input_folder}/depth/depth_*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj_w_c.txt')
        self.semantic = True
        self.cal_class_n()

        self.v_map_function = np.vectorize(self.map_function)


    def map_function(self, value):
            return self.label2class_dict.get(value, value)  # .get(value, default)

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    def cal_class_n(self):
        self.label2class_dict = {}
        self.class2label_dict = {}
        self.n_class = 0
        for i in range(0, self.n_img, 5):
            label_path = f'{self.input_folder}/semantic_class/semantic_class_{i}.png'
            label_data = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            
            unique_labels = set(label_data.flatten())
            for label in unique_labels:
                if label not in self.label2class_dict.keys():
                    self.label2class_dict.update({label: self.n_class})
                    self.class2label_dict.update({self.n_class: label})
                    self.n_class += 1
            
        print('Replica label to class dict:')
        print(self.label2class_dict)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, scale, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)
        self.semantic = False

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet
}
