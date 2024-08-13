import torch
import random
import numpy as np
import argparse

from utils.common import load_config
from slams.dns_slam import DNS_SLAM

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

if __name__ == '__main__':
    setup_seed(10)
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = load_config(args.config, 'configs/slam.yaml')
    if args.input is not None:
        cfg['dataset_dir'] = args.input
    if args.output is not None:
        cfg['out_dir'] = args.output
    print(cfg)
    
    device = [torch.device('cuda:0'), torch.device('cuda:0')]
    
    slam = DNS_SLAM(cfg, device)
    print('Model created.')

    # SLAM
    slam.run()

