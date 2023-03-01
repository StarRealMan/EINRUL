import os
import configargparse
import torch
import cv2

import sys
sys.path.append("..")

from utils import misc

def regular_homepath(path):
    if path[:2] == "~/":
        homepath = os.environ['HOME']
        path = os.path.join(homepath, path[2:])
    return path

def load_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    
    parser.add_argument("--min_ang", type=float, default=20.0, 
                        help='min angle selecting key frame')
    parser.add_argument("--min_trans", type=float, default=0.15, 
                        help='min translation selecting key frame')
    parser.add_argument("--pseudo_lidar_point", type=int, default=15000, 
                        help='pseudo lidar point num')
    
    parser.add_argument('--exp_name', type=str, default = "test_exp",
                        help='experiment name')
    parser.add_argument('--data_path', type=str, default = "./data",
                        help='data file path')
    parser.add_argument("--scene_path", type=str, default="./scene", 
                        help='scene path')
    parser.add_argument('--data_type', type=str, default = "scannet",
                        help='dataset type')
    parser.add_argument('--device', type=str, default = "cuda:0",
                        help='device used')
    
    parser.add_argument("--netdepth", type=int, default=2, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=64, 
                        help='channels per layer')
    parser.add_argument("--direct_loss_lambda", type=float, default=1.0, 
                        help='lambda used on direct loss supervision')
    parser.add_argument("--depth_loss_lambda", type=float, default=0.0, 
                        help='lambda used on depth map supervision')
    parser.add_argument("--normal_loss_lambda", type=float, default=0.4, 
                        help='lambda used on normal supervision')
    parser.add_argument("--bbox", type=float, default=15.0, 
                        help='bbox size')
    
    parser.add_argument("--map_lr", type=float, default=1e-3, 
                        help='map learning rate')
    parser.add_argument("--pose_lr", type=float, default=1e-3, 
                        help='pose learning rate')
    
    parser.add_argument("--sample_rays", type=int, default=100, 
                        help='sample rays per frame')
    parser.add_argument("--ray_points", type=int, default=32, 
                        help='sample points per ray')
    parser.add_argument("--variance", type=float, default=0.3, 
                        help='variance of the normal distribution used to sample z points')
    parser.add_argument("--normal_eps", type=float, default=0.005, 
                        help='neighbor distance when supervising normal loss')
    parser.add_argument("--stratified_portion", type=float, default=0.25, 
                        help='portion used to sample Gaussian and stratified')
    parser.add_argument("--weight_coe", type=float, default=2.83, 
                        help='supervision weight coe of Gaussian function')
    parser.add_argument("--max_dist", type=float, default=8.0, 
                        help='max distance of laser ray')
    parser.add_argument("--iteration", type=int, default=300, 
                        help='iteration num')
    parser.add_argument("--pose_milestone", type=int, default=150, 
                        help='pose scheduler milestone')
    parser.add_argument("--log_step", type=int, default=50, 
                        help='sample step in evaluation')

    parser.add_argument("--ang_disturb", type=float, default=0.05, 
                        help='angular disturbance on gt pose')
    parser.add_argument("--trans_disturb", type=float, default=0.1, 
                        help='translation disturbance on gt pose')
    
    parser.add_argument('--pseudo_lidar', default=False, action="store_true", 
                        help='True: use depth map dataset. False: use self-collected data')
    parser.add_argument("--use_disturb", default=False, action="store_true", 
                        help='whether use disturb on gt pose')
    parser.add_argument("--use_debug_mode", default=False, action="store_true", 
                        help='whether use debug mode')
    parser.add_argument("--use_visualization", default=False, action="store_true", 
                        help='whether use visualization')
    parser.add_argument('--empty_cache', default=False, action="store_true", 
                        help='use torch.cuda.empty_cache() using low cuda memory device')
    parser.add_argument('--finetune', default=False, action="store_true", 
                        help='use prior bbox to finetune network')
    parser.add_argument('--eval_mesh', default=False, action="store_true", 
                        help='whether evaluate mesh')
    parser.add_argument('--eval_depth', default=False, action="store_true", 
                        help='whether evaluate depth map')
    parser.add_argument('--eval_pose', default=False, action="store_true", 
                        help='whether evaluate pose')
    
    parser.add_argument("--render_near", type=float, default=0.1, 
                        help='pseudo focal length using LiDAR')
    parser.add_argument("--render_far", type=float, default=8.0, 
                        help='pseudo focal length using LiDAR')
    parser.add_argument("--mesh_reso", type=int, default=512, 
                        help='extracted mesh resolution')
    parser.add_argument("--eval_chunk", type=int, default=100000, 
                        help='chunk used in evaluation')
    parser.add_argument("--voxel_size", type=float, default=0.05, 
                        help='voxel size to downsample the pcd')
    
    parser.add_argument("--image_height", type=int, default=480, 
                        help='pseudo image height using LiDAR')
    parser.add_argument("--fov_x", type=float, default=70.4, 
                        help='lidar fov in x axis')
    parser.add_argument("--fov_y", type=float, default=77.2, 
                        help='lidar fov in y axis')
    
    args = parser.parse_args()
    
    if args.device == None:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    args.data_path = regular_homepath(args.data_path)
    args.scene_path = regular_homepath(args.scene_path)
    
    args.data_path = os.path.join(args.data_path, args.exp_name)
    args.scene_path = os.path.join(args.scene_path, args.exp_name)
    
    if args.finetune:
        args.pose_lr *= 0.2
        args.bbox = misc.load_bbox_prior(args)
        
        x_range = args.bbox[1][0] - args.bbox[0][0]
        y_range = args.bbox[1][1] - args.bbox[0][1]
        z_range = args.bbox[1][2] - args.bbox[0][2]
        args.bbox[0][0] -= x_range * 0.1
        args.bbox[1][0] += x_range * 0.1
        args.bbox[0][1] -= y_range * 0.1
        args.bbox[1][1] += y_range * 0.1
        args.bbox[0][2] -= z_range * 0.1
        args.bbox[1][2] += z_range * 0.1
        
    else:
        args.bbox = ([-args.bbox, -args.bbox, -args.bbox], [args.bbox, args.bbox, args.bbox])
    
    if args.use_disturb:
        args.disturb = (args.ang_disturb, args.trans_disturb)
    
    if args.pseudo_lidar:
        sample_depth_file = os.path.join(args.data_path, "depth", "0.png")
        sample_depth = cv2.imread(sample_depth_file, cv2.IMREAD_UNCHANGED)
        args.image_height = sample_depth.shape[0]
        args.image_width = sample_depth.shape[1]
        
        intrinsic_file = os.path.join(args.data_path, "intrinsic/intrinsic_depth.txt")
        intrinsic = misc.read_txt_mat(intrinsic_file)
        intrinsic = torch.from_numpy(intrinsic[:3, :3])
        args.intrinsic = intrinsic.float().to(args.device)
        
    else:
        fov_x = torch.tensor(args.fov_x / 180.0 * torch.pi)
        fov_y = torch.tensor(args.fov_y / 180.0 * torch.pi)
        
        cy = args.image_height / 2.0
        fy = cy / torch.tan(fov_y / 2)
        fx = fy
        cx = fx * torch.tan(fov_x / 2)
        
        args.intrinsic = torch.tensor([[fx, 0.0, cx],
                                       [0.0, fy, cy],
                                       [0.0, 0.0, 1.0]], dtype = float, device = args.device)
        
        args.image_width = int(cx * 2)
    
    args.mesh_chunk = args.eval_chunk
    args.dist_chunk = int(args.eval_chunk / 50)
    args.depth_chunk = int(args.eval_chunk * 3)
    
    return args
