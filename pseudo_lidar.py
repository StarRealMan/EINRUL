import os
from tqdm import tqdm
import cv2
import open3d as o3d
import torch
from torchvision.transforms.functional import to_tensor

from utils import geometry, config, misc

def main():
    args = config.load_parser()
    
    if args.use_debug_mode:
        torch.autograd.set_detect_anomaly(True)
    
    depth_path = os.path.join(args.scene_path, "depth")
    pcd_path = os.path.join(args.scene_path, "pointcloud")
    
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)
    
    for depth_file in tqdm(sorted(os.listdir(depth_path), key=lambda s: int(s.split('.')[0])), 
                           desc = "depth2pcd"):
        file_name = depth_file.split('.')[0]
        depth_file = os.path.join(depth_path, depth_file)
        
        img_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        
        # change depth into meter metric
        img_depth = misc.scale_depth(img_depth, args.data_type)
        img_depth = to_tensor(img_depth).to(args.device)
        
        pcd = geometry.depth2pcd(img_depth, args.pseudo_lidar_point, args.intrinsic)
        pcd = geometry.pcd_preprocess(pcd, args.max_dist)
        
        pcd_file = os.path.join(pcd_path, file_name + ".pcd")
        o3d.io.write_point_cloud(pcd_file, pcd)

if __name__ == "__main__":
    main()
    