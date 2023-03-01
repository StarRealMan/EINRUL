from utils import geometry, misc, config
import open3d as o3d
from tqdm import tqdm
import numpy as np
import os
import torch
import copy

args = config.load_parser()
pcd_path = os.path.join(args.scene_path, "pointcloud")
key_frame_info = misc.load_keyframe_info(args)
final_poses = misc.load_final_poses(args)

pcd_path = os.path.join(args.scene_path, "pointcloud")

full_pcd = o3d.geometry.PointCloud()
full_pcd_ours = o3d.geometry.PointCloud()

for num, keyframe in enumerate(tqdm(key_frame_info, desc = "register poses")):
    keyframe_id = keyframe[0]
    keyframe_pose = keyframe[1]
    final_pose = final_poses[num]
    
    Tl2c = torch.tensor([[0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]], device = keyframe_pose.device)
    
    keyframe_pose = keyframe_pose @ Tl2c
    final_pose = final_pose @ Tl2c
    
    pcd_file = os.path.join(pcd_path, str(keyframe_id) + ".pcd")
    
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd_ours = copy.deepcopy(pcd)
    pcd.transform(keyframe_pose.cpu().numpy())
    pcd_ours.transform(final_pose.cpu().numpy())
    # pcd = transform_pcd(pcd, keyframe_pose)
    
    full_pcd_ours += pcd_ours
    
    full_pcd += pcd

full_pcd = full_pcd.voxel_down_sample(voxel_size = 0.01)
pcd_file = os.path.join(args.scene_path, "final_pcd_gt.ply")
o3d.io.write_point_cloud(pcd_file, full_pcd)

# full_pcd_ours = full_pcd_ours.voxel_down_sample(voxel_size = 0.01)
# pcd_ours_file = os.path.join(args.scene_path, "final_pcd_ours.ply")
# o3d.io.write_point_cloud(pcd_ours_file, full_pcd_ours)
