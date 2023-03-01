import os
import copy
from matplotlib.pyplot import contour
from tqdm import tqdm
import shutil
import pickle
import torch
import numpy as np
import open3d as o3d

from utils import config, misc, geometry

def check_condition(pose, last_kf_pose, min_angle, min_distance):
    angle = np.arccos(
            ((np.linalg.inv(pose[:3, :3]) @ last_kf_pose[:3, :3] @ 
              np.array([0, 0, 1]).T) * np.array([0, 0, 1])).sum())
    dis = np.linalg.norm(pose[:3, 3] - last_kf_pose[:3, 3])
    
    if angle > (min_angle / 180) * np.pi or dis > min_distance:
        return True
    else:
        return False

def main():
    args = config.load_parser()
    
    gt_poses = misc.load_gt_pose(args)
    
    if not os.path.exists(args.scene_path):
        os.makedirs(args.scene_path)

    key_frames = []
    keyframes_gt = []
    last_kf_pose = np.eye(4)
    
    if args.pseudo_lidar:
        # load depth file
        depth_path = os.path.join(args.data_path, "depth")
        save_depth_path = os.path.join(args.scene_path, "depth")
        if not os.path.exists(save_depth_path):
            os.makedirs(save_depth_path)
        
        for frame_num, depth_file in enumerate(tqdm(sorted(os.listdir(depth_path), key=lambda s: int(s.split('.')[0])), 
                                                    desc = "raw frames")):
            depth_file = os.path.join(depth_path, depth_file)
            gt_pose = gt_poses[frame_num]
            
            if np.any(np.isnan(gt_pose)) or np.any(np.isinf(gt_pose)):
                continue
            
            if len(key_frames) == 0 or check_condition(gt_pose, last_kf_pose, args.min_ang, args.min_trans):
                if args.use_disturb:
                    init_pose = misc.disturb_pose(copy.deepcopy(gt_pose), args.disturb)
                else:
                    init_pose = gt_pose
                
                key_frame_info = (frame_num, torch.from_numpy(init_pose).to(args.device))
                key_frames.append(key_frame_info)
                keyframes_gt.append(gt_pose)
                
                shutil.copy(depth_file, save_depth_path)
                
                last_kf_pose = gt_pose
                
    else:
        # load pointcloud file
        pcd_path = os.path.join(args.data_path, "pointcloud")
        save_pcd_path = os.path.join(args.scene_path, "pointcloud")
        if not os.path.exists(save_pcd_path):
            os.makedirs(save_pcd_path)
        
        Tc2l = np.array([[0.0, 0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype = np.float32)
        
        for frame_num, pcd_file in enumerate(tqdm(sorted(os.listdir(pcd_path), key=lambda s: int(s.split('.')[0])), 
                                                  desc = "raw frames")):
            save_pcd_file = os.path.join(save_pcd_path, pcd_file)
            pcd_file = os.path.join(pcd_path, pcd_file)
            gt_pose = gt_poses[frame_num]
            gt_pose = gt_pose @ Tc2l
            
            if len(key_frames) == 0 or check_condition(gt_pose, last_kf_pose, args.min_ang, args.min_trans):
                if args.use_disturb:
                    init_pose = misc.disturb_pose(copy.deepcopy(gt_pose), args.disturb)
                else:
                    init_pose = gt_pose
                    
                key_frame_info = (frame_num, torch.from_numpy(init_pose).to(args.device))
                key_frames.append(key_frame_info)
                keyframes_gt.append(gt_pose)
                              
                pcd = o3d.io.read_point_cloud(pcd_file)
                pcd = geometry.pcd_preprocess(pcd, args.max_dist)
                o3d.io.write_point_cloud(save_pcd_file, pcd)
                
                last_kf_pose = gt_pose
    
    save_pickle = os.path.join(args.scene_path, "key_frames.pkl")
    with open(save_pickle, "wb") as pickle_file:
        pickle.dump(key_frames, pickle_file)
    
    save_pickle = os.path.join(args.scene_path, "keyframes_gt.pkl")
    with open(save_pickle, "wb") as pickle_file:
        pickle.dump(keyframes_gt, pickle_file)

if __name__ == "__main__":
    main()
    