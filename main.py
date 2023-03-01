import os
from tqdm import tqdm
import torch
import open3d as o3d
import numpy as np
import pickle

from model import nerf, camera
from tasks import lidar
from utils import config, misc

def main():
    args = config.load_parser()
    
    if args.use_debug_mode:
        torch.autograd.set_detect_anomaly(True)
    
    if args.finetune:
        coarse_poses = misc.load_final_poses(args)
    
    key_frame_info = misc.load_keyframe_info(args)
    
    key_frames = []
    init_pose = []
    
    pcd_path = os.path.join(args.scene_path, "pointcloud")
    for pcd_num, pcd_file in enumerate(tqdm(sorted(os.listdir(pcd_path), key=lambda s: int(s.split('.')[0])), 
                                            desc = "key frames")):
        pcd_file = os.path.join(pcd_path, pcd_file)
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd = np.asarray(pcd.points)
        pcd = torch.from_numpy(pcd).float().to(args.device)

        key_frames.append(pcd)
        
        if args.finetune:
            init_pose.append(coarse_poses[pcd_num])
        else:
            init_pose.append(key_frame_info[pcd_num][1])
            
    init_pose = torch.stack(init_pose, 0)
    
    nerf_model = nerf.occNeRF(args.netdepth, args.netwidth, bbox = args.bbox).to(args.device)
    pose_model = camera.LearnPose(len(key_frames), init_pose).to(args.device)
    
    ckpt_path = os.path.join(args.scene_path, "checkpoints")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    poses = lidar.global_op(key_frames, nerf_model, pose_model, args)
    
    save_pickle = os.path.join(args.scene_path, "final_pose.pkl")
    with open(save_pickle, "wb") as pickle_file:
        pickle.dump(poses, pickle_file)
    
    save_model = os.path.join(args.scene_path, "final_map.pt")
    torch.save(nerf_model.state_dict(), save_model)

if __name__ == "__main__":
    main()
    