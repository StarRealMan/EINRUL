import os
import numpy as np
import torch
import pickle
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from model import nerf

def read_txt_mat(filename):
    mat = []
    with open(filename, "r") as f:
        line = f.readline()
        while line:
            items = line.split(' ')
            mat_line = []
            for item in items:
                mat_line.append(float(item))
            mat.append(mat_line)
            line = f.readline()
    mat = np.array(mat, dtype=np.float32)
    
    return mat

def scale_depth(depth, type = "scannet"):
    if type == "scannet":
        scale_factor = 1000.0
    elif type == "realsense":
        scale_factor = 1000.0
    elif type == "replica":
        scale_factor = 1000.0
    elif type == "9-synthetic-scenes":
        scale_factor = 1000.0
    else:
        scale_factor = 1.0
    
    depth = depth.astype(np.float32)
    depth = depth / scale_factor
    
    return depth

def back_scale_depth(depth, type = "scannet"):
    if type == "scannet":
        scale_factor = 1000.0
    elif type == "realsense":
        scale_factor = 1000.0
    elif type == "replica":
        scale_factor = 1000.0
    elif type == "9-synthetic-scenes":
        scale_factor = 1000.0
    else:
        scale_factor = 1.0
    
    depth = depth.astype(np.float32)
    depth = depth * scale_factor
    
    return depth

def rainbow(portion, normalize = False):
    index = np.array(int(portion * 256)).astype(np.uint8).reshape(1, 1)
    color = cv2.applyColorMap(index, cv2.COLORMAP_RAINBOW)[0, 0]

    if normalize:
        color[0] = color[0] / 255.0
        color[1] = color[1] / 255.0
        color[2] = color[2] / 255.0
    
    return color

def quaternion2rotvec(quaternion):
    rotation = R.from_quat(quaternion)
    rotvec = rotation.as_rotvec()
    
    return rotvec

def rotvec2mat(rotvec):
    rotation = R.from_rotvec(rotvec)
    Rmat = rotation.as_matrix()
    
    return Rmat

def rotvecNtranslation2T(rotvec, trans):
    Rmat = rotvec2mat(rotvec)
    translation = np.array(trans, dtype = np.float32)   
    Tmat = np.eye(4, dtype = np.float32)
    Tmat[:3, :3] = Rmat
    Tmat[:3, 3] = translation
    
    return Tmat

def load_gt_pose(args):
    gt_poses = []
    
    if args.pseudo_lidar:
        gt_pose_path = os.path.join(args.data_path, "pose")
        for gt_pose_file in tqdm(sorted(os.listdir(gt_pose_path), key=lambda s: int(s.split('.')[0])), 
                                 desc = "gt poses"):
            gt_pose_file = os.path.join(gt_pose_path, gt_pose_file)
            gt_pose = read_txt_mat(gt_pose_file)

            gt_poses.append(gt_pose)
        
    else:
        gt_poses_path = os.path.join(args.data_path, "poses.pkl") 
        with open(gt_poses_path, "rb") as file:
            gt_poses_raw = pickle.load(file)
            for gt_pose_raw in gt_poses_raw:
                qauaternion = [gt_pose_raw["rotation"]["x"], gt_pose_raw["rotation"]["y"], 
                               gt_pose_raw["rotation"]["z"], gt_pose_raw["rotation"]["w"]]
                translation = [gt_pose_raw["position"]["x"], gt_pose_raw["position"]["y"], gt_pose_raw["position"]["z"]]
                rotvec = quaternion2rotvec(qauaternion)
                gt_pose = rotvecNtranslation2T(rotvec, translation)
                
                gt_poses.append(gt_pose)
        
    return gt_poses

def load_model(args, iter = 0):
    model = nerf.occNeRF(args.netdepth, args.netwidth, bbox = args.bbox).to(args.device)
    if iter == 0:
        ckpt_file = os.path.join(args.scene_path, "final_map.pt")
    else:
        ckpt_num = int(iter * args.log_step * 20  - 1)
        ckpt_file = os.path.join(args.scene_path, "checkpoints/" + str(ckpt_num) + "_map.pt")
  
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt)
    
    return model

def load_keyframe_gt_pose(args):
    keyframe_gt_path = os.path.join(args.scene_path, "keyframes_gt.pkl")
    with open(keyframe_gt_path, "rb") as pickle_file:
        keyframe_gt = pickle.load(pickle_file)

    return keyframe_gt

def load_keyframe_info(args):
    keyframe_info_path = os.path.join(args.scene_path, "key_frames.pkl")
    with open(keyframe_info_path, "rb") as pickle_file:
        key_frame_info = pickle.load(pickle_file)

    return key_frame_info

def load_final_poses(args):
    final_pose_file = os.path.join(args.scene_path, "final_pose.pkl")
    with open(final_pose_file, "rb") as file:
        key_frames = pickle.load(file)
        
    return key_frames

def load_bbox_prior(args):
    bbox_prior_file = os.path.join(args.scene_path, "bbox_prior.pkl")
    with open(bbox_prior_file, "rb") as file:
        bbox_prio = pickle.load(file)
    
    return bbox_prio

def disturb_pose(pose, disturb):
    rot = pose[:3, :3]
    rot_euler = R.from_matrix(rot).as_euler("zxy")
    trans = pose[:3, 3]
    
    rot_disturb = 2 * disturb[0] * (np.random.rand(3) - 0.5)
    trans_disturb = 2 * disturb[1] * (np.random.rand(3) - 0.5)
    
    rot_euler += rot_disturb
    rot = R.from_euler("zxy", rot_euler).as_matrix()
    trans += trans_disturb
    
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    
    return pose

def Gauss_func(x, miu = 0, sigma = 1.0):
    x = x - miu
    sigma2 = sigma * sigma
    coe = 1 / (2.506628275 * sigma)
    return coe * torch.exp(-torch.mul(x, x) / (2 * sigma2))

def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True

    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor

def get_mask(tensor):
    ''' Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    '''
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
    mask = mask.bool()
    if is_numpy:
        mask = mask.numpy()

    return mask

def cal_information(scan):
    full_size = scan.shape[0]
    information = torch.nonzero(scan).shape[0] / full_size
    
    return information
