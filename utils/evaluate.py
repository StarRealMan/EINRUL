import os
from packaging import version
from tqdm import tqdm
import cv2
import torch
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("..")

from utils import misc, visualize
from thirdparty import align_trajectory

def gen_render_result(args, model, poses, key_frame_info):
    phong_path = os.path.join(args.scene_path, "phong_render")
    if not os.path.exists(phong_path):
        os.makedirs(phong_path)
    depth_path = os.path.join(args.scene_path, "depth_render")
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    
    for keyframe_num, pose in enumerate(tqdm(poses, desc = "key frames")):
        rays_o, rays_d, dists, mask = visualize.cal_dists(pose, model, args)
        keyframe_id = key_frame_info[keyframe_num][0]
        
        # visualize mesh using phong rendering
        output_dict = visualize.phong_renderer(rays_o, rays_d, model, dists, mask)
        rgb = output_dict["rgb"].view(args.image_height, args.image_width, 3)
        rgb_surf = output_dict["rgb_surf"].view(args.image_height, args.image_width, 3)
        # visualize depth map
        output_dict = visualize.depth_renderer(rays_o, rays_d, model, dists, args)
        depth = output_dict["depth"].view(args.image_height, args.image_width)
        depth_surf = output_dict["depth_surf"].view(args.image_height, args.image_width)
        
        visualize.save_render_result(rgb, keyframe_id, "phong", args)
        visualize.save_render_result(rgb_surf, keyframe_id, "surf", args)
        visualize.save_render_result(depth, keyframe_id, "depth", args)
        visualize.save_render_result(depth_surf, keyframe_id, "depth_surf", args)
        
        if args.empty_cache:
            torch.cuda.empty_cache()

def compute_depth_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    mask = (pred != 0.0) & (gt != 0.0) & (abs(gt) != np.inf) & (np.isnan(gt) == False)
    gt = gt[mask]
    pred = pred[mask]
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a102 = (thresh < 1.02).mean()
    a105 = (thresh < 1.05).mean()
    a110 = (thresh < 1.10).mean()
    a125 = (thresh < 1.25).mean()
    a1252 = (thresh < 1.25 ** 2).mean()
    a1253 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a102, a105, a110, a125, a1252, a1253

def eval_2D_depth_map(args):
    gt_depth_path = os.path.join(args.scene_path, "depth")
    rendered_depth_path = os.path.join(args.scene_path, "depth_render")
    
    error_depth_list = []
    error_depth_surf_list = []
    
    for depth_file in tqdm(sorted(os.listdir(gt_depth_path), key=lambda s: int(s.split('.')[0])), 
                           desc = "depth maps"):
        id = depth_file.split('.')[0]
        gt_depth_file = os.path.join(gt_depth_path, depth_file)
        rendered_depth_file = os.path.join(rendered_depth_path, id + "_depth.png")
        rendered_depth_surf_file = os.path.join(rendered_depth_path, id + "_depth_surf.png")
        
        gt_depth = cv2.imread(gt_depth_file, cv2.IMREAD_UNCHANGED)
        rendered_depth = cv2.imread(rendered_depth_file, cv2.IMREAD_UNCHANGED)
        rendered_depth_surf = cv2.imread(rendered_depth_surf_file, cv2.IMREAD_UNCHANGED)
        
        gt_depth = misc.scale_depth(gt_depth, args.data_type)
        rendered_depth = misc.scale_depth(rendered_depth, args.data_type)
        rendered_depth_surf = misc.scale_depth(rendered_depth_surf, args.data_type)
        
        error_depth = compute_depth_errors(gt_depth, rendered_depth)
        error_depth_surf = compute_depth_errors(gt_depth, rendered_depth_surf)
        
        error_depth_list.append(list(error_depth))
        error_depth_surf_list.append(list(error_depth_surf))
    
    error_depth_np = np.array(error_depth_list)
    error_depth_surf_np = np.array(error_depth_surf_list)
    
    eval_res_file = os.path.join(args.scene_path, "eval_res.txt")
    
    depth_metric = np.mean(error_depth_np, 0)
    depth_surf_metric = np.mean(error_depth_surf_np, 0)
    
    abs_rel = depth_metric[0]
    sq_rel = depth_metric[1]
    rmse = depth_metric[2]
    rmse_log = depth_metric[3]
    a102 = depth_metric[4]
    a105 = depth_metric[5]
    a110 = depth_metric[6]
    a125 = depth_metric[7]
    a125_2 = depth_metric[8]
    a125_3 = depth_metric[9]
    surf_abs_rel = depth_surf_metric[0]
    surf_sq_rel = depth_surf_metric[1]
    surf_rmse = depth_surf_metric[2]
    surf_rmse_log = depth_surf_metric[3]
    surf_a102 = depth_surf_metric[4]
    surf_a105 = depth_surf_metric[5]
    surf_a110 = depth_surf_metric[6]
    surf_a125 = depth_surf_metric[7]
    surf_a125_2 = depth_surf_metric[8]
    surf_a125_3 = depth_surf_metric[9]
    
    with open(eval_res_file, "w") as f:
        f.write("depth metrics\n")
        f.write('abs_rel : ' + str(abs_rel) + "\n")
        f.write('sq_rel : ' + str(sq_rel) + "\n")
        f.write('rmse : ' + str(rmse) + "\n")
        f.write('rmse_log : ' + str(rmse_log) + "\n")
        f.write("a102 : " + str(a102) + "\n")
        f.write('a105 : ' + str(a105) + "\n")
        f.write('a110 : ' + str(a110) + "\n")
        f.write('a125 : ' + str(a125) + "\n")
        f.write('a125_2 : ' + str(a125_2) + "\n")
        f.write('a125_3 : ' + str(a125_3) + "\n")
        f.write('surf_abs_rel : ' + str(surf_abs_rel) + "\n")
        f.write('surf_sq_rel : ' + str(surf_sq_rel) + "\n")
        f.write('surf_rmse : ' + str(surf_rmse) + "\n")
        f.write('surf_rmse_log : ' + str(surf_rmse_log) + "\n")
        f.write("surf_a102 : " + str(surf_a102) + "\n")
        f.write('surf_a105 : ' + str(surf_a105) + "\n")
        f.write('surf_a110 : ' + str(surf_a110) + "\n")
        f.write('surf_a125 : ' + str(surf_a125) + "\n")
        f.write('surf_a125_2 : ' + str(surf_a125_2) + "\n")
        f.write('surf_a125_3 : ' + str(surf_a125_3) + "\n")

        print("************* depth metrics *************")
        print('abs_rel : ', abs_rel)
        print('sq_rel : ', sq_rel)
        print('rmse : ', rmse)
        print('rmse_log : ', rmse_log)
        print("a102 : ", a102)
        print('a105 : ', a105)
        print('a110 : ', a110)
        print('a125 : ', a125)
        print('a125_2 : ', a125_2)
        print("a125_3 : ", a125_3)
        print('surf_abs_rel : ', surf_abs_rel)
        print('surf_sq_rel : ', surf_sq_rel)
        print('surf_rmse : ', surf_rmse)
        print('surf_rmse_log : ', surf_rmse_log)
        print('surf_a102 : ', surf_a102)
        print('surf_a105 : ', surf_a105)
        print('surf_a110 : ', surf_a110)
        print('surf_a125 : ', surf_a125)
        print('surf_a125_2 : ', surf_a125_2)
        print('surf_a125_3 : ', surf_a125_3)

def normalize(x):
    return x / np.linalg.norm(x)

def chamfer_dist(tgt_points, ref_points, dist_th = 0.05, tgt_norlmal = None, ref_normal = None):
    ref_points_kd_tree = KDTree(ref_points)
    distances, idx = ref_points_kd_tree.query(tgt_points)
    dist = np.mean(distances)
    dist_ratio = np.mean((distances < dist_th).astype(np.float32))
    
    if tgt_norlmal is not None and ref_normal is not None:
        gt_norlmal = \
            gt_norlmal / np.linalg.norm(gt_norlmal, axis=-1, keepdims=True)
        rec_normal = \
            rec_normal / np.linalg.norm(rec_normal, axis=-1, keepdims=True)

        normals_dot_product = (rec_normal[idx] * gt_norlmal).sum(axis=-1)
        normals_dot_product = np.abs(normals_dot_product)
        
        return dist, dist_ratio, normals_dot_product

    else:
        return dist, dist_ratio

def intersection_over_union(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    
    if version.parse(o3d.__version__) >= version.parse('0.13.0'):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    else:
        reg_p2p = o3d.registration.registration_icp(
            o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPoint())
    
    transformation = reg_p2p.transformation
    return transformation

def eval_3D_mesh(args):
    """
    3D reconstruction metric.

    """
    if args.data_type == "scannet" or args.data_type == "9-synthetic-scenes" or args.data_type == "replica":
        gt_meshfile = os.path.join(args.data_path, args.exp_name + "_vh_clean.ply")
    else:
        gt_meshfile = os.path.join(args.data_path, args.exp_name + "_mesh.ply")
    mesh_gt = trimesh.load(gt_meshfile, process=False)
    
    rec_meshfile = os.path.join(args.scene_path, "final_mesh_clean.ply")
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    
    transformation = get_align_transformation(rec_meshfile, gt_meshfile)
    mesh_rec = mesh_rec.apply_transform(transformation)
    
    if isinstance(mesh_gt, trimesh.PointCloud):
        rec_pc_chioce = np.random.choice(len(mesh_rec.vertices), 200000)
        rec_pc_tri = trimesh.PointCloud(vertices=mesh_rec.vertices[rec_pc_chioce])
        
        gt_pc_chioce = np.random.choice(len(mesh_gt.vertices), 200000)
        gt_pc_tri = trimesh.PointCloud(vertices=mesh_gt.vertices[gt_pc_chioce])
    
    elif isinstance(mesh_gt, trimesh.Trimesh):
        # rec_pc, rec_idx = mesh_rec.sample(200000, return_index=True)
        # normals = mesh_rec.face_normals[rec_idx]
        rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
        rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

        gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
        gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    
    accuracy, precision = chamfer_dist(rec_pc_tri.vertices, gt_pc_tri.vertices)
    completion, recall = chamfer_dist(gt_pc_tri.vertices, rec_pc_tri.vertices)
    F_score = 2 * precision * recall / (precision + recall)
    
    eval_res_file = os.path.join(args.scene_path, "eval_res.txt")
    
    with open(eval_res_file, "w") as f:
        f.write("mesh metrics\n")
        f.write('accuracy : ' + str(accuracy) + "\n")
        f.write('completion : ' + str(completion) + "\n")
        f.write('precision : ' + str(precision) + "\n")
        f.write('recall : ' + str(recall) + "\n")
        f.write('F-score : ' + str(F_score) + "\n")

    print("************* mesh metrics *************")
    print('accuracy : ', accuracy)
    print('completion : ', completion)
    print('precision : ', precision)
    print('recall : ', recall)
    print('F-score : ', F_score)

def alignTrajectory(p_es, p_gt, q_es, q_gt):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    method can be: sim3, se3, posyaw, none;
    n_aligned: -1 means using all the frames
    '''
    assert p_es.shape[1] == 3
    assert p_gt.shape[1] == 3
    assert q_es.shape[1] == 4
    assert q_gt.shape[1] == 4
    
    est_pos = p_es[:, 0:3]
    gt_pos = p_gt[:, 0:3]
    
    _, R, t = align_trajectory.align_umeyama(gt_pos, est_pos, known_scale = True)  # note the order
    
    return R, t

def eval_pose_refinement(args, gt_poses, poses):
    gt_poses_np = np.stack(gt_poses, 0)
    poses_np = torch.stack(poses, 0).cpu().numpy()
    
    gt_R = gt_poses_np[:, :3, :3]  # (N0, 3, 3)
    gt_quat = R.from_matrix(gt_R).as_quat()  # (N0, 4)
    gt_t = gt_poses_np[:, :3, 3]  # (N0, 3)

    pred_R = poses_np[:, :3, :3]  # (N0, 3, 3)
    pred_quat = R.from_matrix(pred_R).as_quat()  # (N0, 4)
    pred_t = poses_np[:, :3, 3]  # (N0, 3)
    
    Rmat, trans = alignTrajectory(pred_t, gt_t, pred_quat, gt_quat)
    
    e_R_list = []
    e_t_list = []
    
    for keyframe_num in range(gt_poses_np.shape[0]):
        gt_R = gt_poses_np[keyframe_num, :3, :3]
        gt_t = gt_poses_np[keyframe_num, :3, 3]
        
        pred_R = poses_np[keyframe_num, :3, :3]
        pred_t = poses_np[keyframe_num, :3, 3]
        
        pred_R = Rmat @ pred_R
        pred_t = Rmat @ pred_t + trans
        
        e_R = np.arccos(
            ((np.linalg.inv(gt_R[:3, :3]) @ pred_R[:3, :3] @ 
            np.array([0, 0, 1]).T) * np.array([0, 0, 1])).sum())
        e_t = np.linalg.norm(gt_t - pred_t)
        
    e_R_list.append(e_R)
    e_t_list.append(e_t)
    
    e_R_np = np.array(e_R_list)
    e_t_np = np.array(e_t_list)
    
    rot_ate = np.mean(e_R_np, 0)
    trans_ate = np.mean(e_t_np, 0)
    
    eval_res_file = os.path.join(args.scene_path, "eval_res.txt")
    
    with open(eval_res_file, "w") as f:
        f.write("pose metrics\n")
        f.write('Rotation ATE : ' + str(rot_ate) + "\n")
        f.write('Translation ATE : ' + str(trans_ate) + "\n")
        
    print("************* pose metrics *************")
    print('Rotation ATE : ', rot_ate)
    print('Translation ATE : ', trans_ate)
    