import os
import torch
import numpy as np
import mcubes
import trimesh
import open3d as o3d
from tqdm import tqdm
import pickle
from scipy.spatial import KDTree

def marching_cubes(model, reso, chunk, device, bbox):
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]
    
    scale = reso / (x_max - x_min)
    offset = np.array([x_min, y_min, z_min])
    
    step_x = reso
    step_y = round(scale * (y_max - y_min))
    step_z = round(scale * (z_max - z_min))
    
    y_max = y_min + step_y / scale
    z_max = z_min + step_z / scale
    
    x = torch.linspace(x_min, x_max, step_x, device = device)
    y = torch.linspace(y_min, y_max, step_y, device = device)
    z = torch.linspace(z_min, z_max, step_z, device = device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], -1)
    points = grid.view(-1, 3)
    
    model_out = []
    with torch.no_grad():
        for chunk_num in range(0, points.shape[0], chunk):
            points_chunk = points[chunk_num:chunk_num+chunk]
            model_out_chunk = model(points_chunk)
            model_out_chunk = model_out_chunk.detach().cpu()
            model_out.append(model_out_chunk)    
        
        model_out = torch.cat(model_out, 0)
        model_out = model_out.view(step_x, step_y, step_z)

    vertices, triangles = mcubes.marching_cubes(model_out.numpy(), 0.5)

    mesh = trimesh.Trimesh(vertices / scale + offset, triangles)
    
    return mesh

def delete_unseen_mesh(mesh, poses, H, W, intrinsic):
    n_imgs = len(poses)
    device = poses[0].device
    pcd = mesh.vertices

    # delete mesh vertices that are not inside any camera's viewing frustum
    whole_mask = np.zeros(pcd.shape[0]).astype(np.bool)
    for i in tqdm(range(0, n_imgs, 1), desc = "camera views"):
        c2w = poses[i]
        points = pcd.copy()
        points = torch.from_numpy(points).to(device)
        w2c = torch.linalg.inv(c2w)
        ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
        homo_points = torch.cat([points, ones], dim=1).reshape(-1, 4, 1).float().to(device)
        cam_cord_homo = w2c@homo_points
        cam_cord = cam_cord_homo[:, :3]

        cam_cord[:, 0] *= -1
        uv = intrinsic.float()@cam_cord.float()
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.float().squeeze(-1).cpu().numpy()
        edge = 0
        mask = (0 <= -z[:, 0, 0].cpu().numpy()) & \
               (uv[:, 0] < H - edge) & (uv[:, 0] > edge) & \
               (uv[:, 1] < W - edge) & (uv[:, 1] > edge)

        whole_mask |= mask
    
    face_mask = whole_mask[mesh.faces].all(axis=1)
    mesh.update_vertices(whole_mask)
    mesh.update_faces(face_mask)
    
    return mesh

def delete_backwall_mesh(mesh, pcd, ball_size):
    mesh_pcd = mesh.vertices
    pcd_np = np.asarray(pcd.points)
    
    mesh_points_kd_tree = KDTree(mesh_pcd)
    idx = mesh_points_kd_tree.query_ball_point(pcd_np, ball_size)
    idx = np.concatenate(idx).astype(np.int32)
    
    vertice_mask = np.zeros(mesh_pcd.shape[0]).astype(bool)
    vertice_mask[idx] = True
    face_mask = vertice_mask[mesh.faces].all(axis=1)
    
    mesh.update_vertices(vertice_mask)
    mesh.update_faces(face_mask)

    return mesh

def transform_pcd(pcd, pose):
    device = pose.device
    Tl2c = torch.tensor([[0.0, -1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], device = device)
    
    pcd_np = np.asarray(pcd.points).astype(np.float32)
    
    pcd_torch = torch.tensor(pcd_np, device = device)
    pcd_torch = torch.cat([pcd_torch, torch.ones_like(pcd_torch[:, :1], device = device)], -1)
    pcd_torch = Tl2c @ pcd_torch.transpose(1, 0)
    pcd_torch = (pose @ pcd_torch).transpose(1, 0)
    
    pcd_np = pcd_torch[:, :3].cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    
    return pcd

def register_pcd(poses, key_frame_info, pcd_path, voxel_size):
    full_pcd = o3d.geometry.PointCloud()
    for keyframe_num, keyframe_pose in enumerate(tqdm(poses, desc = "register poses")):
        keyframe_id = key_frame_info[keyframe_num][0]
        
        Tl2c = torch.tensor([[0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], device = keyframe_pose.device)
        
        keyframe_pose = keyframe_pose @ Tl2c
        
        pcd_file = os.path.join(pcd_path, str(keyframe_id) + ".pcd")
        
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd.transform(keyframe_pose.cpu().numpy())
        # pcd = transform_pcd(pcd, keyframe_pose)
        
        full_pcd += pcd
    
    full_pcd = full_pcd.voxel_down_sample(voxel_size = voxel_size)
    
    return full_pcd

def pcd2mesh(pcd):
    # estimate radius for rolling ball
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    # create the triangular mesh with the vertices and faces from open3d
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                           vertex_normals=np.asarray(mesh.vertex_normals))
    
    return mesh

def cal_bbox(pcd):
    pcd_np = np.asarray(pcd.points)
    
    xmin = np.min(pcd_np[:, 0])
    xmax = np.max(pcd_np[:, 0])
    ymin = np.min(pcd_np[:, 1])
    ymax = np.max(pcd_np[:, 1])
    zmin = np.min(pcd_np[:, 2])
    zmax = np.max(pcd_np[:, 2])
    
    return ([xmin, ymin, zmin], [xmax, ymax, zmax])

def extract_mesh(args, model):
    mesh = marching_cubes(model, args.mesh_reso, args.mesh_chunk, args.device, args.bbox)
    mesh_path = os.path.join(args.scene_path, "final_mesh.ply")
    mesh.export(mesh_path)
    
    pcd_file = os.path.join(args.scene_path, "final_pcd.pcd")
    pcd = o3d.io.read_point_cloud(pcd_file)
    mesh = delete_backwall_mesh(mesh, pcd, args.voxel_size)
    # mesh = delete_unseen_mesh(mesh, poses, args.image_height, args.image_width, args.intrinsic)
    
    mesh_path = os.path.join(args.scene_path, "final_mesh_clean.ply")
    mesh.export(mesh_path)

def save_pcd_bbox(args, poses, key_frame_info):
    pcd_path = os.path.join(args.scene_path, "pointcloud")
    full_pcd = register_pcd(poses, key_frame_info, pcd_path, args.voxel_size)
    
    pcd_file = os.path.join(args.scene_path, "final_pcd.pcd")
    o3d.io.write_point_cloud(pcd_file, full_pcd)
    bbox_prior = cal_bbox(full_pcd)
    
    save_pickle = os.path.join(args.scene_path, "bbox_prior.pkl")
    with open(save_pickle, "wb") as pickle_file:
        pickle.dump(bbox_prior, pickle_file)

def depth2pcd(depth, sample_points, intrinsic):
    nzero = torch.nonzero(depth.squeeze(0))
    sampler = (torch.rand(sample_points) * nzero.shape[0]).long()
    point_image = nzero[sampler, :]
    point_z = depth[:, point_image[:, 0], point_image[:, 1]].squeeze(0)
    
    point_u = point_image[:, 1]
    point_v = point_image[:, 0]
    
    point_x = (point_u - intrinsic[0][2]) / intrinsic[0][0] * point_z
    point_y = (point_v - intrinsic[1][2]) / intrinsic[1][1] * point_z
    
    Tc2l = torch.tensor([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]], device = depth.device)
    
    points = torch.stack([point_x, point_y, point_z], -1)
    points = Tc2l @ points.transpose(1, 0)
    points = points.transpose(1, 0)
    
    pcd_np = points.cpu().numpy()
    
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
    
    return pcd_o3d

def pcd_preprocess(pcd, max_dist):
    pcd_np = np.asarray(pcd.points)
    
    nzero = ~np.equal(pcd_np, 0.0)
    dist = np.linalg.norm(pcd_np, axis = -1)
    nge = np.less(dist, max_dist)
    mask = nzero[:, 0] & nzero[:, 1] & nzero[:, 2] & nge
    pcd_np = pcd_np[mask]
    
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
    
    return pcd_o3d