import os
import torch
import numpy as np
import cv2

import sys
sys.path.append("..")

from utils import misc, render

def secant(model, f_low, f_high, d_low, d_high, n_secant_steps,
                        ray0_masked, ray_direction_masked, tau, it=0):
    ''' Runs the secant method for interval [d_low, d_high].

    Args:
        d_low (tensor): start values for the interval
        d_high (tensor): end values for the interval
        n_secant_steps (int): number of steps
        ray0_masked (tensor): masked ray start points
        ray_direction_masked (tensor): masked ray direction vectors
        model (nn.Module): model model to evaluate point occupancies
        c (tensor): latent conditioned code c
        tau (float): threshold value in logits
    '''
    d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    for i in range(n_secant_steps):
        p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
        with torch.no_grad():
            f_mid = model(p_mid)[...,0] - tau
        ind_low = f_mid < 0
        ind_low = ind_low
        if ind_low.sum() > 0:
            d_low[ind_low] = d_pred[ind_low]
            f_low[ind_low] = f_mid[ind_low]
        if (ind_low == 0).sum() > 0:
            d_high[ind_low == 0] = d_pred[ind_low == 0]
            f_high[ind_low == 0] = f_mid[ind_low == 0]

        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    return d_pred
    
def ray_marching(ray0, ray_direction, model, c=None,
                            tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                            depth_range=[0., 2.4], max_points=3500000):
    ''' Performs ray marching to detect surface points.

    The function returns the surface points as well as d_i of the formula
        ray(d_i) = ray0 + d_i * ray_direction
    which hit the surface points. In addition, masks are returned for
    illegal values.

    Args:
        ray0 (tensor): ray start points of dimension B x N x 3
        ray_direction (tensor):ray direction vectors of dim B x N x 3
        model (nn.Module): model model to evaluate point occupancies
        c (tensor): latent conditioned code
        tay (float): threshold value
        n_steps (tuple): interval from which the number of evaluation
            steps if sampled
        n_secant_steps (int): number of secant refinement steps
        depth_range (tuple): range of possible depth values (not relevant when
            using cube intersection)
        method (string): refinement method (default: secant)
        check_cube_intersection (bool): whether to intersect rays with
            unit cube for evaluation
        max_points (int): max number of points loaded to GPU memory
    '''
    # Shotscuts
    batch_size, n_pts, D = ray0.shape
    device = ray0.device
    tau = 0.5
    n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()
    
    d_proposal = torch.linspace(
        0, 1, steps=n_steps).view(
            1, 1, n_steps, 1).to(device)
    d_proposal = depth_range[0] * (1. - d_proposal) + depth_range[1] * d_proposal
    d_proposal = d_proposal.expand(batch_size, n_pts, -1, -1)
    
    p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
        ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal
    
    # Evaluate all proposal points in parallel
    with torch.no_grad():
        val = []
        for p_split in torch.split(p_proposal.reshape(batch_size, -1, 3), int(max_points / batch_size), dim=1):
            p_split = p_split.contiguous().view(-1, 3)
            res = model(p_split) - tau
            res = res.view(batch_size, -1)
            val.append(res)
        val = torch.cat(val, dim=1).view(batch_size, -1, n_steps)

    # Create mask for valid points where the first point is not occupied
    mask_0_not_occupied = val[:, :, 0] < 0

    # Calculate if sign change occurred and concat 1 (no sign change) in
    # last dimension
    sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                torch.ones(batch_size, n_pts, 1).to(device)],
                            dim=-1)
    cost_matrix = sign_matrix * torch.arange(
        n_steps, 0, -1).float().to(device)

    # Get first sign change and mask for values where a.) a sign changed
    # occurred and b.) no a neg to pos sign change occurred (meaning from
    # inside surface to outside)
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                            torch.arange(n_pts).unsqueeze(-0), indices] < 0

    # Define mask where a valid depth value is found
    mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied 

    # Get depth values and function values for the interval
    # to which we want to apply the Secant method
    n = batch_size * n_pts
    d_low = d_proposal.view(
        n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
    f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
        batch_size, n_pts)[mask]
    indices = torch.clamp(indices + 1, max=n_steps-1)
    d_high = d_proposal.view(
        n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
    f_high = val.view(
        n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]

    ray0_masked = ray0[mask]
    ray_direction_masked = ray_direction[mask]

    # write c in pointwise format
    if c is not None and c.shape[-1] != 0:
        c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]
    
    # Apply surface depth refinement step (e.g. Secant method)
    d_pred = secant(model, 
        f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
        ray_direction_masked, tau)

    # for sanity
    d_pred_out = torch.ones(batch_size, n_pts).to(device)
    d_pred_out[mask] = d_pred
    d_pred_out[mask == 0] = np.inf
    d_pred_out[mask_0_not_occupied == 0] = 0
    return d_pred_out

def cal_dists(pose, model, args):
    rays = render.image2ray(args.image_height, args.image_width, args.intrinsic)
    rays = render.transform_ray(rays, pose)
    rays_o = rays[None, :, :3]
    rays_d = rays[None, :, 3:6]
    rays_o = rays_o
    rays_d = rays_d/rays_d.norm(2,2).unsqueeze(-1)
    
    device = rays.device
    
    # run ray tracer / depth function --> 3D point on surface (differentiable)
    d_i_out = []
    for chunk_num in range(0, rays_o.shape[1], args.dist_chunk):
        rays_o_chunk = rays_o[:, chunk_num:chunk_num+args.dist_chunk]
        rays_d_chunk = rays_d[:, chunk_num:chunk_num+args.dist_chunk]
        
        with torch.no_grad():
            d_i_chunk = ray_marching(rays_o_chunk, rays_d_chunk, model, n_secant_steps=8,
                            depth_range = [args.render_near, args.render_far], n_steps=[int(512),int(512)+1])

        d_i_out.append(d_i_chunk)
        
    d_i = torch.cat(d_i_out, 1)
    
    # Get mask for where first evaluation point is occupied
    d_i = d_i.detach()

    mask_zero_occupied = d_i == 0
    mask_pred = misc.get_mask(d_i).detach()

    # For sanity for the gradients
    with torch.no_grad():
        dists =  torch.zeros_like(d_i).to(device)
        dists[mask_pred] = d_i[mask_pred].detach()
        dists[mask_zero_occupied] = 0.
        network_object_mask = mask_pred & ~mask_zero_occupied
        network_object_mask = network_object_mask[0]
        dists = dists.squeeze(0)
    
    return rays_o, rays_d, dists, network_object_mask

def phong_renderer(rays_o, rays_d, model, dists, mask):
    batch_size, n_points, _ = rays_o.shape
    device = rays_o.device

    light_source = rays_o[0,0]
    light = (light_source / light_source.norm(2)).unsqueeze(1).cuda()

    diffuse_per = torch.Tensor([0.7,0.7,0.7]).float()
    ambiant = torch.Tensor([0.3,0.3,0.3]).float()

    with torch.no_grad():
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        points = rays_o + rays_d * dists.unsqueeze(-1)
        points = points.view(-1,3)
        rgb_values = torch.ones_like(points).float().cuda()

        surface_points = points[mask]

        # Derive Normals
        grad = []
        for pnts in torch.split(surface_points, 1000000, dim=0):
            grad.append(model.gradient(pnts)[:,0,:].detach())
            torch.cuda.empty_cache()
        grad = torch.cat(grad,0)
        surface_normals = grad / grad.norm(2,1,keepdim=True)

    diffuse = torch.mm(surface_normals, light).clamp_min(0).repeat(1, 3) * diffuse_per.unsqueeze(0).cuda()
    rgb_values[mask] = (ambiant.unsqueeze(0).cuda() + diffuse).clamp_max(1.0)

    with torch.no_grad():
        rgb_val = torch.zeros(batch_size * n_points, 3, device=device)
        rgb_val[mask] = model(surface_points)

    out_dict = {
        'rgb': rgb_values.reshape(batch_size, -1, 3),
        'rgb_surf': rgb_val.reshape(batch_size, -1, 3),
    }

    return out_dict

def depth_renderer(rays_o, rays_d, model, dists, args):
    batch_size, n_points, _ = rays_o.shape
    device = rays_o.device

    with torch.no_grad():
        depth_surf = dists.unsqueeze(-1)
        rays_o = rays_o.squeeze(0)
        rays_d = rays_d.squeeze(0)
        
        near = torch.clamp_min(depth_surf - 3 * args.variance, 0.0)
        far = depth_surf + 3 * args.variance
        dist = torch.randn((batch_size * n_points, args.ray_points), device = device)
        
        if int(args.ray_points * args.stratified_portion) != 0:
            # uniform random sampling
            t_vals = torch.linspace(0., 1., steps = int(args.ray_points * args.stratified_portion)).cuda()
            dist_strat = -3 * args.variance * (1.-t_vals) + 3 * args.variance * (t_vals)
            dist_strat = dist_strat.expand([n_points, -1])
            mids = .5 * (dist_strat[...,1:] + dist_strat[...,:-1])
            upper = torch.cat([mids, dist_strat[...,-1:]], -1)
            lower = torch.cat([dist_strat[...,:1], mids], -1)
            t_rand = torch.rand(dist_strat.shape).cuda()
            dist_strat = lower + (upper - lower) * t_rand        
            dist = torch.cat([dist, dist_strat], -1)
        
        dist = torch.sort(dist, -1).values
        
        z_vals = dist * args.variance + depth_surf
        z_vals = torch.clamp(z_vals, min = near, max = far)
        
        depth_surf = depth_surf.squeeze(-1)
        rays_o = rays_o.view(-1, 1, 3)
        rays_d = rays_d.view(-1, 1, 3)
         
        xyz = rays_o + rays_d * z_vals.unsqueeze(-1)
        xyz = xyz.view(-1, 3)
        
        model_out = []
        for chunk_num in range(0, xyz.shape[0], args.depth_chunk):
            xyz_chunk = xyz[chunk_num:chunk_num+args.depth_chunk]
            model_out_chunk = model(xyz_chunk)
            model_out.append(model_out_chunk)
            
        model_out = torch.cat(model_out, 0)
        
        # model out represent occupancy
        alphas = model_out.view(-1, dist.shape[-1])
        
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1) # [1, 1-a1, 1-a2, ...]
        
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)

        depth = torch.sum(weights * z_vals, 1)
    
    out_dict = {
        'depth': depth.reshape(batch_size, -1, 1),
        'depth_surf': depth_surf.reshape(batch_size, -1, 1),
    }
    return out_dict

def save_render_result(image, id, type, args):
    if type == "phong" or type == "surf":
        image = image * 255
        image = image.byte().cpu().numpy()
        file_name = os.path.join(args.scene_path, "phong_render", str(id) + "_" + type + ".jpg")
        cv2.imwrite(file_name, image)
        
    elif type == "depth" or type == "depth_surf":
        image = image.cpu().numpy()
        image = image.astype(np.float32)
        
        pseudo = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=50), cv2.COLORMAP_JET)
        
        image = misc.back_scale_depth(image, args.data_type)
        image = image.astype(np.uint16)

        file_name = os.path.join(args.scene_path, "depth_render", str(id) + "_" + type + ".png")
        pseudo_file_name = os.path.join(args.scene_path, "depth_render", str(id) + "_" + type + "_pseudo" + ".png")
        cv2.imwrite(file_name, image)
        cv2.imwrite(pseudo_file_name, pseudo)

