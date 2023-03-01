import torch

def gen_weight_label(dist, weight_coe):
    sign = torch.sign(dist)
    label = (sign + 1.0) / 2.0
    inv_label = (-sign + 1.0) / 2.0
    weight = torch.exp(-weight_coe * torch.mul(dist, dist)) * label + torch.ones_like(dist) * inv_label
    
    # using soft label
    # label = torch.sigmoid(label_coe * dist)
    
    return weight, label

def image2ray(H, W, intrinsic):
    device = intrinsic.device
    uu = torch.linspace(0, W-1, W, device = device)
    vv = torch.linspace(0, H-1, H, device = device)
    point_u, point_v = torch.meshgrid(uu, vv, indexing = "xy")
    
    point_u = point_u.contiguous().view(-1)
    point_v = point_v.contiguous().view(-1)

    point_x = (point_u - intrinsic[0][2]) / intrinsic[0][0]
    point_y = (point_v - intrinsic[1][2]) / intrinsic[1][1]
    point_z = torch.ones_like(point_x)
    
    rays_o = torch.zeros((point_x.shape[0], 3), device = device)
    rays_d = torch.stack([point_x, point_y, point_z], -1)
    depth = point_z.unsqueeze(-1)
    
    rays = torch.cat([rays_o, rays_d, depth], -1)
    
    return rays
    
def random_sample_pcd(pcd, sample_rays):
    sampler = torch.randint(pcd.shape[0], (sample_rays,))
    pcd = pcd[sampler]
    
    return pcd

def pcd2ray(pcd):
    Tl2c = torch.tensor([[0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [1.0, 0.0, 0.0]], device = pcd.device)
    
    pcd = Tl2c @ pcd.permute(1, 0)
    pcd = pcd.permute(1, 0)
    
    depth = pcd[:, 2].unsqueeze(-1)
    
    rays_d = pcd / depth
    rays_o = torch.zeros_like(rays_d)
    rays = torch.cat([rays_o, rays_d], -1)
    
    return rays, depth

def transform_ray(rays, pose):
    # avoid inplace operation
    rays_o = rays[:, :3] + pose[:3, 3]
    rays_d = pose[:3, :3] @ rays[:, 3:6].transpose(1, 0)
    rays_d = rays_d.transpose(1, 0)
    
    rays = torch.cat([rays_o, rays_d], -1)
    
    return rays

def render_rays(rays, gt_depths, nerf_model, ray_points, variance, weight_coe, stratified = None):
    sample_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    device = rays.device
    
    # sample and get z_val using gt_depth
    near = torch.clamp_min(gt_depths - 3 * variance, 0.0)
    far = gt_depths + 3 * variance
    dist = torch.randn((sample_rays, ray_points), device = device)
    
    if stratified is not None and int(ray_points * stratified) != 0:
        # uniform random sampling
        t_vals = torch.linspace(0., 1., steps = int(ray_points * stratified)).cuda()
        dist_strat = -3 * variance * (1.-t_vals) + 3 * variance * (t_vals)
        dist_strat = dist_strat.expand([sample_rays, -1])
        mids = .5 * (dist_strat[...,1:] + dist_strat[...,:-1])
        upper = torch.cat([mids, dist_strat[...,-1:]], -1)
        lower = torch.cat([dist_strat[...,:1], mids], -1)
        t_rand = torch.rand(dist_strat.shape).cuda()
        dist_strat = lower + (upper - lower) * t_rand        
        dist = torch.cat([dist, dist_strat], -1)
    
    dist = torch.sort(dist, -1).values
    
    z_vals = dist * variance + gt_depths
    z_vals = torch.clamp(z_vals, min = near, max = far)
    
    rays_o = rays_o.view(-1, 1, 3)
    rays_d = rays_d.view(-1, 1, 3)
    
    occ_weight, label = gen_weight_label(dist, weight_coe)
    
    xyz = rays_o + rays_d * z_vals.unsqueeze(-1)
    xyz = xyz.view(-1, 3)
    
    occ_weight = occ_weight.view(-1)
    label = label.view(-1)
    
    model_out = nerf_model(xyz)
    occ = model_out.squeeze(-1)
    
    # model out represent occupancy
    alphas = model_out.view(-1, dist.shape[-1])
    
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1) # [1, 1-a1, 1-a2, ...]
    
    weights = \
        alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)

    depths = torch.sum(weights * z_vals, 1)
    
    return occ, label, occ_weight, depths

def render_normal(rays, gt_depths, nerf_model, epsilon):
    sample_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    device = rays.device
    
    xyz = rays_o + rays_d * gt_depths
    xyz = xyz.view(-1, 3)
    
    xyz_neighbor = xyz + torch.randn_like(xyz, device = device) * epsilon
    
    xyz_full = torch.concat([xyz, xyz_neighbor], 0)
    
    gradient_full = nerf_model.gradient(xyz_full)
    gradient_full = gradient_full.squeeze(1) / torch.norm(gradient_full, dim = -1)
    gradient = gradient_full[:sample_rays]
    gradient_neighbor = gradient_full[sample_rays:]
    
    return gradient, gradient_neighbor