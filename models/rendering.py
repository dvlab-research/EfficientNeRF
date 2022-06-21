import torch
# from torchsearchsorted import searchsorted

__all__ = ['render_rays']


def render_rays(models,
                embeddings,
                rays,
                sigma_voxels,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def sigma2weights(z_vals, sigmas, dirs):
            # Convert these values using volume rendering (Section 4)
            deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

            # Multiply each distance by the norm of its corresponding direction ray
            # to convert to real world distance (accounts for non-unit directions).
            deltas = deltas * torch.norm(dirs.unsqueeze(1), dim=-1)

            noise = torch.randn(sigmas.shape, device=sigmas.device) * 0.0

            # compute alpha by the formula (3)
            alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas+noise)) # (N_rays, N_samples_)
            alphas_shifted = \
                torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
            weights = \
                alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
            return weights
    

    def inference(model, embedding_xyz, xyz_, dirs, dir_embedded, z_vals, idx_render, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dirs: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_[idx_render[:, 0], idx_render[:, 1]].view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = dir_embedded.unsqueeze(1).expand(-1, N_samples_, -1)
            dir_embedded = dir_embedded[idx_render[:, 0], idx_render[:, 1]]
            view_dir = dirs.unsqueeze(1).expand(-1, N_samples_, -1)
            view_dir = view_dir[idx_render[:, 0], idx_render[:, 1]]
        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, view_dir[i:i+chunk], sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)
        if weights_only:
            out_sigma = torch.full((N_rays, N_samples_, 1), -20.0, device=rays.device)
            out_sigma[idx_render[:, 0], idx_render[:, 1]] = out
            out = out_sigma
            sigmas = out.view(N_rays, N_samples_)
        else:
            out_rgb = torch.full((N_rays, N_samples_, 3), 1.0, device=rays.device)
            out_sigma = torch.full((N_rays, N_samples_, 1), -20.0, device=rays.device)
            out_defaults = torch.cat([out_rgb, out_sigma], dim=2)
            out_defaults[idx_render[:, 0], idx_render[:, 1]] = out
            out = out_defaults

            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        # deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        # delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        # deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # # Multiply each distance by the norm of its corresponding direction ray
        # # to convert to real world distance (accounts for non-unit directions).
        # deltas = deltas * torch.norm(dirs.unsqueeze(1), dim=-1)

        # noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # # compute alpha by the formula (3)
        # alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas+noise)) # (N_rays, N_samples_)
        # alphas_shifted = \
        #     torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        # weights = \
        #     alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)

        weights = sigma2weights(z_vals, sigmas, dirs)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights, rgbs, sigmas


    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    is_training = model_coarse.training

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if is_training:
        z_vals = z_vals + torch.empty_like(z_vals).normal_(0.0, 0.002) * (far - near)

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    # range of voxel
    scope = 1.1 * (far[0, 0] - near[0, 0])  
    xyz_coarse_sampled_ = xyz_coarse_sampled.reshape(-1, 3)

    VOXEL_SIZE = sigma_voxels.shape[0]
    idx_voxels = ((xyz_coarse_sampled_ / scope / 2 + 0.5) * VOXEL_SIZE).round()
    idx_voxels = torch.clamp(idx_voxels, 0, VOXEL_SIZE-1).long()
    N_rays = rays.shape[0]
    sigmas = sigma_voxels[idx_voxels[:, 0], idx_voxels[:, 1], idx_voxels[:, 2]].reshape(N_rays, N_samples)
    weights = sigma2weights(z_vals, sigmas, rays_d)

    if is_training:
        idx_render_coarse = torch.nonzero(sigmas >= -20)

        rgb_coarse, depth_coarse, weights_coarse, colors_coarse, sigmas_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, idx_render_coarse, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1),
                #   'z_val_coarse': z_vals,
                #   'sigma_coarse': sigmas_coarse,
                #   'weight_coarse': weights_coarse
                 }
        sigma_voxels[idx_voxels[:, 0], idx_voxels[:, 1], idx_voxels[:, 2]] = \
                                    0.9 * sigma_voxels[idx_voxels[:, 0], idx_voxels[:, 1], idx_voxels[:, 2]]  + \
                                    0.1 * sigmas_coarse.reshape(-1).detach()
    else:
        weights_coarse = weights
        result = {
            # 'weight_coarse': weights_coarse
        }        

    if N_importance > 0: # sample points for fine model
        idx_render = torch.nonzero(weights_coarse >= min(1e-3, weights_coarse.max().item())).long()  # (M, 2)

        scale = N_importance
        z_0 = torch.cat([z_vals, z_vals[:, -1:]], dim=-1)
        for i in range(1, scale):
            z_vals_mid = i / scale * z_0[:, 1:] + (1 - i / scale) * z_0[:, :-1]
            z_vals = torch.sort(torch.cat([z_vals, z_vals_mid], dim=-1), dim=-1)[0]

        idxs = [idx_render.clone() for _ in range(scale)]
        for i in range(scale):
            idxs[i][:, 1] = idxs[i][:, 1] * scale + i - scale // 2
        idx_render_fine = torch.cat(idxs, dim=0)
        idx_render_fine[:, 1] = torch.clamp(idx_render_fine[:, 1], 0, int(N_samples * scale))

        if idx_render_fine.shape[0] >= N_rays * 64:
            indices = torch.randperm(idx_render_fine.shape[0])[:N_rays * 64]
            idx_render_fine = idx_render_fine[indices]
        
        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine, colors_fine, sigmas_fine = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, idx_render_fine, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)
        if is_training:
            result['mean_samples_coarse'] = torch.FloatTensor([idx_render_coarse.shape[0] / N_rays])
        result['mean_samples_fine'] = torch.FloatTensor([idx_render_fine.shape[0] / N_rays])
        # result['z_val_fine'] = z_vals
        # result['sigma_fine'] = sigmas_fine
        # result['weight_fine'] = weights_fine

    return result