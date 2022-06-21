import os, sys

from torch._C import device, dtype
from opt import get_opts
import torch
import torch.nn.functional as F
from collections import defaultdict
from torchvision import transforms

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from spherical_harmonic import eval_sh_torch
from models.sh import eval_sh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import time
import imageio.v2 as imageio
import glob

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

class NerfTree_Pytorch(object):  # This is only based on Pytorch implementation
    def __init__(self, xyz_min, xyz_max, grid_coarse, grid_fine, deg, sigma_init, sigma_default, device):
        '''
        xyz_min: list (3,) or (1, 3)
        scope: float
        '''
        super().__init__()
        self.sigma_init = sigma_init
        self.sigma_default = sigma_default

        self.sigma_voxels_coarse = torch.full((grid_coarse,grid_coarse,grid_coarse), self.sigma_init, device=device)
        self.index_voxels_coarse = torch.full((grid_coarse,grid_coarse,grid_coarse), 0, dtype=torch.long, device=device)
        self.voxels_fine = None

        self.xyz_min = xyz_min[0]
        self.xyz_max = xyz_max[0]
        self.xyz_scope = self.xyz_max - self.xyz_min
        self.grid_coarse = grid_coarse
        self.grid_fine = grid_fine
        self.res_coarse = grid_coarse
        self.res_fine = grid_coarse * grid_fine        
        self.dim_sh = 3 * (deg + 1)**2
        self.device = device
    
    def calc_index_coarse(self, xyz):
        ijk_coarse = ((xyz - self.xyz_min) / self.xyz_scope * self.grid_coarse).long().clamp(min=0, max=self.grid_coarse-1)
        # return index_coarse[:, 0] * (self.grid_coarse**2) + index_coarse[:, 1] * self.grid_coarse + index_coarse[:, 2]
        return ijk_coarse

    def update_coarse(self, xyz, sigma, beta):
        '''
            xyz: (N, 3)
            sigma: (N,)
        '''
        ijk_coarse = self.calc_index_coarse(xyz)

        self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] \
                    = (1 - beta) * self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] + \
                        beta * sigma
    
    def create_voxels_fine(self):
        ijk_coarse = torch.logical_and(self.sigma_voxels_coarse > 0, self.sigma_voxels_coarse != self.sigma_init).nonzero().squeeze(1)  # (N, 3)
        num_valid = ijk_coarse.shape[0] + 1

        index = torch.arange(1, num_valid, dtype=torch.long, device=ijk_coarse.device)
        self.index_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] = index

        self.voxels_fine = torch.zeros(num_valid, self.grid_fine, self.grid_fine, self.grid_fine, self.dim_sh+1, device=self.device)
        self.voxels_fine[...,  0] = self.sigma_default
        self.voxels_fine[..., 1:] = 0.0

    def calc_index_fine(self, xyz):
        # xyz_norm = (xyz - self.xyz_min) / self.xyz_scope
        # xyz_coarse =  (xyz_norm * self.grid_coarse).long() * self.grid_fine
        # xyz_fine = (xyz_norm * self.res_fine).long()
        # index_fine = ((xyz_fine - xyz_coarse)).clamp(0, self.grid_fine-1)

        xyz_norm = (xyz - self.xyz_min) / self.xyz_scope
        xyz_fine = (xyz_norm * self.res_fine).long()
        index_fine = xyz_fine % self.grid_fine
        return index_fine
        
    def update_fine(self, xyz, sigma, sh):
        '''
            xyz: (N, 3)
            sigma: (N, 1)
            sh: (N, F)
        '''
        # calc ijk_coarse
        index_coarse = self.query_coarse(xyz, 'index')
        nonzero_index_coarse = torch.nonzero(index_coarse).squeeze(1)
        index_coarse = index_coarse[nonzero_index_coarse]

        # calc index_fine
        ijk_fine = self.calc_index_fine(xyz[nonzero_index_coarse])

        # feat
        feat = torch.cat([sigma, sh], dim=-1)

        self.voxels_fine[index_coarse, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]] = feat[nonzero_index_coarse]
    
    def query_coarse(self, xyz, type='sigma'):
        '''
            xyz: (N, 3)
        '''
        ijk_coarse = self.calc_index_coarse(xyz)

        if type == 'sigma':
            out = self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
        else:
            out = self.index_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
        return out

    def query_fine(self, xyz):
        '''
            x: (N, 3)
        '''
        # calc index_coarse
        index_coarse = self.query_coarse(xyz, 'index')

        # calc index_fine
        ijk_fine = self.calc_index_fine(xyz)

        return self.voxels_fine[index_coarse, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]]


class EfficientNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(EfficientNeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.deg = 2
        self.dim_sh = 3 * (self.deg + 1)**2

        self.nerf_coarse = NeRF(D=4, W=128,
                                in_channels_xyz=63, in_channels_dir=27, 
                                skips=[2], deg=self.deg)
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(D=8, W=256,
                                in_channels_xyz=63, in_channels_dir=27, 
                                skips=[4], deg=self.deg)
            self.models += [self.nerf_fine]
        self.sigma_init = hparams.sigma_init
        self.sigma_default = hparams.sigma_default

        # sparse voxels
        coord_scope = hparams.coord_scope
        self.nerf_tree = NerfTree_Pytorch(xyz_min=[-coord_scope, -coord_scope, -coord_scope], 
                                          xyz_max=[coord_scope, coord_scope, coord_scope], 
                                          grid_coarse=384, 
                                          grid_fine=3,
                                          deg=self.deg, 
                                          sigma_init=self.sigma_init, 
                                          sigma_default=self.sigma_default,
                                          device='cuda')
        os.makedirs(f'logs/{self.hparams.exp_name}/ckpts', exist_ok=True)
        self.nerftree_path = os.path.join(f'logs/{self.hparams.exp_name}/ckpts', 'nerftree.pt')
        if self.hparams.ckpt_path != None and os.path.exists(self.nerftree_path):
            voxels_dict = torch.load(self.nerftree_path)
            self.nerf_tree.sigma_voxels_coarse = voxels_dict['sigma_voxels_coarse']
        
        self.xyz_min = self.nerf_tree.xyz_min
        self.xyz_max = self.nerf_tree.xyz_max
        self.xyz_scope = self.nerf_tree.xyz_scope
        self.grid_coarse = self.nerf_tree.grid_coarse
        self.grid_fine = self.nerf_tree.grid_fine
        self.res_coarse = self.nerf_tree.res_coarse
        self.res_fine = self.nerf_tree.res_fine
        
    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs
    
    def sigma2weights(self, deltas, sigmas):
        # compute alpha by the formula (3)
        # if self.training:
        noise = torch.randn(sigmas.shape, device=sigmas.device)
        sigmas = sigmas + noise

        # alphas = 1-torch.exp(-deltas*torch.nn.ReLU()(sigmas)) # (N_rays, N_samples_)
        alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas)) # (N_rays, N_samples_)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        return weights, alphas
    
    def render_rays(self, 
                models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                noise_std=0.0,
                N_importance=0,
                chunk=1024*32,
                white_back=False
                ):

        def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, idx_render):
            N_samples_ = xyz_.shape[1]
            # Embed directions
            xyz_ = xyz_[idx_render[:, 0], idx_render[:, 1]].view(-1, 3) # (N_rays*N_samples_, 3)
            view_dir = dir_.unsqueeze(1).expand(-1, N_samples_, -1)
            view_dir = view_dir[idx_render[:, 0], idx_render[:, 1]]
            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            out_chunks = []
            for i in range(0, B, chunk):
                out_chunks += [model(embedding_xyz(xyz_[i:i+chunk]), view_dir[i:i+chunk])]
            out = torch.cat(out_chunks, 0)
           
            out_rgb = torch.full((N_rays, N_samples_, 3), 1.0, device=device)
            out_sigma = torch.full((N_rays, N_samples_, 1), self.sigma_default, device=device)
            out_sh = torch.full((N_rays, N_samples_, self.dim_sh), 0.0, device=device)
            out_defaults = torch.cat([out_sigma, out_rgb, out_sh], dim=2)
            out_defaults[idx_render[:, 0], idx_render[:, 1]] = out
            out = out_defaults

            sigmas, rgbs, shs = torch.split(out, (1, 3, self.dim_sh), dim=-1)
            del out
            sigmas = sigmas.squeeze(-1)
                    
            # Convert these values using volume rendering (Section 4)
            deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
            
            weights, alphas = self.sigma2weights(deltas, sigmas)

            weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

            # compute final weighted outputs
            rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
            depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

            if white_back:
                rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

            return rgb_final, depth_final, weights, sigmas, shs

        # Extract models from lists
        model_coarse = models[0]
        embedding_xyz = embeddings[0]
        device = rays.device
        is_training = model_coarse.training
        result = {}

        # Decompose the inputs
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        # Embed direction
        dir_embedded = None
        
        N_samples_coarse = self.N_samples_coarse
        z_vals_coarse = self.z_vals_coarse.clone().expand(N_rays, -1)
        if is_training:
            delta_z_vals = torch.empty(N_rays, 1, device=device).uniform_(0.0, self.distance/N_samples_coarse)
            z_vals_coarse = z_vals_coarse + delta_z_vals
        
        xyz_sampled_coarse = rays_o.unsqueeze(1) + \
                             rays_d.unsqueeze(1) * z_vals_coarse.unsqueeze(2) # (N_rays, N_samples_coarse, 3)

        xyz_coarse = xyz_sampled_coarse.reshape(-1, 3)

        # valid sampling
        sigmas = self.nerf_tree.query_coarse(xyz_coarse, type='sigma').reshape(N_rays, N_samples_coarse)
        
        # update density voxel during coarse training
        if is_training and self.nerf_tree.voxels_fine == None: 
            with torch.no_grad():
                # introduce uniform sampling, not necessary
                sigmas[torch.rand_like(sigmas[:, 0]) < self.hparams.uniform_ratio] = self.sigma_init 

                if self.hparams.warmup_step > 0 and self.trainer.global_step <= self.hparams.warmup_step:
                    # during warmup, treat all points as valid samples
                    idx_render_coarse = torch.nonzero(sigmas >= -1e10).detach()
                else:
                    # or else, treat points whose density > 0 as valid samples
                    idx_render_coarse = torch.nonzero(sigmas > 0.0).detach()

            rgb_coarse, depth_coarse, weights_coarse, sigmas_coarse, _ = \
                inference(model_coarse, embedding_xyz, xyz_sampled_coarse, rays_d,
                        dir_embedded, z_vals_coarse, idx_render_coarse)
            result['rgb_coarse'] = rgb_coarse
            result['z_vals_coarse'] = self.z_vals_coarse
            result['depth_coarse'] = depth_coarse
            result['sigma_coarse'] = sigmas_coarse
            result['weight_coarse'] = weights_coarse
            result['opacity_coarse'] = weights_coarse.sum(1)
            result['num_samples_coarse'] = torch.FloatTensor([idx_render_coarse.shape[0] / N_rays])       
            
            # update 
            xyz_coarse_ = xyz_sampled_coarse[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
            sigmas_coarse_ = sigmas_coarse.detach()[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
            self.nerf_tree.update_coarse(xyz_coarse_, sigmas_coarse_, self.hparams.beta)
        
        # deltas_coarse = self.deltas_coarse
        with torch.no_grad():
            deltas_coarse = z_vals_coarse[:, 1:] - z_vals_coarse[:, :-1] # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas_coarse[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas_coarse = torch.cat([deltas_coarse, delta_inf], -1)  # (N_rays, N_samples_)
            weights_coarse, _ = self.sigma2weights(deltas_coarse, sigmas)
            weights_coarse = weights_coarse.detach()

        # pivotal sampling
        idx_render = torch.nonzero(weights_coarse >= min(self.hparams.weight_threashold, weights_coarse.max().item()))
        scale = N_importance
        z_vals_fine = self.z_vals_fine.clone()
        if is_training:
            z_vals_fine = z_vals_fine + delta_z_vals

        idx_render = idx_render.unsqueeze(1).expand(-1, scale, -1)  # (B, scale, 2)
        idx_render_fine = idx_render.clone()
        idx_render_fine[..., 1] = idx_render[..., 1] * scale + (torch.arange(scale, device=device)).reshape(1, scale)
        idx_render_fine = idx_render_fine.reshape(-1, 2)

        if idx_render_fine.shape[0] > N_rays * 64:
            indices = torch.randperm(idx_render_fine.shape[0])[:N_rays * 64]
            idx_render_fine = idx_render_fine[indices]
        
        xyz_sampled_fine = rays_o.unsqueeze(1) + \
                            rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(2) # (N_rays, N_samples*scale, 3)

        # if self.nerf_tree.voxels_fine != None:
        #     xyz_norm = (xyz_sampled_fine - self.xyz_min) / self.xyz_scope
        #     xyz_norm = (xyz_norm * self.res_fine).long().float() / float(self.res_fine)
        #     xyz_sampled_fine = xyz_norm * self.xyz_scope + self.xyz_min

        model_fine = models[1]
        rgb_fine, depth_fine, _, sigmas_fine, shs_fine = \
            inference(model_fine, embedding_xyz, xyz_sampled_fine, rays_d,
                    dir_embedded, z_vals_fine, idx_render_fine)
        
        if is_training and self.nerf_tree.voxels_fine != None:
            with torch.no_grad():
                xyz_fine_ = xyz_sampled_fine[idx_render_fine[:, 0], idx_render_fine[:, 1]]
                sigmas_fine_ = sigmas_fine.detach()[idx_render_fine[:, 0], idx_render_fine[:, 1]].unsqueeze(-1)
                shs_fine_ = shs_fine.detach()[idx_render_fine[:, 0], idx_render_fine[:, 1]]
                self.nerf_tree.update_fine(xyz_fine_, sigmas_fine_, shs_fine_)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['num_samples_fine'] = torch.FloatTensor([idx_render_fine.shape[0] / N_rays])

        return result

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        # if self.nerf_tree.voxels_fine == None or self.models[0].training:
        #     chunk = self.hparams.chunk
        # else:
        #     chunk = B // 8
        chunk = self.hparams.chunk
        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                self.render_rays(self.models,
                            self.embeddings,
                            rays[i:i+chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back
                                )
                            
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results
    
    def optimizer_step(self, epoch=None, 
                    batch_idx=None, 
                    optimizer=None, 
                    optimizer_idx=None, 
                    optimizer_closure=None, 
                    on_tpu=None, 
                    using_native_amp=None, 
                    using_lbfgs=None):
        if self.hparams.warmup_step > 0 and self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.warmup_step))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        self.train_dataset = dataset(split='train', **kwargs)
        if self.hparams.dataset_name == 'blender':
            self.val_dataset = dataset(split='test', **kwargs)
        else:
            self.val_dataset = dataset(split='val', **kwargs)
        
        self.near = self.train_dataset.near
        self.far = self.train_dataset.far
        self.distance = self.far - self.near
        near = torch.full((1,), self.near, dtype=torch.float32, device='cuda')
        far = torch.full((1,), self.far, dtype=torch.float32, device='cuda')

        # z_vals_coarse
        self.N_samples_coarse = self.hparams.N_samples
        z_vals_coarse = torch.linspace(0, 1, self.N_samples_coarse, device='cuda') # (N_samples_coarse)
        if not self.hparams.use_disp: # use linear sampling in depth space
            z_vals_coarse = near * (1-z_vals_coarse) + far * z_vals_coarse
        else: # use linear sampling in disparity space
            z_vals_coarse = 1/(1/near * (1-z_vals_coarse) + 1/far * z_vals_coarse)   # (N_rays, N_samples_coarse)
        self.z_vals_coarse = z_vals_coarse.unsqueeze(0)

        # z_vals_fine
        self.N_samples_fine = self.hparams.N_samples * self.hparams.N_importance
        z_vals_fine = torch.linspace(0, 1, self.N_samples_fine, device='cuda') # (N_samples_coarse)
        if not self.hparams.use_disp: # use linear sampling in depth space
            z_vals_fine = near * (1-z_vals_fine) + far * z_vals_fine
        else: # use linear sampling in disparity space
            z_vals_fine = 1/(1/near * (1-z_vals_fine) + 1/far * z_vals_fine)   # (N_rays, N_samples_coarse)
        self.z_vals_fine = z_vals_fine.unsqueeze(0)

        # delta
        deltas = self.z_vals_coarse[:, 1:] - self.z_vals_coarse[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        self.deltas_coarse = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        deltas = self.z_vals_fine[:, 1:] - self.z_vals_fine[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        self.deltas_fine = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_idx):
        self.log('train/lr', get_learning_rate(self.optimizer), on_step=True, prog_bar=True)
        rays, rgbs = self.decode_batch(batch)
        extract_time = self.current_epoch >= (self.hparams.num_epochs - 1)

        if extract_time and self.nerf_tree.voxels_fine == None:
            self.nerf_tree.create_voxels_fine()
    
        results = self(rays)

        loss_total = loss_rgb = self.loss(results, rgbs)
        self.log('train/loss_rgb', loss_rgb, on_step=True)

        # if self.hparams.weight_tv > 0.0:
        #     alphas_coarse = results['alpha_coarse']
        #     loss_tv = self.hparams.weight_tv * (alphas_coarse[:, 1:] - alphas_coarse[:, :-1]).pow(2).mean()
        #     self.log('train/loss_tv', loss_tv, on_step=True)
        #     loss_total += loss_tv

        self.log('train/loss_total', loss_total, on_step=True)

        if 'num_samples_coarse' in results:
            self.log('train/num_samples_coarse', results['num_samples_coarse'].mean(), on_step=True)

        if 'num_samples_fine' in results:
            self.log('train/num_samples_fine', results['num_samples_fine'].mean(), on_step=True)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_idx % 1000 == 0 and self.nerf_tree.voxels_fine == None:
            fig = plt.figure()
            depths = results['z_vals_coarse'][0].detach().cpu().numpy()
            sigmas = torch.nn.ReLU()(results['sigma_coarse'][0]).detach().cpu().numpy()
            weights = results['weight_coarse'][0].detach().cpu().numpy()
            near = self.near - (self.far - self.near) * 0.1
            far = self.far + (self.far - self.near) * 0.1
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
            ax[0].scatter(x=depths, y=sigmas)
            ax[0].set_xlabel('Depth', fontsize=16)
            ax[0].set_ylabel('Density', fontsize=16)
            ax[0].set_title('Density Distribution of a Ray', fontsize=16)
            ax[0].set_xlim([near, far])

            ax[1].scatter(x=depths, y=weights)
            ax[1].set_xlabel('Depth', fontsize=16)
            ax[1].set_ylabel('Weight', fontsize=16)
            ax[1].set_title('Weight Distribution of a Ray', fontsize=16)
            ax[1].set_xlim([near, far])

            self.logger.experiment.add_figure('train/distribution',
                                               fig, self.global_step)
            plt.close()

        feats = {}
        with torch.no_grad():
            psnr_fine = psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr_fine', psnr_fine, on_step=True, prog_bar=True)

            if 'rgb_coarse' in results:
                psnr_coarse = psnr(results['rgb_coarse'], rgbs)
                self.log('train/psnr_coarse', psnr_coarse, on_step=True)

        if batch_idx % 1000 == 0:
            torch.cuda.empty_cache()
        return loss_total

    def validation_step(self, batch, batch_idx):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)

        results = self(rays)
        log = {}
        log['val_loss'] = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        
        W, H = self.hparams.img_wh
        img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        img = img.permute(2, 0, 1) # (3, H, W)
        img_path = os.path.join(f'logs/{hparams.exp_name}/video', "%06d.png" % batch_idx)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        transforms.ToPILImage()(img).convert("RGB").save(img_path)
        
        idx_selected = 0
        if batch_idx == idx_selected:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            stack = torch.stack([img_gt, img]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/gt_pred',
                                               stack, self.global_step)
            
            img_path = os.path.join(f'logs/{hparams.exp_name}', f'epoch_{self.current_epoch}.png')
            transforms.ToPILImage()(img).convert("RGB").save(img_path)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        torch.cuda.empty_cache()
        return log

    def validation_epoch_end(self, outputs):
        log = {}
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        num_voxels_coarse = torch.logical_and(self.nerf_tree.sigma_voxels_coarse > 0, self.nerf_tree.sigma_voxels_coarse != self.sigma_init).nonzero().shape[0]
        self.log('val/loss', mean_loss, on_epoch=True)
        self.log('val/psnr', mean_psnr, on_epoch=True, prog_bar=True)
        self.log('val/num_voxels_coarse', num_voxels_coarse, on_epoch=True)

        # save sparse voxels
        sigma_voxels_coarse_clean = self.nerf_tree.sigma_voxels_coarse.clone()
        sigma_voxels_coarse_clean[sigma_voxels_coarse_clean == self.sigma_init] = self.sigma_default
        voxels_dict = {
            'sigma_voxels_coarse': sigma_voxels_coarse_clean,
            'index_voxels_coarse': self.nerf_tree.index_voxels_coarse,
            'voxels_fine': self.nerf_tree.voxels_fine
        }
        torch.save(voxels_dict, self.nerftree_path)

        img_paths = glob.glob(f'logs/{hparams.exp_name}/video/*.png')
        writer = imageio.get_writer(f'logs/{hparams.exp_name}/video/video_{self.current_epoch}.mp4', fps=40)
        for im in img_paths:
            writer.append_data(imageio.imread(im))
        writer.close()


if __name__ == '__main__':
    hparams = get_opts()
    system = EfficientNeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'logs/{hparams.exp_name}/ckpts',
                                                                '{epoch:d}'),
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=5,)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      gpus=hparams.num_gpus,
                      strategy='ddp' if hparams.num_gpus>1 else None,
                      benchmark=True)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)