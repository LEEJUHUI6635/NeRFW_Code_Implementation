from JH_model import Stratified_Sampling, Positional_Encoding, Hierarchical_Sampling, NeRF, viewing_directions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# train, validation, test, Classic Volume Rendering 함수 or class + Data_loader 설정
import numpy as np
import cv2 as cv
import os
import time
import tqdm
import sys
# train 할 때에는 random하게 ray를 섞어야 하기 때문에, ray를 합쳐 하나의 image로 만드는 작업 -> 하나의 함수
# learning rate decay -> iteration이 한 번씩 돌 때마다

# Checkpoints 저장 -> epoch 마다 저장 or 마지막 epoch만 저장
class Save_Checkpoints(object):
    def __init__(self, epoch, model, optimizer, loss, save_path, select='model'): # select -> model or embedding
        # epoch, model_state_dict, optimizer_state_dict, loss 저장
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.save_path = save_path
        self.select = select
        # epoch 마다 저장 or 마지막 epoch만 저장 -> optional
        if self.select == 'model':
            self.save_checkpoints_model()
        elif self.select == 'embedding':
            self.save_checkpoints_embedding()
            
    def save_checkpoints_model(self):
        torch.save({'epoch': self.epoch, 
                    'model': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(), 
                    'loss': self.loss}, os.path.join(self.save_path, 'checkpoints_{}.pt'.format(self.epoch))) # self.save_path + 'checkpoints_{}.pt'
    def save_checkpoints_embedding(self):
        torch.save({'epoch': self.epoch,
                    'model': self.model.state_dict()}, os.path.join(self.save_path, 'checkpoints_{}.pt'.format(self.epoch)))
        
class Solver(object):
    def __init__(self, data_loader, val_data_loader, test_data_loader, config, i_val, height, width):
        self.data_loader = data_loader # rays dataloader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        
        # iterations
        self.resume_iters = config.resume_iters
        self.nb_epochs = config.nb_epochs
        self.save_val_iters = config.save_val_iters
        self.save_model_iters = config.save_model_iters # 15
        self.batch_size = config.batch_size
        self.coarse_num = config.coarse_num # 64
        self.fine_num = config.fine_num # 64
        self.sample_num = self.coarse_num # 64
        self.near = config.near # 0.
        self.far = config.far # 1.
        self.L_pts = config.L_pts # 10
        self.L_dirs = config.L_dirs # 4
        self.learning_rate = config.learning_rate
        
        # appearance embedding vector, transient embedding vector
        self.appearance_embedded = config.appearance_embedded
        self.transient_embedded = config.transient_embedded
        self.custom_dataset = config.custom_dataset
        self.appearance_embedding_word = config.appearance_embedding_word
        self.transient_embedding_word = config.transient_embedding_word
        self.appearance_channel = config.appearance_embedding_dim
        self.transient_channel = config.transient_embedding_dim
        
        # pts_channel, output_channel, dir_channel 설정
        self.pts_channel = 3 + 2 * self.L_pts * 3 # 3 + 2 x 10 x 3
        self.output_channel = 9 # static rgb + static density + uncertainty + transient rgb + transient density
        self.dir_channel = 3 + 2 * self.L_dirs * 3 # 3 + 2 x 4 x 3
        
        # save path
        self.save_results_path = config.save_results_path
        self.save_train_path = config.save_train_path
        self.save_test_path = config.save_test_path
        self.save_model_path = config.save_model_path
        self.save_coarse_path = config.save_coarse_path
        self.save_fine_path = config.save_fine_path
        self.save_appearance_embedding_path = config.save_appearance_embedding_path
        self.save_transient_embedding_path = config.save_transient_embedding_path
        
        # validation
        self.i_val = i_val
        self.factor = config.factor
        self.height = height
        self.width = width
    
        self.basic_setting()
    
    # 고민 : optimizer가 학습할 parameter -> 한 번에 이렇게 학습해도 되나?
    def basic_setting(self): # Q. 2개의 network를 학습?
        # TODO : 학습해야 하는 appearance embedding vector + transient embedding vector
        # TODO : coarse model만 먼저 학습해보기 -> Debugging
        if self.appearance_embedded == True:
            self.appearance_embedding_vector = torch.nn.Embedding(self.appearance_embedding_word, self.appearance_channel) # image 개수대로 appearance embedding vector 생성
        if self.transient_embedded == True:
            self.transient_embedding_vector = torch.nn.Embedding(self.transient_embedding_word, self.transient_channel) # image 개수대로 transient embedding vector 생성
        
        # model -> Coarse + Fine
        # Coarse + Fine Network
        # 고민 -> 두 개의 network가 아니라 한 개의 network를 학습해야 하는 것이 아닌가? 즉, forward 부분도 하나로 통일해야 gradient가 한 번에 학습되는 것이 아닌가?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coarse_model = NeRF(self.pts_channel, self.output_channel, self.dir_channel, 0, self.transient_channel, self.batch_size, self.sample_num, self.device).to(self.device)
        # self.coarse_model = torch.nn.DataParallel(self.coarse_model, device_ids=self.device)
        grad_variables = list(self.coarse_model.parameters())
        
        self.fine_model = NeRF(self.pts_channel, self.output_channel, self.dir_channel, self.appearance_channel, self.transient_channel, self.batch_size, self.sample_num, self.device).to(self.device)
        # self.fine_model = torch.nn.DataParallel(self.fine_model, device_ids=self.device)
        grad_variables += list(self.fine_model.parameters())
        
        # appearance embedding vector + transient embedding vector -> 학습 대상에 추가
        if self.appearance_embedded == True:
            grad_variables += list(self.appearance_embedding_vector.parameters())
        if self.transient_embedded == True:
            grad_variables += list(self.transient_embedding_vector.parameters())
        
        # optimizer
        self.optimizer = optim.Adam(params=grad_variables, lr=self.learning_rate, betas=(0.9, 0.999))
        # learning rate decay를 실행시키기 위해서는 self.learning_rate가 self.optimizer에 반영되어야 한다. 하지만, 위 함수는 한 번만 정의되기 때문에, 새로운 learning rate를 반영하지 못할 것이다.
        # check -> optimizer를 출력해보면 된다.
        
        # loss function
        # coarse -> 기존의 NeRF와 동일, coarse color loss function
        # fine -> fine color loss function, uncertainty loss function, transient density function
        # coarse color loss function
        self.coarse_color_loss = lambda x, y : 0.5 * torch.mean((x - y) ** 2)
        # fine color loss function
        self.fine_color_loss = lambda x, y, z : 0.5 * torch.mean((x - y) ** 2 / z ** 2)
        # uncertainty loss function
        self.uncertainty_loss = lambda z : 3 + torch.mean((torch.log(z)) ** 2) # positive
        # transient density loss function
        self.transient_density_loss = lambda r : torch.mean(0.01 * r) # r -> transient density
        
        # evaluation metric -> PSNR
        self.psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(self.device))

    # TODO : train과 test 구별하여 코드 구현, coarse와 fine 또한 구별
    # Classic Volume Rendering
    # self.classic_volume_rendering(outputs, z_vals, rays, self.device)
    # def classic_volume_rendering(self, raw, z_vals, rays, device): # input -> Network의 outputs [1024, 64, 4] + z_vals / output -> 2D color [1024, 3] -> rgb
    #     rays_d = rays[:,1:2,:] # viewing direction -> [1024, 1, 3]
    #     raw = raw.to(self.device) # [static_rgb_outputs, static_density_outputs, uncertainty_outputs, transient_rgb_outputs, transient_density_outputs]
    #     z_vals = z_vals.to(self.device)
    #     rays = rays.to(self.device)
    #     rays_d = rays_d.to(self.device)
        
    #     # static
    #     static_rgb_3d = torch.sigmoid(raw[:,:,:3]) # [1024, 64, 3]
    #     static_rgb_3d = static_rgb_3d.to(self.device)
    #     static_density = raw[:,:,3:4] # [1024, 64, 1]
    #     static_density = static_density.to(self.device)
        
    #     # transient
    #     transient_rgb_3d = torch.sigmoid(raw[:,:,5:8]) # [1024, 64, 3]
    #     transient_rgb_3d = transient_rgb_3d.to(self.device)
    #     transient_density = raw[:,:,8:] # [1024, 64, 1]
    #     transient_density = transient_density.to(self.device)
    #     uncertainty = raw[:,:,4:5] # [1024, 64, 1]
    #     uncertainty = uncertainty.to(self.device)
        
    #     dists = z_vals[:,1:] - z_vals[:,:-1] # [1024, 63]
    #     dists = dists.to(device)
    #     # print(dists.shape) # [1024, 63]
    #     dists = torch.cat([dists, (torch.Tensor([1e10]).expand(dists.shape[0], 1)).to(self.device)], dim=-1)
    #     # print(dists.shape) # [1024, 64]
    #     dists = dists * torch.norm(rays_d, dim=-1)
    #     # print(dists.shape) # [1024, 64]
    #     dists = dists.to(self.device)
    #     active_func = nn.ReLU()
    #     noise = torch.randn_like(dists)
    #     noise = noise.to(self.device)
        
    #     # static
    #     static_alpha = 1 - torch.exp(-active_func((static_density.squeeze() + noise) * dists))
        
    #     # transient
    #     transient_alpha = 1 - torch.exp(-active_func((transient_density.squeeze() + noise) * dists))
        
    #     # static + transient
    #     alpha = 1 - torch.exp(-active_func((static_density.squeeze() + transient_density.squeeze() + noise) * dists))
    #     transmittance = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(self.device), (1 - alpha + 1e-10).to(self.device)], dim=-1), dim=-1)[:,:-1]
        
    #     # static
    #     static_weights = static_alpha * transmittance
        
    #     # transient
    #     transient_weights = transient_alpha * transmittance
        
    #     # static + transient
    #     weights = alpha * transmittance
        
    #     # static 
    #     static_rgb_2d = torch.sum(static_weights[...,None] * static_rgb_3d, dim=-2)
    #     # transient
    #     transient_rgb_2d = torch.sum(transient_weights[...,None] * transient_rgb_3d, dim=-2)
    #     # static + transient
    #     rgb_2d = static_rgb_2d + transient_rgb_2d
        
    #     # uncertainty
    #     # print((transient_weights * uncertainty.squeeze()).shape) # [1024, 64]
    #     beta = torch.sum(transient_weights * uncertainty.squeeze(), dim=1) + 0.03
    #     # print(beta.shape) # [1024]
        
    #     return rgb_2d, weights, beta
    
    # TODO : GPU-utils 높이고 CPU-utils 낮추기 위해, classic volume rendering 수식에서 device 사용 x
    # TODO : Coarse model -> static_rgb_2d, static_density / Fine model -> static_rgb_2d, static_density, uncertainty, transient_rgb_2d, transient_density
    def classic_volume_rendering(self, raw, z_vals, rays, sampling):
        # Coarse network -> input : raw = [rgb_3d, density],
        # Fine network -> input : raw = [static_rgb_3d, static_density, uncertainty, transient_rgb_3d, transient_density]
        rays_d = rays[:,1:2,:] # viewing direction -> [1024, 1, 3]
        
        # static
        static_rgb_3d = torch.sigmoid(raw[:,:,:3]) # [1024, 64, 3]
        static_density = raw[:,:,3:4] # [1024, 64, 1]
        
        if sampling.lower() == 'fine':
            # transient
            transient_rgb_3d = torch.sigmoid(raw[:,:,5:8]) # [1024, 64, 3]
            transient_density = raw[:,:,8:] # [1024, 64, 1]
            uncertainty = raw[:,:,4:5] # [1024, 64, 1]
        
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        # print(deltas.shape) # [2048, 64]
        # print(static_density.shape) # [2048, 64, 1]
        
        if sampling.lower() == 'coarse':
            alphas = 1-torch.exp(-deltas*static_density.squeeze())
        elif sampling.lower() == 'fine':
            static_alphas = 1-torch.exp(-deltas*static_density.squeeze()) # static alpha compositing
            transient_alphas = 1-torch.exp(-deltas*transient_density.squeeze()) # transient alpha compositing
            alphas = 1-torch.exp(-deltas*(static_density.squeeze()+transient_density.squeeze()))
        
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
        
        if sampling.lower() == 'fine':
            static_weights = static_alphas * transmittance
            transient_weights = transient_alphas * transmittance
        weights = alphas * transmittance # weights -> 'n1 n2 -> n1', 'sum'
        
        # debugging
        # weights = torch.sum(weights, dim=1)
        if sampling.lower() == 'coarse':
            rgb_2d = weights.unsqueeze(dim=-1) * static_rgb_3d
            rgb_2d = torch.sum(rgb_2d, dim=1)
            return rgb_2d, weights
        
        elif sampling.lower() == 'fine':
            static_rgb_map = static_weights.unsqueeze(dim=-1) * static_rgb_3d
            static_rgb_map = torch.sum(static_rgb_map, dim=1)
            transient_rgb_map = transient_weights.unsqueeze(dim=-1) * transient_rgb_3d
            transient_rgb_map = torch.sum(transient_rgb_map, dim=1)
            rgb_2d = static_rgb_map + transient_rgb_map
            
            beta = transient_weights * uncertainty.squeeze()
            beta = torch.sum(beta, dim=1) + 0.1
        
        # static_rgb_map = static_weights(n1 n2 -> n1 n2 1) * static_rgbs(n1 n2 c -> n1 c, sum)
        # transient_rgb_map = transient_weights(n1 n2 -> n1 n2 1) * transient_rgbs(n1 n2 c -> n1 c, sum)
        # beta = transient_weights * uncertainty(n1 n2 -> n1, sum)
        
        return rgb_2d, weights, beta, static_rgb_map, transient_rgb_map

    # def classic_volume_rendering(self, raw, z_vals, rays, device):
    #     rays_d = rays[:,1:2,:] # viewing direction -> [1024, 1, 3]
    #     raw = raw.to(self.device) # [static_rgb_outputs, static_density_outputs, uncertainty_outputs, transient_rgb_outputs, transient_density_outputs]
    #     z_vals = z_vals.to(self.device)
    #     rays = rays.to(self.device)
    #     rays_d = rays_d.to(self.device)
        
    #     # static
    #     static_rgb_3d = torch.sigmoid(raw[:,:,:3]) # [1024, 64, 3]
    #     static_rgb_3d = static_rgb_3d.to(self.device)
    #     static_density = raw[:,:,3:4] # [1024, 64, 1]
    #     static_density = static_density.to(self.device)
        
    #     # transient
    #     transient_rgb_3d = torch.sigmoid(raw[:,:,5:8]) # [1024, 64, 3]
    #     transient_rgb_3d = transient_rgb_3d.to(self.device)
    #     transient_density = raw[:,:,8:] # [1024, 64, 1]
    #     transient_density = transient_density.to(self.device)
    #     uncertainty = raw[:,:,4:5] # [1024, 64, 1]
    #     uncertainty = uncertainty.to(self.device)
        
    #     deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    #     delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    #     deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
    #     # print(deltas.shape) # [2048, 64]
    #     # print(static_density.shape) # [2048, 64, 1]
    #     static_alphas = 1-torch.exp(-deltas*static_density.squeeze()) # static alpha compositing
    #     transient_alphas = 1-torch.exp(-deltas*transient_density.squeeze()) # transient alpha compositing
    #     alphas = 1-torch.exp(-deltas*(static_density.squeeze()+transient_density.squeeze()))
        
    #     alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
    #     transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
        
    #     static_weights = static_alphas * transmittance
    #     transient_weights = transient_alphas * transmittance
        
    #     weights = alphas * transmittance # weights -> 'n1 n2 -> n1', 'sum'
        
    #     # debugging
    #     # weights = torch.sum(weights, dim=1)
        
    #     static_rgb_map = static_weights.unsqueeze(dim=-1) * static_rgb_3d
    #     static_rgb_map = torch.sum(static_rgb_map, dim=1)
    #     transient_rgb_map = transient_weights.unsqueeze(dim=-1) * transient_rgb_3d
    #     transient_rgb_map = torch.sum(transient_rgb_map, dim=1)

    #     beta = transient_weights * uncertainty.squeeze()
    #     beta = torch.sum(beta, dim=1)
        
    #     # static_rgb_map = static_weights(n1 n2 -> n1 n2 1) * static_rgbs(n1 n2 c -> n1 c, sum)
    #     # transient_rgb_map = transient_weights(n1 n2 -> n1 n2 1) * transient_rgbs(n1 n2 c -> n1 c, sum)
    #     # beta = transient_weights * uncertainty(n1 n2 -> n1, sum)
    #     rgb_2d = static_rgb_map + transient_rgb_map
    #     return rgb_2d, weights, beta, static_rgb_map, transient_rgb_map
    
    # *****net_chunk*****
    def train(self): # device -> dataset, model
        # 학습 재기 -> 마지막에 저장된 checkpoints의 epoch, coarse_model, fine_model, optimizer를 가져온다.
        start_iters = 0
        if self.resume_iters != None: # self.resume_iters
            # TODO : appearance embedding vector와 transient embedding vector의 checkpoints load
            coarse_ckpt = torch.load(os.path.join(self.save_coarse_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
            fine_ckpt = torch.load(os.path.join(self.save_fine_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
            if self.appearance_embedded == True:
                appearance_ckpt = torch.load(os.path.join(self.save_appearance_embedding_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
                self.appearance_embedding_vector.load_state_dict(appearance_ckpt['model'])
            if self.transient_embedded == True:
                transient_ckpt = torch.load(os.path.join(self.save_transient_embedding_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
                self.transient_embedding_vector.load_state_dict(transient_ckpt['model'])
            self.coarse_model.load_state_dict(coarse_ckpt['model'])
            self.fine_model.load_state_dict(fine_ckpt['model'])
            self.optimizer.load_state_dict(coarse_ckpt['optimizer'])
            # embedding vector에 대한 checkpoints loading
            start_iters = self.resume_iters + 1
            self.coarse_model.train()
            self.fine_model.train()
            # Q. self.appearance_embedding_vector.train()
            # self.transient_embedding_vector.train()
            
        # Time check
        start_time = time.time()
        # for epoch in range(start_iters, self.nb_epochs):
        for epoch in tqdm.tqdm(range(start_iters, self.nb_epochs)):
            # Dataloader -> 1024로 나눠 학습
            s = 0
            for idx, [rays, view_dirs, rays_t] in enumerate(self.data_loader): # Dataloader -> rays = rays_o + rays_d + rays_rgb / view_dirs
            # for idx, [rays, view_dirs] in enumerate(self.data_loader):
                # print(rays_t.shape) # 2048 -> batch_size
                
                rays = rays.float()
                view_dirs = view_dirs.float()
                batch_size = rays.shape[0]
                # view_dirs -> NDC 처리 전의 get_rays로부터
                view_dirs = viewing_directions(view_dirs) # [1024, 3]
                rays_o = rays[:,0,:]
                rays_d = rays[:,1,:]
                rays_rgb = rays[:,2,:] # True

                # Coarse model -> 기존 NeRF와 동일
                # Stratified Sampling -> rays_o + rays_d -> view_dirs x
                pts, z_vals = Stratified_Sampling(rays_o, rays_d, batch_size, self.sample_num, self.near, self.far, self.device).outputs()
                pts = pts.reshape(batch_size, self.coarse_num, 3) # sample_num, [1024, 64, 3]
                coarse_view_dirs = view_dirs[:,None].expand(pts.shape) # [1024, 64, 3]
                pts = pts.reshape(-1, 3) # [65536, 3], 65536 = 1024 x 64
                coarse_view_dirs = coarse_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                coarse_pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                coarse_view_dirs = Positional_Encoding(self.L_dirs).outputs(coarse_view_dirs) # viewing direction
                coarse_pts = coarse_pts.to(self.device)
                coarse_view_dirs = coarse_view_dirs.to(self.device)
                
                # # Appearance embedding + Transient embedding
                # appearance_embedded = self.appearance_embedding_vector(rays_appearance_t) # [1024, 48]
                # appearance_embedded = appearance_embedded.reshape(batch_size, 1, self.appearance_channel)
                # appearance_embedded = appearance_embedded.repeat(1, self.coarse_num, 1) # [1024, 64, 48]
                # appearance_embedded = appearance_embedded.reshape(-1, self.appearance_channel).to(self.device)
                # transient_embedded = self.transient_embedding_vector(rays_t) # [1024, 16]
                # transient_embedded = transient_embedded.reshape(batch_size, 1, self.transient_channel)
                # transient_embedded = transient_embedded.repeat(1, self.coarse_num, 1)
                # transient_embedded = transient_embedded.reshape(-1, self.transient_channel).to(self.device)
                
                # input -> pts + viewing_dirs + appearance embedding vector + transient embedding vector
                inputs = torch.cat([coarse_pts, coarse_view_dirs], dim=-1)
                inputs = inputs.to(self.device)
                # print(inputs.shape) # [65536, 154], 154 = 63 + 27 + 48 + 16

                # Coarse Network
                # debugging
                outputs = self.coarse_model(inputs, sampling='coarse')
                # outputs = self.coarse_model(inputs)
                outputs = outputs.reshape(batch_size, self.coarse_num, 4) # rgb + density
                # print(outputs.shape) # [1024, 64, 4]

                rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, 'coarse')
                rgb_2d = rgb_2d.to(self.device)
                rays_rgb = rays_rgb.to(self.device)
                
                # print(weights.shape) # [2048, 64]
                
                # Fine Network
                # Hierarchical sampling + viewing_directions
                fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
                fine_pts = fine_pts.reshape(batch_size, self.coarse_num + self.fine_num, 3) # [1024, 128, 3] -> [1024, self.coarse_num + self.fine_num, 3]
                
                fine_view_dirs = view_dirs[:,None].expand(fine_pts.shape) # [1024, 128, 3]
                fine_pts = fine_pts.reshape(-1, 3)
                fine_view_dirs = fine_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
                fine_view_dirs = Positional_Encoding(self.L_dirs).outputs(fine_view_dirs)
                fine_pts = fine_pts.to(self.device)
                fine_view_dirs = fine_view_dirs.to(self.device)
                
                # TODO : embedding vector를 어디에서 정의해야 하는가?
                # Appearance embedding + Transient embedding
                appearance_fine_embedded = self.appearance_embedding_vector(rays_t)
                appearance_fine_embedded = appearance_fine_embedded.reshape(batch_size, 1, self.appearance_channel)
                appearance_fine_embedded = appearance_fine_embedded.repeat(1, self.coarse_num + self.fine_num, 1) # [1024, 64, 48]
                appearance_fine_embedded = appearance_fine_embedded.reshape(-1, self.appearance_channel).to(self.device)
                
                transient_fine_embedded = self.transient_embedding_vector(rays_t) # [1024, 16]
                transient_fine_embedded = transient_fine_embedded.reshape(batch_size, 1, self.transient_channel)
                transient_fine_embedded = transient_fine_embedded.repeat(1, self.coarse_num + self.fine_num, 1)
                transient_fine_embedded = transient_fine_embedded.reshape(-1, self.transient_channel).to(self.device)

                # input -> pts + viewing_dirs + appearance embedding vector + transient embedding vector
                fine_inputs = torch.cat([fine_pts, fine_view_dirs, appearance_fine_embedded, transient_fine_embedded], dim=-1)
                fine_inputs = fine_inputs.to(self.device)
                
                # Fine model
                fine_outputs = self.fine_model(fine_inputs, sampling='fine')
                # fine_outputs = self.fine_model(fine_inputs)
                fine_outputs = fine_outputs.reshape(rays.shape[0], self.coarse_num + self.fine_num, 9) # 128 = self.coarse_num + self.fine_num
                
                transient_density = fine_outputs[:,:,-1:] # [1024, 128, 9]
                # print(transient_density.shape)
                
                # classic volume rendering
                fine_rgb_2d, _, fine_beta, *_ = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, 'fine')
                
                fine_beta = fine_beta.unsqueeze(dim=1)
                loss = self.coarse_color_loss(rgb_2d, rays_rgb) + self.fine_color_loss(fine_rgb_2d, rays_rgb, fine_beta) + self.uncertainty_loss(fine_beta) + self.transient_density_loss(transient_density)
                
                # optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # iteration n번 마다 출력
                print(idx, loss, rays_t)

                if idx == 0:
                    train_image_arr = fine_rgb_2d
                elif idx != 0 and train_image_arr.shape[0] <= self.height * self.width:
                    train_image_arr = torch.cat([train_image_arr, fine_rgb_2d], dim=0)
                
                if s == 0 and train_image_arr.shape[0] >= self.height * self.width:    
                    train_image_list = [train_image_arr[i:i+1,:] for i in range(self.height * self.width)]
                    train_image_arr = torch.cat(train_image_list, dim=0)
                    train_image = train_image_arr.reshape(self.height, self.width, 3)
                    train_image = train_image * 255.0
                    train_image = np.array(train_image.detach().cpu())
                    cv.imwrite('./results/train/train_image_{}.png'.format(epoch), train_image)
                    s += 1
            
            print('one epoch passed!')
            
            # 한 epoch가 지날 때마다,
            print(idx, loss)
            print('----{}s seconds----'.format(time.time() - start_time))
            
            # # Learning rate decay -> self.optimizer에도 적용되어야 한다.
            decay_rate = 0.1
            decay_steps = 250 * 1000
            new_lrate = self.learning_rate * (decay_rate ** (epoch / decay_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lrate

            # 한 epoch마다 model, optimizer 저장 -> 15 epoch마다
            # Save_Checkpoints -> 하나로 합치기
            if epoch % self.save_model_iters == 0 and epoch > 0: 
            # if epoch % 1 == 0 and epoch > 0:
                # TODO : appearance embedding vector와 transient embedding vector의 checkpoints 저장
                Save_Checkpoints(epoch, self.appearance_embedding_vector, None, None, self.save_appearance_embedding_path, 'embedding')
                Save_Checkpoints(epoch, self.transient_embedding_vector, None, None, self.save_transient_embedding_path, 'embedding')
                Save_Checkpoints(epoch, self.coarse_model, self.optimizer, loss, self.save_coarse_path, 'model')
                Save_Checkpoints(epoch, self.fine_model, self.optimizer, loss, self.save_fine_path, 'model')
            
            # # Validation -> Validataion Dataloader = rays + NDC space 전에 추출한 get_rays의 view_dirs -> Train과 똑같이 처리한다. 다만, rays_rgb는 가져올 필요 없다.
            # if epoch % self.save_val_iters == 0 and epoch > 0: # if epoch % 10 == 0 and epoch > 0:
            # # if epoch % 1 == 0 and epoch > 0:
            #     with torch.no_grad():
            #         val_image_list = []
            #         for idx, [rays, view_dirs] in enumerate(tqdm.tqdm(self.val_data_loader)): # rays + view_dirs
            #             batch_size = rays.shape[0]
            #             # view_dirs -> NDC 처리 전의 get_rays로부터
            #             view_dirs = viewing_directions(view_dirs) # [1024, 3]
            #             rays_o = rays[:,0,:]
            #             rays_d = rays[:,1,:]
            #             rays_rgb = rays[:,2,:] # True
                        
            #             # Stratified Sampling -> rays_o + rays_d -> view_dirs x
            #             pts, z_vals = Stratified_Sampling(rays_o, rays_d, batch_size, self.sample_num, self.near, self.far, self.device).outputs()
            #             pts = pts.reshape(batch_size, self.coarse_num, 3) # sample_num
            #             coarse_view_dirs = view_dirs[:,None].expand(pts.shape) # [1024, 64, 3]
            #             pts = pts.reshape(-1, 3)
            #             coarse_view_dirs = coarse_view_dirs.reshape(-1, 3)
                        
            #             # Positional Encoding
            #             coarse_pts = Positional_Encoding(self.L_pts).outputs(pts) # position
            #             coarse_view_dirs = Positional_Encoding(self.L_dirs).outputs(coarse_view_dirs) # viewing direction
            #             coarse_pts = coarse_pts.to(self.device)
            #             coarse_view_dirs = coarse_view_dirs.to(self.device)

            #             inputs = torch.cat([coarse_pts, coarse_view_dirs], dim=-1)
            #             inputs = inputs.to(self.device)
                        
            #             # Coarse Network
            #             outputs = self.coarse_model(inputs, sampling='coarse')
            #             # outputs = self.coarse_model(inputs)
            #             outputs = outputs.reshape(batch_size, self.coarse_num, 4)
            #             rgb_2d, weights = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
                        
            #             # Hierarchical sampling + viewing_directions
            #             fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
            #             fine_pts = fine_pts.reshape(batch_size, self.coarse_num + self.fine_num, 3) # [1024, 128, 3] -> 128 = self.coarse_num + self.fine_num
                        
            #             fine_view_dirs = view_dirs[:,None].expand(fine_pts.shape) # [1024, 128, 3]
            #             fine_pts = fine_pts.reshape(-1, 3)
            #             fine_view_dirs = fine_view_dirs.reshape(-1, 3)
                        
            #             # Positional Encoding
            #             fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
            #             fine_view_dirs = Positional_Encoding(self.L_dirs).outputs(fine_view_dirs)
            #             fine_pts = fine_pts.to(self.device)
            #             fine_view_dirs = fine_view_dirs.to(self.device)
            #             fine_inputs = torch.cat([fine_pts, fine_view_dirs], dim=-1)
            #             fine_inputs = fine_inputs.to(self.device)
                        
            #             # Fine model
            #             fine_outputs = self.fine_model(fine_inputs, sampling='fine')
            #             # fine_outputs = self.fine_model(fine_inputs)
            #             fine_outputs = fine_outputs.reshape(rays.shape[0], self.coarse_num + self.fine_num, 4) # 128 = self.coarse_num + self.fine_num

            #             # classic volume rendering
            #             fine_rgb_2d, fine_weights = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
            #             fine_rgb_2d = fine_rgb_2d.to(self.device)
            #             # print(fine_rgb_2d.shape) # [1024, 3]
            #             fine_rgb_2d = fine_rgb_2d.cpu().detach().numpy()
            #             fine_rgb_2d = (255*np.clip(fine_rgb_2d,0,1)).astype(np.uint8)
            #             val_image_list.append(fine_rgb_2d)
                        
            #         val_image_arr = np.concatenate(val_image_list, axis=0)
            #         val_image_arr = val_image_arr.reshape(2, self.height, self.width, 3) # validation image 개수만큼 -> flexible
            #         for i in range(2): # 2 -> flexible
            #             image = val_image_arr[i,:,:,:]
            #             # cv.imwrite('./results/val/{}_{}.png'.format(epoch, i), image)
            #             cv.imwrite(os.path.join(self.save_train_path, 'validation_epoch_{}_{}.png'.format(epoch, i)), image)
    
    def test(self):
        # render_only -> model checkpoints 가져오기
        # validation과 비슷하게 수행 -> test_dataloader에서 가져오기
        # 학습 재기 -> 마지막에 저장된 checkpoints의 coarse model과 fine model을 가져온다.
        
        # appearance embedding vector와 transient embedding vector에 대한 checkpoints를 load한다.
        appearance_embedding_ckpt = torch.load(os.path.join(self.save_appearance_embedding_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
        transient_embedding_ckpt = torch.load(os.path.join(self.save_transient_embedding_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
        self.appearance_embedding_vector.load_state_dict(appearance_embedding_ckpt['model'])
        self.transient_embedding_vector.load_state_dict(transient_embedding_ckpt['model'])
        self.appearance_embedding_vector.eval()
        self.transient_embedding_vector.eval()
        
        coarse_ckpt = torch.load(os.path.join(self.save_coarse_path, 'checkpoints_{}.pt'.format(self.resume_iters)))
        self.coarse_model.load_state_dict(coarse_ckpt['model'])
        self.coarse_model.eval()

        with torch.no_grad(): # 학습 x
            rgb_test_image_list = []
            static_test_image_list = []
            transient_test_image_list = []
            beta_test_image_list = []
            start_time = time.time()
            i = 0
            for idx, [rays, view_dirs] in enumerate(tqdm.tqdm(self.test_data_loader)): # rays + view_dirs
                batch_size = rays.shape[0]
                # view_dirs -> NDC 처리 전의 get_rays로부터
                view_dirs = viewing_directions(view_dirs) # [1024, 3]
                rays_o = rays[:,0,:]
                rays_d = rays[:,1,:]
                
                rays_JH = 1 * torch.ones(batch_size, dtype=torch.long)

                # Stratified Sampling -> rays_o + rays_d -> view_dirs x
                pts, z_vals = Stratified_Sampling(rays_o, rays_d, batch_size, self.sample_num, self.near, self.far, self.device).outputs()
                pts = pts.reshape(batch_size, self.coarse_num, 3) # sample_num
                coarse_view_dirs = view_dirs[:,None].expand(pts.shape) # [1024, 64, 3]
                pts = pts.reshape(-1, 3)
                coarse_view_dirs = coarse_view_dirs.reshape(-1, 3)
                
                # Positional Encoding
                coarse_pts = Positional_Encoding(self.L_pts).outputs(pts) # position
                coarse_view_dirs = Positional_Encoding(self.L_dirs).outputs(coarse_view_dirs) # viewing direction
                coarse_pts = coarse_pts.to(self.device)
                coarse_view_dirs = coarse_view_dirs.to(self.device)

                # Appearance embedding + Transient embedding
                # TODO : 지정하고 싶은 image의 index -> rays_t
                appearance_embedded = self.appearance_embedding_vector(rays_JH)
                appearance_embedded = appearance_embedded.reshape(batch_size, 1, self.appearance_channel)
                appearance_embedded = appearance_embedded.repeat(1, self.coarse_num, 1)
                appearance_embedded = appearance_embedded.reshape(-1, self.appearance_channel).to(self.device)
                
                transient_embedded = self.transient_embedding_vector(rays_JH)
                transient_embedded = transient_embedded.reshape(batch_size, 1, self.transient_channel)
                transient_embedded = transient_embedded.repeat(1, self.coarse_num, 1)
                transient_embedded = transient_embedded.reshape(-1, self.transient_channel).to(self.device)
                
                # input -> pts + viewing_dirs + appearance embedding vector + transient embedding vector
                inputs = torch.cat([coarse_pts, coarse_view_dirs, appearance_embedded, transient_embedded], dim=-1)
                inputs = inputs.to(self.device)

                # Coarse Network
                outputs = self.coarse_model(inputs, sampling='coarse')
                # outputs = self.coarse_model(inputs)
                outputs = outputs.reshape(batch_size, self.coarse_num, 9)
                rgb_2d, weights, beta, static_rgb_map, transient_rgb_map = self.classic_volume_rendering(outputs, z_vals, rays, self.device)
                
                
                # # Hierarchical sampling + viewing_directions
                # fine_pts, fine_z_vals = Hierarchical_Sampling(rays, z_vals, weights, batch_size, self.sample_num, self.device).outputs()
                # fine_pts = fine_pts.reshape(batch_size, self.coarse_num + self.fine_num, 3) # [1024, 128, 3], 128 = self.coarse_num + self.fine_num
                
                # fine_view_dirs = view_dirs[:,None].expand(fine_pts.shape) # [1024, 128, 3]
                # fine_pts = fine_pts.reshape(-1, 3)
                # fine_view_dirs = fine_view_dirs.reshape(-1, 3)
                
                # # Positional Encoding
                # fine_pts = Positional_Encoding(self.L_pts).outputs(fine_pts)
                # fine_view_dirs = Positional_Encoding(self.L_dirs).outputs(fine_view_dirs)
                # fine_pts = fine_pts.to(self.device)
                # fine_view_dirs = fine_view_dirs.to(self.device)
                # fine_inputs = torch.cat([fine_pts, fine_view_dirs], dim=-1)
                # fine_inputs = fine_inputs.to(self.device)
                
                # # Fine model
                # fine_outputs = self.fine_model(fine_inputs, sampling='fine')
                # # fine_outputs = self.fine_model(fine_inputs)
                # fine_outputs = fine_outputs.reshape(rays.shape[0], self.coarse_num + self.fine_num, 4) # 128 = self.coarse_num + self.fine_num

                # # classic volume rendering
                # fine_rgb_2d, fine_weights = self.classic_volume_rendering(fine_outputs, fine_z_vals, rays, self.device) # z_vals -> Stratified sampling된 후의 z_vals
                # fine_rgb_2d = fine_rgb_2d.to(self.device)
                # # print(fine_rgb_2d.shape) # [1024, 3]
                # fine_rgb_2d = fine_rgb_2d.cpu().detach().numpy()
                # fine_rgb_2d = (255*np.clip(fine_rgb_2d,0,1)).astype(np.uint8)
                # # 하나의 image 씩 -> batch_size가 1024가 아닐 때, (다른 dataset의 경우) image list의 길이가 378 x 504를 넘을 때, 하나의 이미지로 만든다.
                
                # tensor -> numpy, detach CUDA
                rgb_2d = rgb_2d.cpu().detach().numpy()
                rgb_2d = (255*np.clip(rgb_2d, 0, 1)).astype(np.uint8)
                
                static_rgb_map = static_rgb_map.cpu().detach().numpy()
                static_rgb_map = (255*np.clip(static_rgb_map, 0, 1)).astype(np.uint8)
                
                transient_rgb_map = transient_rgb_map.cpu().detach().numpy()
                transient_rgb_map = (255*np.clip(transient_rgb_map, 0, 1)).astype(np.uint8)

                rgb_test_image_list.append(rgb_2d)
                static_test_image_list.append(static_rgb_map)
                transient_test_image_list.append(transient_rgb_map)
                beta_test_image_list.append(beta)
                
                if batch_size != self.batch_size or len(rgb_test_image_list) >= self.height * self.width: # self.batch_size = 1024
                    print("hello")
                    rgb_test_image_arr = np.concatenate(rgb_test_image_list, axis=0)
                    print(rgb_test_image_arr)
                    print(rgb_test_image_arr.shape)
                    print(self.height)
                    print(self.width)
                    rgb_test_image_arr = rgb_test_image_arr.reshape(120, self.height, self.width, 3)
                    static_test_image_arr = np.concatenate(static_test_image_list, axis=0)
                    static_test_image_arr = static_test_image_arr.reshape(120, self.height, self.width, 3)
                    transient_test_image_arr = np.concatenate(transient_test_image_list, axis=0)
                    transient_test_image_arr = transient_test_image_arr.reshape(120, self.height, self.width, 3)
                    
                    # cv.imwrite('./results/test/{}.png'.format(), test_image_arr)
                    for i in range(120):
                        cv.imwrite(os.path.join(self.save_test_path, 'test_{}.png'.format(i)), rgb_test_image_arr[i,...])
                        cv.imwrite(os.path.join(self.save_test_path, 'static_test_{}.png'.format(i)), static_test_image_arr[i,...])
                        cv.imwrite(os.path.join(self.save_test_path, 'transient_test_{}.png'.format(i)), transient_test_image_arr[i,...])
                    
                    sys.exit()
                    rgb_test_image_list = [] # 이미지 1개 만들어내면, list 비우기
                    static_test_image_list = []
                    transient_test_image_list = []
                    i += 1