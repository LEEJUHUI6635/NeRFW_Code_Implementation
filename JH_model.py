import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class Stratified_Sampling(object):
    def __init__(self, rays_o, view_dirs, batch_size, sample_num, near, far, device):
        self.rays_o = rays_o
        self.rays_d = view_dirs
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.near = torch.tensor(near) # 0 -> scalar
        self.far = torch.tensor(far) # 1 -> scalar
        self.device = device
        self.z_sampling()
        # self.outputs()
        
        # *****NDC가 아닌 경우 -> z_sampling을 다르게 만들기(optional)*****
        # return z_vals
    def z_sampling(self): # pts = rays_o + rays_d + z_vals
        near = self.near.expand([self.batch_size, 1]) # 특정 크기의 array로 확장
        far = self.far.expand([self.batch_size, 1]) # 특정 크기의 array로 확장
        near = near.to(self.device)
        far = far.to(self.device)
        t_vals = torch.linspace(start=0., end=1., steps=self.sample_num) # 간격
        t_vals = t_vals.to(self.device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.to(self.device)
        mids = 0.5 * (z_vals[:,1:] + z_vals[:,:-1])
        mids = mids.to(self.device)
        upper = torch.cat([mids, z_vals[:,-1:]], dim=-1)
        upper = upper.to(self.device)
        lower = torch.cat([z_vals[:,:1], mids], dim=-1)
        lower = lower.to(self.device)
        t_vals = torch.rand(z_vals.shape).to(self.device)
        self.z_vals = lower + (upper - lower) * t_vals
        self.z_vals = self.z_vals.to(self.device)
        
    def outputs(self):
        self.rays_o = self.rays_o.to(self.device)
        self.rays_d = self.rays_d.to(self.device)
        self.z_vals = self.z_vals.to(self.device)
        
        # pts = self.rays_o + self.rays_d * self.z_vals[:,:,None]
        pts = self.rays_o[...,None,:] + self.rays_d[...,None,:] * self.z_vals[...,:,None] # [1024, 1, 3] + [1024, 1, 3] * [1024, 64, 1]
        pts = pts.reshape(-1, 3)
        z_vals = self.z_vals
        return pts, z_vals # [1024, 64, 3]

def viewing_directions(rays): # rays = [1024, 1, 3]
    rays_d = rays[:,0,:] 
    dirs = rays_d / torch.norm(input=rays_d, dim=-1, keepdim=True)
    return dirs

# Positional Encoding
class Positional_Encoding(object): # shuffle x 
    def __init__(self, L):
        self.L = L # pts : L = 10 / dirs : L = 4
    def outputs(self, x):
        freq_bands = torch.linspace(start=0, end=self.L - 1, steps=self.L)
        freq = 2 ** freq_bands 
        freq_list = []
        for i in freq:
            freq_list.append(torch.sin(x * i))
            freq_list.append(torch.cos(x * i))
        freq_arr = x
        for i in freq_list:
            freq_arr = torch.cat([freq_arr, i], dim=-1)
        return freq_arr

class Hierarchical_Sampling(object):
    def __init__(self, rays, z_vals, weights, batch_size, sample_num, device): # z_vals -> [1024, 64], weights -> [1024, 64]
        self.rays = rays # [1024, 3, 3] = [1024, 1, 3](rays_o) + [1024, 2, 3](rays_d) + [1024, 3, 3](rays_rgb)
        self.rays_o = self.rays[:,0:1,:]
        self.rays_d = self.rays[:,1:2,:]
        self.rays_rgb = self.rays[:,2:3,:]
        self.z_vals = z_vals
        self.weights = weights
        self.batch_size = batch_size
        self.sample_num = sample_num # fine sampling -> 64
        self.device = device
        self.z_fine_sampling()
    def z_fine_sampling(self):
        self.z_vals = self.z_vals.to(self.device)
        mids = 0.5 * (self.z_vals[:,1:] + self.z_vals[:,:-1])
        # print(self.weights.shape)
        weights = self.weights[:,1:-1].to(self.device)
        weights = weights + 1e-5 # 추후에 0으로 나뉘는 것을 방지
        pdf = weights / torch.norm(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:,:1]), cdf], dim=-1)
        u = torch.rand(size=[self.batch_size, self.sample_num])
        u = torch.Tensor(u)
        u = u.to(self.device)
        cdf = cdf.to(self.device)
        idx = torch.searchsorted(sorted_sequence=cdf, input=u, right=True)
        below = torch.max(torch.zeros_like(idx-1), idx-1) # below = idx - 1
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(idx), idx) # above = idx
        idx_ab = torch.stack([below, above], dim=-1) # index_above_below
        
        mat_size = [idx_ab.shape[0], idx_ab.shape[1], cdf.shape[-1]]
        cdf_idx = torch.gather(input=cdf.unsqueeze(1).expand(mat_size), dim=-1, index=idx_ab)
        mids_idx = torch.gather(input=mids.unsqueeze(1).expand(mat_size), dim=-1, index=idx_ab)
        denorm = cdf_idx[...,1] - cdf_idx[...,0]
        denorm = torch.where(denorm<1e-5, torch.ones_like(denorm), denorm)
        t = (u - cdf_idx[...,0]) / denorm
        z_fine_vals = mids_idx[...,0] + t * (mids_idx[...,1] - mids_idx[...,0])
        self.z_fine_vals = z_fine_vals.squeeze()
    def outputs(self):
        z_vals = torch.cat([self.z_vals, self.z_fine_vals], dim=-1) # [1024, 128]
        z_vals, _ = torch.sort(z_vals, dim=-1) # sorting
        self.rays_o = self.rays_o.to(self.device)
        self.rays_d = self.rays_d.to(self.device)
        z_vals = z_vals.to(self.device)
        fine_pts = self.rays_o + self.rays_d * z_vals[:,:,None]
        fine_z_vals = z_vals
        return fine_pts, fine_z_vals # [1024, 64+64, 3]

# input_channel = 3 / output_channel = 9
# *****Viewing direction -> optional하게 만들기*****
# appearance embedding -> 48, transient embedding -> 16
# static rgb -> sigmoid
# static density -> softplus
# transient rgb -> sigmoid
# transient density -> softplus
# uncertainty -> softplus

# Coarse -> appearance_channel = 0
# Fine -> appearance_channel = 48

# TODO : nn.Sequential x, nn.ModuleList o
# TODO : NeRF in the Wild network
class NeRF(nn.Module):
    def __init__(self, 
                 sample_num=64,
                 pts_channel=63, 
                 dir_channel=27, 
                 appearance_channel=48, 
                 transient_channel=16, 
                 num_layers_main=8, 
                 hidden_layers_main=256, # 128-dim
                 hidden_layers_static_rgb=128,
                 num_layers_transient=4,
                 hidden_layers_transient=128
                 ):
        super().__init__() # NeRF 클래스의 부모 클래스 초기화
        # input
        self.sample_num = sample_num
        self.pts_channel = pts_channel # positional encoding
        self.dir_channel = dir_channel # viewing direction encoding
        self.appearance_channel = appearance_channel
        self.transient_channel = transient_channel
        
        self.num_layers_main = num_layers_main # 8
        self.hidden_layers_main = hidden_layers_main # 256-dim
        self.hidden_layers_static_rgb = hidden_layers_static_rgb # 128-dim
        self.num_layers_transient = num_layers_transient # 4
        self.hidden_layers_transient = hidden_layers_transient # 128-dim
        
        # main network
        main_net = []
        for l in range(self.num_layers_main):
            if l == 0:
                in_dim = self.pts_channel # 63-dim
            elif l == 3:
                in_dim = self.pts_channel + self.hidden_layers_main # 63-dim + 256-dim
            else:
                in_dim = self.hidden_layers_main # 256-dim
            out_dim = hidden_layers_main # 256-dim
            
            main_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.main_net = nn.ModuleList(main_net)
        
        # static sigma network
        static_sigma_net = []
        static_sigma_net.append(nn.Linear(hidden_layers_main, 1, bias=False)) # [256, 1]
        self.static_sigma_net = nn.ModuleList(static_sigma_net)
        
        # embedding network
        embedding_net = []
        embedding_net.append(nn.Linear(hidden_layers_main, hidden_layers_main))
        self.embedding_net = nn.ModuleList(embedding_net)
        
        # static rgb network
        static_rgb_net = []
        for l in range(2):
            if l == 0:
                in_dim = self.dir_channel + self.appearance_channel + self.hidden_layers_main # 27-dim + 48-dim + 256-dim
            else: # 1
                in_dim = self.hidden_layers_static_rgb # 128-dim
            if l == 0:
                out_dim = self.hidden_layers_static_rgb # 128-dim
            else:
                out_dim = 3 # rgb
            
            static_rgb_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.static_rgb_net = nn.ModuleList(static_rgb_net)
        
        # transient network
        transient_net = []
        for l in range(num_layers_transient):
            if l == 0:
                in_dim = self.transient_channel + self.hidden_layers_main # 16-dim + 256-dim
            else:
                in_dim = self.hidden_layers_transient # 128-dim
            out_dim = self.hidden_layers_transient # 128-dim
            transient_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.transient_net = nn.ModuleList(transient_net)
        
        # transient sigma, transient rgb, beta
        transient_sigma = []
        transient_sigma.append(nn.Linear(self.hidden_layers_transient, 1)) # 128-dim -> 1-dim
        transient_rgb = []
        transient_rgb.append(nn.Linear(self.hidden_layers_transient, 3)) # 128-dim -> 3-dim
        transient_beta = []
        transient_beta.append(nn.Linear(self.hidden_layers_transient, 1)) # 128-dim -> 1-dim
        self.transient_sigma = nn.ModuleList(transient_sigma)
        self.transient_rgb = nn.ModuleList(transient_rgb)
        self.transient_beta = nn.ModuleList(transient_beta)
    
    # [batch_size * samples_num, input_dim]
    def forward(self, x): # input -> positional encoding + viewing direction encoding + appearance embedding vector + transient embedding vector
        acc_channel = self.pts_channel # 63-dim
        pts = x[:, :acc_channel] # 63-dim
        acc_channel += self.dir_channel # 63-dim + 27-dim = 90-dim
        dir = x[:, acc_channel-self.dir_channel:acc_channel]
        acc_channel += self.appearance_channel
        appearance_embedding_vector = x[:, acc_channel-self.appearance_channel:acc_channel]
        acc_channel += self.transient_channel
        transient_embedding_vector = x[:, acc_channel-self.transient_channel:]
        # print(pts.shape) # torch.Size([131072, 63])
        # print(dir.shape) # torch.Size([131072, 27])
        # print(appearance_embedding_vector.shape) # torch.Size([131072, 48])
        # print(transient_embedding_vector.shape) # torch.Size([131072, 16])
        h = pts
        # main network
        for l in range(self.num_layers_main):
            if l == 3:
                h = torch.cat([pts, h], dim=-1)
            h = self.main_net[l](h)
            h = F.relu(h, inplace=True)
        
        feature_map = h
        # print(h.shape)
        
        # static_sigma network
        h = self.static_sigma_net[0](h)
        static_sigma = F.softplus(h)
        
        # embedding network
        feature_map = self.embedding_net[0](feature_map)
        
        h = feature_map
        h = torch.cat([appearance_embedding_vector, dir, h], dim=-1)
        # static_rgb network
        for l in range(2):
            h = self.static_rgb_net[l](h)
            if l == 0:
                h = F.relu(h, inplace=True)
        static_rgb = torch.sigmoid(h)
        
        # transient network
        h = torch.cat([transient_embedding_vector, feature_map], dim=-1)
        for l in range(self.num_layers_transient):
            h = self.transient_net[l](h)
            h = F.relu(h, inplace=True)
        
        transient_feature_map = h
        
        # transient_sigma, transient_rgb, transient_beta
        transient_sigma = self.transient_sigma[0](transient_feature_map)
        transient_sigma = F.softplus(transient_sigma)
        transient_rgb = self.transient_rgb[0](transient_feature_map)
        transient_rgb = torch.sigmoid(transient_rgb)
        transient_beta = self.transient_beta[0](transient_feature_map)
        transient_beta = F.softplus(transient_beta)
        
        static_sigma = static_sigma.reshape([x.shape[0] // self.sample_num, self.sample_num, -1])
        static_rgb = static_rgb.reshape([x.shape[0] // self.sample_num, self.sample_num, -1])
        transient_sigma = transient_sigma.reshape([x.shape[0] // self.sample_num, self.sample_num, -1])
        transient_rgb = transient_rgb.reshape([x.shape[0] // self.sample_num, self.sample_num, -1])
        transient_beta = transient_beta.reshape([x.shape[0] // self.sample_num, self.sample_num, -1])
        
        return {
            'static_sigma': static_sigma,
            'static_rgb': static_rgb,
            'transient_sigma': transient_sigma,
            'transient_rgb': transient_rgb,
            'transient_beta': transient_beta
        }

# class NeRF(nn.Module):
#     def __init__(self, pts_channel, output_channel, dir_channel, appearance_channel, transient_channel, batch_size, sample_num, device):
#         super(NeRF, self).__init__()
#         self.pts_channel = pts_channel # [x, y, z] points -> 63
#         self.output_channel = output_channel # 9 = static_rgb(3) + static_density(1) + uncertainty(1) + transient_rgb(3) + transient_density(1)
#         self.hidden_channel = 256
#         self.hidden2_channel = 128
#         self.dir_channel = dir_channel # viewing direction -> 27
#         # appearance embedding + transient embedding
#         self.appearance_channel = appearance_channel # 48
#         self.transient_channel = transient_channel # 16
        
#         self.batch_size = batch_size # 1024
#         self.sample_num = sample_num # 64
#         self.device = device
        
#         # static rgb + static density + uncertainty + transient rgb + transient density
#         self.static_rgb_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 3), nn.Sigmoid()) # [128, 3]
#         self.static_density_outputs = nn.Sequential(nn.Linear(self.hidden_channel, 1), nn.Softplus()) # [256, 1]
#         self.uncertainty_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 1), nn.Softplus()) # [128, 1]
#         self.transient_rgb_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 3), nn.Sigmoid()) # [128, 3]
#         self.transient_density_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 1), nn.Softplus()) # [128, 1]
        
#         # forward에서 쓰일 block들
#         # position input을 받는 network
#         # appearance embedding input과 viewing direction input을 받는 network
#         # transient embedding input을 받는 network
        
#         self.residual()
#         self.main()
#         self.appearance()
#         self.transient()
    
#     # Coarse model : position(pts) -> 첫 번째 MLP -> static density, viewing direction -> 두 번째 MLP -> static rgb
#     # Fine model : position(pts) -> 첫 번째 MLP -> static density, appearance embedding vector + viewing direction -> 두 번째 MLP -> static rgb,
#     # transient embedding vector + 첫 번째 MLP의 feature vector -> 세 번째 MLP -> uncertainty, transient rgb, transient density
    
#     # Coarse + Fine
#     # skip connection
#     def residual(self):
#         # residual learning
#         # [63, 256] -> [256, 256] -> [256, 256] -> [256, 256]
#         self.residual_list = []
#         self.residual_list.append(nn.Linear(in_features=self.pts_channel, out_features=self.hidden_channel)) # [63, 256]
#         self.residual_list.append(nn.ReLU(inplace=True))
#         for i in range(3):
#             self.residual_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel))
#             self.residual_list.append(nn.ReLU(inplace=True))
#         self.residual_block = nn.Sequential(*self.residual_list)
    
#     # Coarse + Fine
#     def main(self):
#         self.main_list = []
#         self.main_list.append(nn.Linear(in_features=self.pts_channel + self.hidden_channel, out_features=self.hidden_channel))
#         self.main_list.append(nn.ReLU(inplace=True))
#         for i in range(4):
#             self.main_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel))
#             self.main_list.append(nn.ReLU(inplace=True))
#         self.main_block = nn.Sequential(*self.main_list)
    
#     # Coarse -> viewing direction만 input으로 받는다.
#     # Fine -> viewing direction + appearance embedding vector를 input으로 받는다.
#     def appearance(self):
#         self.appearance_list = []
#         self.appearance_list.append(nn.Linear(self.appearance_channel + self.dir_channel + self.hidden_channel, self.hidden2_channel)) # 48 + 27 + 256
#         self.appearance_list.append(nn.ReLU(inplace=True))
#         self.appearance_block = nn.Sequential(*self.appearance_list)
    
#     # Fine
#     def transient(self):
#         self.transient_list = []
#         self.transient_list.append(nn.Linear(self.hidden_channel + self.transient_channel, self.hidden2_channel)) # 256 + 16
#         self.transient_list.append(nn.ReLU(inplace=True))
#         for i in range(3):
#             self.transient_list.append(nn.Linear(self.hidden2_channel, self.hidden2_channel))
#             self.transient_list.append(nn.ReLU(inplace=True))
#         self.transient_block = nn.Sequential(*self.transient_list)
    
#     # TODO : coarse model -> 기존의 NeRF와 같이, positional encoding -> position(pts) + viewing direction / fine model -> position(pts) + viewing direction + appearance embedding vector + transient embedding vector
#     def forward(self, x, sampling): # forward() : gradient의 학습을 결정 -> coarse와 fine을 한 개로 통일해야 한다.
#         # input -> pts + viewing_dirs + appearance embedding vector + transient embedding vector
#         # coarse -> [65536, 154] = [1024 x 64, 63 + 27 + 48 + 16] / fine -> [131072, 154] = [1024 x 128, 63 + 27 + 48 + 16]
#         # TODO : 먼저 coarse model만 고려하기
#         if sampling.lower() == 'coarse':
#             sample_num = 64 # TODO : 변수로 치환
#         elif sampling.lower() == 'fine':
#             sample_num = 128 # TODO : 변수로 치환
            
#         # coarse input -> pts, dirs
#         # fine input -> pts, dirs, appearance embedding vector, transient embedding vector
        
#         pts = x[:,:self.pts_channel] # 63
#         acc_channel = self.pts_channel
#         dirs = x[:,acc_channel:acc_channel+self.dir_channel] # 27
#         acc_channel += self.dir_channel
        
#         if sampling.lower() == 'fine':
#             appearance_embedding = x[:,acc_channel:acc_channel+self.appearance_channel] # 48
#             acc_channel += self.appearance_channel
#             transient_embedding = x[:,acc_channel:] # 16
        
#         # Coarse model + Fine model
#         # residual learning
#         feature = self.residual_block(pts) # [65536, 256]
#         feature = torch.cat([pts, feature], dim=1) # [65536, 63 + 256]
#         # feature2 -> 두 번째 MLP -> static rgb
#         feature2 = self.main_block(feature) # [65536, 256]
        
#         # Coarse model -> input : viewing direction + feature2
#         # Fine model -> input : appearance embedding + viewing direction + feature2
#         # appearance block의 input
#         # debugging
#         if sampling.lower() == 'coarse':
#             appearance_input = torch.cat([dirs, feature2], dim=1)
#             # print(appearance_input.shape) # [65536, 331], 331 = 48 + 27 + 256
#         elif sampling.lower() == 'fine':
#             appearance_input = torch.cat([appearance_embedding, dirs, feature2], dim=1)
            
#         feature3 = self.appearance_block(appearance_input)
#         static_rgb_outputs = self.static_rgb_outputs(feature3)

#         # static density
#         # Debugging
#         static_density_outputs = self.static_density_outputs(feature2) # [65536, 3]
        
#         # Fine model
#         if sampling.lower() == 'fine':
#             # transient block의 input
#             transient_input = torch.cat([transient_embedding, feature2], dim=1)
#             # print(transient_input.shape) # [65536, 272], 272 = 256 + 16
#             feature4 = self.transient_block(transient_input)
#             # uncertainty
#             uncertainty_outputs = self.uncertainty_outputs(feature4)
#             # transient rgb
#             transient_rgb_outputs = self.transient_rgb_outputs(feature4)
#             # transient density
#             transient_density_outputs = self.transient_density_outputs(feature4)
        
#         # static + transient
#         static_outputs = torch.cat([static_rgb_outputs, static_density_outputs], dim=1)
        
#         if sampling.lower() == 'coarse':
#             outputs = static_outputs.reshape([x.shape[0] // sample_num, sample_num, -1])
#             return outputs # [1024, 64, 4]
        
#         # Fine model
#         transient_outputs = torch.cat([uncertainty_outputs, transient_rgb_outputs, transient_density_outputs], dim=1)

#         outputs = torch.cat([static_outputs, transient_outputs], dim=1) # [65536, 9], 9 = 3 + 1 + 3 + 1 + 1
#         outputs = outputs.reshape([x.shape[0] // sample_num, sample_num, self.output_channel]) # [1024, 128, 9]
        
#         return outputs