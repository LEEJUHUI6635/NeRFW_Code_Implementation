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

# TODO : 각 output의 activation function 확인
class NeRF(nn.Module):
    def __init__(self, pts_channel, output_channel, dir_channel, appearance_channel, transient_channel, batch_size, sample_num, device):
        super(NeRF, self).__init__()
        self.pts_channel = pts_channel # [x, y, z] points -> 63
        self.output_channel = output_channel # 9 = static_rgb(3) + static_density(1) + uncertainty(1) + transient_rgb(3) + transient_density(1)
        self.hidden_channel = 256
        self.hidden2_channel = 128
        self.dir_channel = dir_channel # viewing direction -> 27
        # appearance embedding + transient embedding
        self.appearance_channel = appearance_channel # 48
        self.transient_channel = transient_channel # 16
        
        self.batch_size = batch_size # 1024
        self.sample_num = sample_num # 64
        self.device = device
        
        # static rgb + static density + uncertainty + transient rgb + transient density
        self.static_rgb_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 3), nn.Sigmoid()) # [128, 3]
        self.static_density_outputs = nn.Sequential(nn.Linear(self.hidden_channel, 1), nn.Softplus()) # [256, 1]
        self.uncertainty_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 1), nn.Softplus()) # [128, 1]
        self.transient_rgb_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 3), nn.Sigmoid()) # [128, 3]
        self.transient_density_outputs = nn.Sequential(nn.Linear(self.hidden2_channel, 1), nn.Softplus()) # [128, 1]
        
        # forward에서 쓰일 block들
        # position input을 받는 network
        # appearance embedding input과 viewing direction input을 받는 network
        # transient embedding input을 받는 network
        self.residual()
        self.main()
        self.appearance()
        self.transient()
    
    # skip connection
    def residual(self):
        # residual learning
        # [63, 256] -> [256, 256] -> [256, 256] -> [256, 256]
        self.residual_list = []
        self.residual_list.append(nn.Linear(in_features=self.pts_channel, out_features=self.hidden_channel)) # [63, 256]
        self.residual_list.append(nn.ReLU(inplace=True))
        for i in range(3):
            self.residual_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel))
            self.residual_list.append(nn.ReLU(inplace=True))
        self.residual_block = nn.Sequential(*self.residual_list)
    
    def main(self):
        self.main_list = []
        self.main_list.append(nn.Linear(in_features=self.pts_channel + self.hidden_channel, out_features=self.hidden_channel))
        self.main_list.append(nn.ReLU(inplace=True))
        for i in range(4):
            self.main_list.append(nn.Linear(in_features=self.hidden_channel, out_features=self.hidden_channel))
            self.main_list.append(nn.ReLU(inplace=True))
        self.main_block = nn.Sequential(*self.main_list)
        
    def appearance(self):
        self.appearance_list = []
        self.appearance_list.append(nn.Linear(self.appearance_channel + self.dir_channel + self.hidden_channel, self.hidden2_channel)) # 48 + 27 + 256
        self.appearance_list.append(nn.ReLU(inplace=True))
        self.appearance_block = nn.Sequential(*self.appearance_list)
        
    def transient(self):
        self.transient_list = []
        self.transient_list.append(nn.Linear(self.hidden_channel + self.transient_channel, self.hidden2_channel)) # 256 + 16
        self.transient_list.append(nn.ReLU(inplace=True))
        for i in range(3):
            self.transient_list.append(nn.Linear(self.hidden2_channel, self.hidden2_channel))
            self.transient_list.append(nn.ReLU(inplace=True))
        self.transient_block = nn.Sequential(*self.transient_list)
        
    def forward(self, x, sampling): # forward() : gradient의 학습을 결정 -> coarse와 fine을 한 개로 통일해야 한다.
        # input -> pts + viewing_dirs + appearance embedding vector + transient embedding vector
        # coarse -> [65536, 154] = [1024 x 64, 63 + 27 + 48 + 16] / fine -> [131072, 154] = [1024 x 128, 63 + 27 + 48 + 16]
        # TODO : 먼저 coarse model만 고려하기
        if sampling.lower() == 'coarse':
            sample_num = 64 # TODO : 변수로 치환
        elif sampling.lower() == 'fine':
            sample_num = 128 # TODO : 변수로 치환
        # input -> pts, dirs, appearance embedding vector, transient embedding vector
        pts = x[:,:self.pts_channel] # 63
        acc_channel = self.pts_channel
        dirs = x[:,acc_channel:acc_channel+self.dir_channel] # 27
        acc_channel += self.dir_channel
        appearance_embedding = x[:,acc_channel:acc_channel+self.appearance_channel] # 48
        acc_channel += self.appearance_channel
        transient_embedding = x[:,acc_channel:] # 16
        
        # residual learning
        feature = self.residual_block(pts) # [65536, 256]
        feature = torch.cat([pts, feature], dim=1) # [65536, 63 + 256]
        feature2 = self.main_block(feature) # [65536, 256]
        # appearance block의 input
        # debugging
        appearance_input = torch.cat([appearance_embedding, dirs, feature2], dim=1)
        # print(appearance_input.shape) # [65536, 331], 331 = 48 + 27 + 256
        feature3 = self.appearance_block(appearance_input)
        static_rgb_outputs = self.static_rgb_outputs(feature3)

        # static density
        static_density_outputs = self.static_density_outputs(feature2) # [65536, 3]
        
        # transient block의 input
        transient_input = torch.cat([transient_embedding, feature2], dim=1)
        # print(transient_input.shape) # [65536, 272], 272 = 256 + 16
        feature4 = self.transient_block(transient_input)

        # uncertainty
        uncertainty_outputs = self.uncertainty_outputs(feature4)
        # transient rgb
        transient_rgb_outputs = self.transient_rgb_outputs(feature4)
        # transient density
        transient_density_outputs = self.transient_density_outputs(feature4)
        
        # static + transient
        static_outputs = torch.cat([static_rgb_outputs, static_density_outputs], dim=1)
        transient_outputs = torch.cat([uncertainty_outputs, transient_rgb_outputs, transient_density_outputs], dim=1)
        # print(static_outputs.shape) # [65536, 4]
        # print(transient_outputs.shape) # [65536, 5]

        outputs = torch.cat([static_outputs, transient_outputs], dim=1) # [65536, 9], 9 = 3 + 1 + 3 + 1 + 1
        outputs = outputs.reshape([x.shape[0] // sample_num, sample_num, self.output_channel]) # [1024, 64, 9]
        
        return outputs