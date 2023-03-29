from JH_data_loader import LLFF, Rays_DATALOADER
from JH_solver import Solver

import torch
from torch.backends import cudnn
import numpy as np
import argparse
import os
import sys

# Random seed 고정
torch.manual_seed(1234)
cudnn.deterministic = True # ?
cudnn.benchmark = True # For fast training 

# TODO : parser -> class 화, 나머지 함수 -> main 문

# Paser
parser = argparse.ArgumentParser(description='NeRF-W Implementation by JH')

# LLFF
parser.add_argument('--base_dir', type=str, default='./data/nerf_llff_data/fern')
parser.add_argument('--factor', type=int, default=8) # factor = 1 -> killed
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--ndc_space', type=bool, default=True)

# train
parser.add_argument('--resume_iters', type=int, default=None)
parser.add_argument('--mode', type=str, default='Train', choices=['Train', 'Test']) # list 형식으로 만들기
parser.add_argument('--nb_epochs', type=int, default=900) # 60 epoch = 20000 iterations
parser.add_argument('--save_val_iters', type=int, default=5) # 1 epoch마다 validation 수행
parser.add_argument('--save_model_iters', type=int, default=5)

parser.add_argument('--near', type=int, default=0)
parser.add_argument('--far', type=int, default=1)
parser.add_argument('--coarse_num', type=int, default=64)
parser.add_argument('--fine_num', type=int, default=64)
parser.add_argument('--L_pts', type=int, default=10)
parser.add_argument('--L_dirs', type=int, default=4)
parser.add_argument('--learning_rate', type=int, default=5e-4) # learning rate = 5e-4
parser.add_argument('--sample_num', type=int, default=64)

# NeRF-W -> appearance embedding vector의 단어 수, 차원, transient embedding vector의 단어 수, 차원
parser.add_argument('--appearance_embedded', type=bool, default=True)
parser.add_argument('--transient_embedded', type=bool, default=True)
parser.add_argument('--custom_dataset', type=bool, default=False)
parser.add_argument('--appearance_embedding_word', type=int, default=100) # appearance embedding vector의 dim = 2 -> 조도 밝 / 어
parser.add_argument('--appearance_embedding_dim', type=int, default=48)
parser.add_argument('--transient_embedding_word', type=int, default=100) # appearance embedding vector의 dim = 2 -> 조도 밝 / 어
parser.add_argument('--transient_embedding_dim', type=int, default=16)

# save path
parser.add_argument('--save_results_path', type=str, default='results/')
parser.add_argument('--save_train_path', type=str, default='results/val/')
parser.add_argument('--save_test_path', type=str, default='results/test/')
parser.add_argument('--save_model_path', type=str, default='models/')
parser.add_argument('--save_coarse_path', type=str, default='models/coarse/')
parser.add_argument('--save_fine_path', type=str, default='models/fine/')

# NeRF-W
parser.add_argument('--save_appearance_embedding_path', type=str, default='models/appearance_embedding/')
parser.add_argument('--save_transient_embedding_path', type=str, default='models/transient_embedding/')

config = parser.parse_args()

# make directories
if not os.path.exists(config.save_results_path):
    os.makedirs(config.save_results_path)
if not os.path.exists(config.save_train_path):
    os.makedirs(config.save_train_path)
if not os.path.exists(config.save_test_path):
    os.makedirs(config.save_test_path)
if not os.path.exists(config.save_model_path):
    os.makedirs(config.save_model_path)
if not os.path.exists(config.save_coarse_path):
    os.makedirs(config.save_coarse_path)
if not os.path.exists(config.save_fine_path):
    os.makedirs(config.save_fine_path)
if not os.path.exists(config.save_appearance_embedding_path):
    os.makedirs(config.save_appearance_embedding_path)
if not os.path.exists(config.save_transient_embedding_path):
    os.makedirs(config.save_transient_embedding_path)

# Dataset preprocessing
images, poses, bds, render_poses, i_val, focal = LLFF(config.base_dir, config.factor, config.appearance_embedded, config.transient_embedded, config.custom_dataset).outputs() # focal도 추출

# 하나의 함수로 묶을까?
# 아래의 것들은 JH_data_loader에 넣어버린다. -> 안 넣어도 될 듯.
height = images.shape[1] # 378 # 어떻게 자동화시키지?
width = images.shape[2] # 504

# print(focal) # 407.0
intrinsic = np.array([
            [focal, 0, 0.5*width], # 0.5*W = x축 방향의 주점
            [0, focal, 0.5*height], # 0.5*H = y축 방향의 주점
            [0, 0, 1]])
near = 1. # const처럼 만들기

# Train -> data_loader + val_data_loader만 키고, Test -> test_data_loader만 키자. -> for 시간 단축
# Train data loader -> shuffle = True / Validation data loader -> shuffle = False
if config.mode == 'Train':
    # tqdm
    data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, poses, i_val, images, config.ndc_space, test=False, train=True, shuffle=False, drop_last=False).data_loader() # Train
    # data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, poses, i_val, images, near, config.ndc_space, False, True, shuffle=True, drop_last=False) # Train -> tqdm
    # val_data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, poses, i_val, images, config.ndc_space, test=False, train=False, shuffle=False, drop_last=False).data_loader() # Validation
    val_data_loader = None
    Solver(data_loader, val_data_loader, None, config, i_val, height, width).train()
elif config.mode == 'Test':
    test_data_loader = Rays_DATALOADER(config.batch_size, height, width, intrinsic, render_poses, None, None, config.ndc_space, test=True, train=False, shuffle=False, drop_last=False).data_loader() # Test
    Solver(None, None, test_data_loader, config, None, height, width).test()