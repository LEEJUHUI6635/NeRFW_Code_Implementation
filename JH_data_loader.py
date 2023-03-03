import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2 as cv
import torch.nn as nn
import sys
from PIL import Image, ImageDraw
# Dataloader에서 ***bds_factor = 0.75***

def normalize(x):
    return x / np.linalg.norm(x)

def new_matrix(vec2, up, center):
    vec2 = normalize(vec2) # Z축
    vec1_avg = up # Y축
    vec0 = normalize(np.cross(vec1_avg, vec2)) # X축 = Y축 x Z축
    vec1 = normalize(np.cross(vec2, vec0)) # Y축 = Z축 x X축
    matrix = np.stack([vec0, vec1, vec2, center], axis=1)
    return matrix

def new_origin(poses): # input -> poses[20, 3, 5], output -> average pose[3, 5]
    hwf = poses[0,:,-1:]
    # center -> translation의 mean
    center = poses[:,:,3].mean(0) # 이미지 개수에 대한 mean
    # vec2 -> [R3, R6, R9] rotation의 Z축에 대해 sum + normalize
    vec2 = normalize(poses[:,:,2].sum(0)) # 이미지 개수에 대한 sum
    # up -> [R2, R5, R8] rotation의 Y축에 대한 sum
    up = poses[:,:,1].sum(0) # 이미지 개수에 대한 sum
    new_world = new_matrix(vec2, up, center)
    new_world = np.concatenate([new_world, hwf], axis=1)
    return new_world

# appearance와 obstacle -> image에 추가
class LLFF(object): # *** bd_factor를 추가해야 한다. ***
    def __init__(self, base_dir, factor, bd_factor=0.75, appearance_embedded = True, transient_embedded = True): # bd_factor = 0.75로 고정
        self.base_dir = base_dir
        self.factor = factor
        self.bd_factor = bd_factor # ***
        self.appearance_embedded = appearance_embedded
        self.transient_embedded = transient_embedded
        self.custom_dataset = True # TODO : user setting
        self.preprocessing() # Test
        self.load_images() # appearance embedding x, transient embedding x
        self.pre_poses() # Test
        self.spiral_path() # Test

    def preprocessing(self):
        # poses_bounds.npy 파일에서 pose와 bds를 얻는다.
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))
        self.image_num = poses_bounds.shape[0]
        poses = poses_bounds[:,:-2] # depth를 제외한 pose
        self.bds = poses_bounds[:,-2:] # depth
        self.poses = poses.reshape(-1, 3, 5) # [20, 3, 5]
        
    def load_images(self): # images -> [height, width, 3, image_num] + 255로 나누어 normalize
        # 모든 image를 불러온다.
        image_dir = os.path.join(self.base_dir, 'images')
        files = sorted([file for file in os.listdir(image_dir)]) # Q. colmap의 순서?
        images_list = []
        for idx, file in enumerate(files):
            images_RGB = cv.imread(os.path.join(image_dir, file), flags=cv.IMREAD_COLOR) # RGB로 읽기
            self.width = images_RGB.shape[1]
            self.height = images_RGB.shape[0]
            if self.appearance_embedded == False and self.transient_embedded == False: # 기존의 NeRF
                # appearance_embedded = False and transient_embedded = False
                images_resize = cv.resize(images_RGB, dsize=(self.width // self.factor, self.height // self.factor)) # width, height 순서 
                images_norm = images_resize / 255 # normalization
                images_list.append(images_norm)
            # appearance embedding + transient embedding
            # TODO : appearance embedding과 transient embedding 따로 처리
            elif self.appearance_embedded == True and self.transient_embedded == True: # NeRF in the wild
                if self.custom_dataset == False:
                    np.random.seed(idx) # image에 따른 random seed
                    # appearance embedding
                    # normalize
                    images_norm = images_RGB / 255 # normalization -> 0 ~ 1
                    scale = np.random.uniform(low=0.8, high=1.2, size=3)
                    bias = np.random.uniform(low=-0.2, high=0.2, size=3)
                    images_norm = np.clip(scale*images_norm+bias, 0, 1)
                    images_distorted = np.uint8(255*images_norm)

                    # transient embedding
                    # images_distorted numpy -> PIL
                    images_distorted = Image.fromarray(images_distorted)
                    draw = ImageDraw.Draw(images_distorted)
                    left = np.random.randint(self.width//4, self.width//2) # width의 1/4 ~ 1/2
                    top = np.random.randint(self.height//4, self.height//2) # height의 1/4 ~ 1/2
                    for i in range(10):
                        np.random.seed(10*idx+i)
                        random_color = tuple(np.random.choice(range(256), 3))
                        draw.rectangle(((left+self.width//40*i, top), (left+self.width//40*(i+1), top+self.width//4)), fill=random_color) # user parameter
                
                    # images_distorted PIL -> numpy
                    images_distorted = np.array(images_distorted)
                    images_resize = cv.resize(images_distorted, dsize=(self.width // self.factor, self.height // self.factor)) # width, height 순서
                    images_norm = images_resize / 255 # normalization
                    images_list.append(images_norm)
                elif self.custom_dataset == True:
                    # 먼저, transient embedding vector는 고려 x, appearance embedding vector만 고려 o
                    # 기존의 밝은 dataset -> 절반 정도의 dataset(번갈아가면서 -> TODO : idx가 홀수인 경우) -> 어둡게 만든다.
                    # TODO : 더 사실적으로 어둡게 만들기
                    images_norm = images_RGB / 255
                    if idx % 2 != 0:
                        images_norm = np.clip(images_norm - 0.45, 0, 1) # TODO : 0.45 -> user setting
                    # images_distorted = np.uint8(255*images_norm)
                    # cv.imwrite('results/train/train_data_{}.png'.format(idx), images_distorted)
                    images_list.append(images_norm)
        self.images = np.array(images_list)

    def pre_poses(self): # bds_factor에 대해 rescale을 처리해야 한다.
        sc = 1. if self.bd_factor is None else 1./(self.bds.min() * self.bd_factor) # sc = 1 / (가장 작은 depth boundary x bd_factor)
        # 좌표축 변환, [-u, r, -t] -> [r, u, -t]
        self.poses = np.concatenate([self.poses[:,:,1:2], -self.poses[:,:,0:1], self.poses[:,:,2:]], axis=-1)
        
        # bd_factor로 rescaling -> Q. 
        self.poses[:,:3,3] *= sc # translation에 해당하는 부분
        self.bds *= sc
        
        image_num = self.poses.shape[0] # image의 개수
        # 20개 pose들의 average pose를 구하고, 새로운 world coordinate를 생성한다. -> camera to world coordinate
        new_world = new_origin(self.poses) # 새로운 world coordinate, c2w
        # 새로운 world coordinate를 중심으로 새로운 pose를 계산한다. -> poses = np.linalg.inv(c2w) @ poses
        last = np.array([0, 0, 0, 1]).reshape(1, 4)
        # image_height, image_width, focal_length를 factor로 나눈다.
        hwf = new_world[:,-1].reshape(1, 3, 1) // self.factor
        self.focal = np.squeeze(hwf[:,-1,:])
        
        new_world = np.concatenate([new_world[:,:4], last], axis=0)
        last = last.reshape(1, 1, 4)
        lasts = np.repeat(last, image_num, axis=0)
        
        self.new_poses = np.concatenate([self.poses[:,:,:4], lasts], axis=1)
        self.new_poses = np.linalg.inv(new_world) @ self.new_poses
        self.new_poses = self.new_poses[:,:3,:4]
        hwfs = np.repeat(hwf, image_num, axis=0)
        self.poses = np.concatenate([self.new_poses, hwfs], axis=-1)
        # i_test -> recentered pose의 average pose의 translation과 가장 적게 차이가 나는 pose를 validation set으로 이용
        avg_pose = new_origin(self.poses)
        trans = np.sum(np.square(avg_pose[:3,3] - self.poses[:,:3,3]), -1) # avg_pose - poses = [20, 3, 3]
        self.i_val = np.argmin(trans)
    # 나선형으로 new rendering path 만들기 -> *****rendering path 이해하기*****
    
    def spiral_path(self):
        # new global origin에 대하여 recentered된 pose들의 새로운 origin을 다시 구한다. -> 이 말은 곧 recentered pose의 average pose
        # new global origin과 위처럼 새롭게 구한 origin은 다른 값이 나온다.
        z_rate = 0.5
        
        poses_avg = new_origin(self.poses) # recentered pose의 average pose
        hwf = poses_avg[:,-1:]
        # recenter된 pose의 up vector를 구한다. -> Y축
        up = self.poses[:,:3,1].sum(0)
        
        # focus depth를 구한다.
        # close_depth -> bds.min x 0.9, inf_depth -> bds.max x 5.
        close_depth, inf_depth = self.bds.min()*0.9, self.bds.max()*5.
        dt = 0.75
        # mean_focal
        mean_focal = 1/((1-dt)/close_depth + dt/inf_depth)
        
        # spiral path의 radius를 구한다.
        # recentered poses의 translation
        trans = self.poses[:,:3,3]
        # radius = translation의 절대값 -> 90 퍼센트에 해당하는 크기
        radius = np.percentile(a=trans, q=0.9, axis=0)
        
        # 새로 만들 view 개수 -> 120
        view_num = 120
        # rotation 횟수 -> 2
        rotation_num = 2
        # radius -> [1, 3] 마지막에 1을 붙여서 homogeneous coordinate으로 만들기
        last = np.array([1]).reshape(1, 1)
        radius = radius.reshape(1, 3)
        radius = np.concatenate([radius, last], axis=1)
        radius = radius.reshape(4, 1)
        # 두 바퀴를 회전하는 spiral path를 생성 -> 2 바퀴를 돌고 마지막 index를 제외 -> 120개의 new rendered path
        render_poses_list = []
        for theta in np.linspace(start=0., stop=rotation_num*2*np.pi, num=view_num+1)[:-1]:
            # Look vector -> recentered poses의 average pose의 R|t @ [radius * cos(theta), -radius * sin(theta), -radius * sin(z_rate * theta)]        
            look = poses_avg[:3,:4] @ (np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1]).reshape(4, 1) * radius)
            look = look.reshape([3,])
            # [3, 4] x [cos, -sin, -sin, 1] -> [4, 1]
            # z = Look vector(target 위치) - eye vector(현재 camera 위치)
            eye = poses_avg[:3,:4] @ np.array([0, 0, -mean_focal, 1]).reshape(4, 1)
            eye = eye.reshape([3,])
            z = normalize(look - eye)
            z = z.reshape([3,])
            # 새로운 pose에서의 camera to world matrix -> new matrix 함수 이용
            render_poses = new_matrix(vec2=z, up=up, center=look)
            render_poses = np.concatenate([render_poses, hwf], axis=-1)
            render_poses_list.append(render_poses)
        self.render_poses = np.array(render_poses_list)
        
    def outputs(self):
        images = self.images.astype(np.float32) # self.appearance_embedded = True and self.transient_embedded = False -> distorted image
        poses = self.poses.astype(np.float32) # Test
        bds = self.bds # Test 
        render_poses = self.render_poses.astype(np.float32) # Test
        i_val = self.i_val # Test
        focal = self.focal # focal length
        return images, poses, bds, render_poses, i_val, focal

# NeRF-W : rays_t -> 해당 ray가 어떠한 이미지에 속하는지에 대한 정보도 추가해야 한다.
# poses <- poses : Train or Validation / poses <- render_poses : Test
# TODO : NDC space optional
# TODO : dataloader -> GPU utils를 높이도록 설계
class Rays_DATASET(Dataset): # parameter -> kwargs
    def __init__(self, 
                 height, 
                 width, 
                 intrinsic, 
                 poses, 
                 i_val, 
                 images,
                 near=1.0, 
                 ndc_space=True, 
                 test=False, 
                 train=True): # pose -> [20, 3, 5] / Test
        super(Rays_DATASET, self).__init__()
        self.height = height # 378
        self.width = width # 504
        self.intrinsic = intrinsic # intrinsic parameter
        self.pose = poses[:,:,:4] # [?, 3, 4]
        self.i_val = i_val # validation index
        self.images = images
        self.near = near # Q.
        self.ndc_space = ndc_space # optional
        self.test = test
        self.train = train
        # appearance embedding vector + trasient embedding vector
        # self.appearance_embedding_vector = nn.Embedding(appearance_enbedding_word, appearance_embedding_dim) # [1500, 48]
        # self.transient_embedding_vector = nn.Embedding(transient_embedding_word, transient_embedding_dim) # [1500, 16]
        
        # image 별로 index 설정 -> self.appearance_embedding_vector(rays_t의 값 = image frame id), self.transient_embedding_vector(rays_t의 값 = image frame id)
        # rays_t = image frame id x torch.ones(image height x image width)
        
        self.focal = self.intrinsic[0][0] # focal length

        self.image_num = self.pose.shape[0] # Train과 Test 모두에 사용된다. -> 20
        
        if self.test == False: # Train or Validation
            self.train_idx = []
            self.val_idx = []
            for i in range(self.image_num): # 20개만큼 반복
                if i % self.i_val == 0:
                    self.val_idx.append(i) # [0, 12]
                    # print(val_idx) 
                else: # image frame id
                    self.train_idx.append(i) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
            # print('train_idx', self.train_idx)
            # print('val_idx', self.val_idx) # [0, 30]
            
            if self.train == True: # Train
                self.pose = self.pose[self.train_idx,:,:]
                # self.images = self.images[train_idx,:,:,:]
                self.images = self.images[self.train_idx,:,:,:]
                self.image_num = self.pose.shape[0] # 18
                
            elif self.train == False: # Validation
                self.pose = self.pose[self.val_idx,:,:]
                self.images = self.images[self.val_idx,:,:,:]
                self.image_num = self.pose.shape[0] # 2
        
        # TODO : 보강
        if self.ndc_space == False:
            self.all_rays() # view_dirs + rays_o + rays_d
                
        elif self.ndc_space == True: # default -> LLFF dataloader
            self.all_rays() # view_dirs
            self.ndc_all_rays() # rays_o + rays_d
        
        # TODO : embedding vector를 추가할 것인가에 대한 flag 생성
        if self.test == False:
            self.embedding_vector()
        
        # train과 test의 차이는 train -> rays_o + rays_d + rays_rgb VS test -> rays_o + rays_d
        
    # 하나의 image에 대한 camera to world -> rays_o는 image마다 다르기 때문
    def get_rays(self, pose): # rays_o, rays_d
        # height, width에 대하여 pixel마다 sampling -> pixel 좌표 sampling
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32), indexing='xy')
        
        # pixel 좌표계 -> metric 좌표계
        # (i - cx) / fx, (j - cy) / fy
        # cx = K[0][2], cy = K[1][2], fx = K[0][0], fy = K[1][1]
        # metric 좌표계의 y축은 pixel 좌표계의 v축과 반대
        # z축에 해당하는 값 -1 -> ray는 3D point에서 2D point로의 방향이기 때문
        
        pix2metric = np.stack([(u-self.intrinsic[0][2])/self.intrinsic[0][0], -(v-self.intrinsic[1][2])/self.intrinsic[1][1], -np.ones_like(u)], axis=-1)
        # print(pix2metric.shape) # [378, 504, 3]
        
        # camera 좌표계 -> world 좌표계
        rays_d = np.sum(pose[:3,:3] @ pix2metric.reshape(self.height, self.width, 3, 1), axis=-1)

        # 하나의 image에 대한 모든 픽셀의 ray에 대한 원점은 같아야 한다. ray는 3D point에서 2D point로 향하는 방향이다.
        rays_o = np.broadcast_to(pose[:3,3], rays_d.shape)
        return rays_o, rays_d # world 좌표계로 표현
    
    # NDC -> projection matrix
    # *****NDC 수식 이해하기***** -> Q. near = 1.?
    def ndc_rays(self, near, focal, rays_o, rays_d): # optional
        # rays_o, rays_d = self.get_rays(self.pose[0,:3,:4])
        # rays_o, rays_d -> [378, 504, 3]
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        # print(t.shape) # [378, 504]
        rays_o = rays_o + t[...,np.newaxis] * rays_d
        # print(rays_o.shape) # [378, 504, 3]
        o1 = -1.*focal/(self.width/2) * rays_o[...,0] / rays_o[...,2]
        o2 = -1.*focal/(self.height/2) * rays_o[...,1] / rays_o[...,2]
        o3 = 1. + 2. * near / rays_o[...,2]
        
        d1 = -1.*focal/(self.width/2) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        d2 = -1.*focal/(self.height/2) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        d3 = -2.* near / rays_o[...,2]
        
        rays_o = np.stack([o1, o2, o3], axis=0)
        rays_d = np.stack([d1, d2, d3], axis=0)
        rays = np.stack([rays_o, rays_d], axis=0) # [2, 378, 504, 3]

        return rays # NDC space로 표현된 rays
    
    # TODO : NDC 처리를 하지 않은 경우 -> rays_o + rays_d + rays_rgb + viewing direction
    def all_rays(self): # 모든 image에 대한 rays -> rays_o + rays_d + rgb
        # rays + rgb -> [2, 378, 504, 3(x, y, -1)] + [1, 378, 504, 3(r, g, b)]
        # get_rays -> rays_o + rays_d
        rays = np.stack([np.stack(self.get_rays(poses), axis=0) for poses in self.pose[:,:3,:4]], axis=0)
        # print(rays.shape) # [18, 2, 378, 504, 3]
        if self.test == False:
            self.images = self.images[:,np.newaxis,...]
            rays = np.concatenate([rays, self.images], axis=1)
        
        rays = np.moveaxis(rays, source=1, destination=3)
        # print(rays.shape) # [18, 378, 504, 3, 3] -> test : [120, 378, 504, 2, 3]
        if self.test == False:
            rays = rays.reshape([-1, 3, 3])
        else:
            rays = rays.reshape([-1, 2, 3])
        self.rays = rays.astype(np.float32)
        self.rays_rgb_list_no_ndc = np.split(self.rays, self.rays.shape[0], axis=0)
        if self.test == False:
            self.rays_rgb_list_no_ndc = [self.rays_rgb_list_no_ndc[i].reshape(3, 3) for i in range(len(self.rays_rgb_list_no_ndc))]
        else:
            self.rays_rgb_list_no_ndc = [self.rays_rgb_list_no_ndc[i].reshape(2, 3) for i in range(len(self.rays_rgb_list_no_ndc))]
        
        self.view_dirs_list = [self.rays_rgb_list_no_ndc[i][1:2,:] for i in range(len(self.rays_rgb_list_no_ndc))]
        
    def ndc_all_rays(self): # ndc 처리
        rays_list = []
        for i in range(self.image_num):
            rays_o, rays_d = self.get_rays(self.pose[i,:3,:4])
            rays = self.ndc_rays(self.near, self.focal, rays_o, rays_d)
            rays = np.moveaxis(rays, source=1, destination=-1)
            rays_list.append(rays)
        rays_arr = np.array(rays_list)

        if self.test == False:
            rays_arr = np.concatenate([rays_arr, self.images], axis=1)
        rays_arr = np.moveaxis(rays_arr, source=1, destination=3)

        if self.test == False:
            rays_arr = rays_arr.reshape(-1, 3, 3) # [3429216, 3, 3], 3429216 = 18 x 504 x 378
        else: # self.test = True
            rays_arr = rays_arr.reshape(-1, 2, 3)
        self.rays_rgb_list_ndc = [rays_arr[i,:,:] for i in range(rays_arr.shape[0])]
    
    # TODO : list -> 시간 소요
    def embedding_vector(self): # Q. 이미지에 따른 appearance embeddding vector + transient embedding vector를 만들기 위한 rays_t index
        if self.test == False and self.train == True: # train
            rays_t_list = [i * torch.ones(self.height * self.width, dtype=torch.long) for i in self.train_idx]
            # appearance embedding vector -> 0 or 1, 0 -> 밝은 조도 상황, 1 -> 어두운 조도 상황
            # TODO : self.custom == True
            # rays_appearance_t_list = [0 * torch.ones(self.height * self.width, dtype=torch.long) if i % 2 == 0 else 1 * torch.ones(self.height * self.width, dtype=torch.long) for i in self.train_idx]
        elif self.test == False and self.train == False: # validation
            rays_t_list = [i * torch.ones(self.height * self.width, dtype=torch.long) for i in self.val_idx]
        rays_t_arr = torch.cat(rays_t_list, dim=0)
        self.rays_t_list = [rays_t_arr[i] for i in range(len(rays_t_arr))]
        # if self.test == False and self.train == True: # train
        #     rays_appearance_t_arr = torch.cat(rays_appearance_t_list, dim=0)
        #     self.rays_appearance_t_list = [rays_appearance_t_arr[i] for i in range(len(rays_appearance_t_arr))]
        
    def __len__(self): # should be iterable
        return len(self.rays_rgb_list_no_ndc)
    
    # TODO : GPU utils를 높이기 위해서는 getitem에 많은 것을 구현 x
    def __getitem__(self, index): # should be iterable
        if self.ndc_space == True:
            samples = self.rays_rgb_list_ndc[index]
        elif self.ndc_space == False:
            samples = self.rays_rgb_list_no_ndc[index]
            
        view_dirs = self.view_dirs_list[index] # Debugging -> test시에는 없다.
        if self.test == False: # train or validation
            rays_t = self.rays_t_list[index]
            # if self.train == True:
            #     rays_appearance_t = self.rays_appearance_t_list[index]
            # # rays_t 추가
            # if self.train == False: # validation
            #     results = [samples, view_dirs, rays_t]
            # elif self.train == True: # train
            #     results = [samples, view_dirs, rays_t, rays_appearance_t]
            results = [samples, view_dirs, rays_t]
            # return samples, view_dirs # rays_o + rays_d + rgb
        elif self.test == True:
            results = [samples, view_dirs]
        return results

class Rays_DATALOADER(object):
    def __init__(self, 
                 batch_size, 
                 height, 
                 width, 
                 intrinsic, 
                 poses, 
                 i_val, 
                 images,
                 near, 
                 ndc_space, 
                 test, 
                 train, 
                 shuffle,
                 drop_last): # TODO : parameter -> kwargs
        self.height = height
        self.width = width
        self.intrinsic = intrinsic
        self.poses = poses
        self.i_val = i_val
        self.images = images
        self.near = near # 1.0 -> default
        self.ndc_space = ndc_space
        self.test = test
        self.train = train
        self.batch_size = batch_size
        self.results = Rays_DATASET(self.height, 
                                    self.width, 
                                    self.intrinsic, 
                                    self.poses, 
                                    self.i_val, 
                                    self.images,
                                    self.near, 
                                    self.ndc_space,
                                    self.test,
                                    self.train)
        self.shuffle = shuffle # 나중에 train이면 shuffle = True / validation 혹은 test이면 shuffle = False 되게끔 만들기 !
        self.drop_last = drop_last # 나중에 train이면 drop_last = True / validation 혹은 test이면 drop_last = False 되게끔 만들기 !
        
    def data_loader(self): # shuffle = False
        dataloader = DataLoader(dataset=self.results, batch_size=self.batch_size, shuffle = self.shuffle, num_workers=4, pin_memory=True, drop_last=False) # drop_last = False -> 마지막 batch 또한 학습한다.
        return dataloader