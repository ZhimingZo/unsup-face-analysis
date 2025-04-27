import os 
from scipy.io import loadmat
from torchvision.io import read_image 
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np
from dataset.tps_sampler import TPSRandomSampler
import torch
from utils.utils import _get_smooth_mask
import PIL
from torchvision.transforms.functional import hflip 

class CelebAImageDataset(Dataset): 

    def __init__(self, annotations, img_csv_file, img_dir, 
               image_size=[128, 128],
               vertical_points=10, horizontal_points=10,
               rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
               warpsd=[0.001, 0.005, 0.001, 0.01],
               name='TPSDataset', transform=None):
           if annotations is not None: 
               self.landmarks = annotations
           else: 
               self.landmarks = None
           self.img = pd.read_csv(img_csv_file, header=None)
           self.img_dir = img_dir
           self.transform = transform 
            
           self.img_size = image_size
           self.im_size = image_size[0]
           # apply tps 
            
           self._target_sampler = TPSRandomSampler(
                  image_size[1], image_size[0], rotsd=rotsd[0], scalesd=scalesd[0],
                  transsd=transsd[0], warpsd=warpsd[:2], pad=False)
           self._source_sampler = TPSRandomSampler(
                  image_size[1], image_size[0], rotsd=rotsd[1], scalesd=scalesd[1],
                  transsd=transsd[1], warpsd=warpsd[2:], pad=False)
            
    def __len__(self,):
           return len(self.img)


    def __getitem__(self, idx):
           img_path = os.path.join(self.img_dir, self.img.iloc[idx, 0])
           #img = read_image(img_path).float()/255.0
           img = Image.open(img_path)
           org_size = list(img.size) # w, h
           
           ########## crop img ############### 
           crop_percent = 0.8 
           resize_sz = np.round(self.im_size/crop_percent).astype(np.int32)
           margin = np.round((resize_sz - self.im_size) / 2.0).astype(np.int32)
           img = img.resize((resize_sz, resize_sz), PIL.Image.Resampling.BILINEAR)
           img = np.asarray(img) #160 160 3
           img = img[margin:margin+self.im_size, margin:margin+self.im_size]
           img = Image.fromarray(img)     
           img = self.transform(img)
           img = torch.unsqueeze(img, 0).numpy(); #1* 3 * 128 * 128
           
           mask = _get_smooth_mask(self.img_size[0], self.img_size[1], 10, 20)[:, :, None] #128 * 128 * 1 tensor
           mask2d = torch.unsqueeze(mask.permute(2,0,1), 0).numpy() #1 1 128 128
           
           img = np.concatenate((mask2d, img), axis=1)
           target_img = self._target_sampler.forward_py(img)
           source_img = self._source_sampler.forward_py(target_img)           
           img = torch.from_numpy(img)
           source_img = torch.from_numpy(source_img)
           target_img = torch.from_numpy(target_img)
           
           final_source_img = source_img[0, 1:]
           final_target_img = target_img[0, 1:]
           final_mask_2d = target_img[0, 0:1]
           
           flip_target_img = hflip(final_target_img)
           
           #landmarks 
           if self.landmarks is not None: 
               label_xy = self.landmarks[idx] # label is (x,y)
               ratio_x, ratio_y =  resize_sz / org_size[0] , resize_sz / org_size[1]
               label_xy [:, 0] = (label_xy[:, 0]) * ratio_x - margin 
               label_xy [:, 1] = (label_xy[:, 1]) * ratio_y - margin
               return final_source_img, final_target_img, flip_target_img, img[0, 1:], final_mask_2d, label_xy
           return final_source_img, final_target_img, flip_target_img , img[0, 1:], final_mask_2d 


def import_dataset_celeba(is_train):
    if is_train: 
        annotation = None
        img_csv_file = './Data/celeba/celeba_training_dataset.csv'
        img_dir = './Data/celeba/Img/img_align_celeba_hq'
    else: 
        annotations = np.load('./Data/celeba/mafl_keypoints.npz')
        annotation=annotations['test_kps']
        img_csv_file = './Data/celeba/mafl_testing_dataset.csv'
        img_dir = './Data/celeba/Img/img_align_celeba_hq'
        
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = CelebAImageDataset(annotations=annotation, img_csv_file=img_csv_file, img_dir=img_dir, transform=transform)
    return dataset


