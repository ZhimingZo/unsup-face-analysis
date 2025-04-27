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

class AFLWImageDataset(Dataset): 

    def __init__(self, annotations_file, img_csv_file, img_dir, 
               image_size=[128, 128], 
               vertical_points=10, horizontal_points=10,
               rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
               warpsd=[0.001, 0.005, 0.001, 0.01],
               name='TPSDataset', transform=None):
 
           self.landmarks = loadmat(annotations_file)
           
           self.img = pd.read_csv(img_csv_file, header=None)
           self.img_dir = img_dir
           self.transform = transform 

           self.img_size = image_size
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
           img = read_image(img_path).float()/255.0
           img = torch.unsqueeze(img, 0); #1* 3 * 128 * 128
           img = self.transform(img).numpy()
           mask = _get_smooth_mask(self.img_size[0], self.img_size[1], 10, 20)[:, :, None] #128 * 128 * 1 tensor
           mask2d = torch.unsqueeze(mask.permute(2,0,1), 0).numpy()
           img = np.concatenate((mask2d, img), axis=1)
           target_img = self._target_sampler.forward_py(img)
           source_img = self._source_sampler.forward_py(target_img)           
            
           img = torch.from_numpy(img)
           source_img = torch.from_numpy(source_img)
           target_img = torch.from_numpy(target_img)
           
           final_source_img = source_img[0, 1:4]
           final_target_img = target_img[0, 1:4]
           final_mask_2d = target_img[0, 0:1] #[1, 128, 128]
 
           
           label_xy = self.landmarks['gt'][:,:,[1,0]][idx]
           org_size = self.landmarks['hw'][idx]
           ratio_y, ratio_x = org_size / self.img_size[0] 
           label_xy [:, 0] /= ratio_x 
           label_xy [:, 1] /= ratio_y  
           
           return final_source_img, final_target_img, img[0, 1:], final_mask_2d, label_xy


def import_dataset_aflw(is_train=True):
    if is_train: 
        annotation_file_path='./Data/aflw_dataset/aflw_train_keypoints.mat'
        img_csv_file='./Data/aflw_dataset/aflw_train_images.csv'
        img_dir='./Data/aflw_dataset/output'
    else: 
        annotation_file_path='./Data/aflw_dataset/aflw_test_keypoints.mat'
        img_csv_file='./Data/aflw_dataset/aflw_test_images.csv'
        img_dir='./Data/aflw_dataset/output'    

    transform = transforms.Compose([
            transforms.Resize((128, 128)),
    ])
    dataset = AFLWImageDataset(annotations_file=annotation_file_path, img_csv_file=img_csv_file, img_dir=img_dir, transform = transform)
    return dataset



