# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import pickle   
import os
import random
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
from network.FaceAwareNet import AwareNet 
from utils.dataset import ImageFolderWithPaths
import code
random.seed(2)

def eval_angle(network):
	test_data_root = "./Data/data_biwi/biwi/BIWI"
	transform = transforms.Compose(
		[	
			transforms.Resize((128,128)),
			transforms.ToTensor(),
		]) 

	dataset = ImageFolderWithPaths(test_data_root, transform=transform)
	test_data_loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)

	pred_a = []; pred_e = []; pred_t = []
	gt_a = []; gt_e = []; gt_t = []

	fnames = []
	network.eval()
	for i, (sampled_batch, _, fname) in tqdm(enumerate(test_data_loader)):
		fname = fname[0]
		_, _, _, _, R, t, _3d, rotation = network(sampled_batch.cuda(), sampled_batch.cuda())  
		pred_a.append(rotation[:, 1].cpu().detach().numpy()) # y az
		pred_e.append(rotation[:, 0].cpu().detach().numpy()) # x el
		pred_t.append(rotation[:, 2].cpu().detach().numpy()) # z tl

		gt_a.append(float(fname.split('/')[-1].split('_')[6]))  # yaw
		gt_e.append(float(fname.split('/')[-1].split('_')[4]))  # pitch
		gt_t.append(float(fname.split('/')[-1].split('_')[8]))  # roll
 

	pred_a = np.asarray(pred_a)
	pred_e = np.asarray(pred_e)
	pred_t = np.asarray(pred_t)

	gt_a = np.asarray(gt_a)
	gt_e = np.asarray(gt_e)
	gt_t = np.asarray(gt_t)
                

	mae=0
	sample_inds = []
	for i in range(0, 100):
	    n = random.randint(0, 100)
	    while n in sample_inds:
	       n = random.randint(0, 15040)
	    sample_inds.append(n)
	       
	az_x = pred_a[sample_inds]
	az_y = gt_a[sample_inds]
	linreg1 = LinearRegression().fit(az_x.reshape(az_x.shape[0],1), az_y.reshape(az_x.shape[0],1))         
	linreg_pred_a = linreg1.predict(pred_a.reshape(gt_a.shape[0],-1))
	linreg_pred_a = np.delete(linreg_pred_a, sample_inds)
	gt_a = np.delete(gt_a, sample_inds)
	err_a = (abs(linreg_pred_a - gt_a)).mean()
	print('Azimuth error: ',err_a)
	mae+=err_a

	el_x = pred_e[sample_inds]
	el_y = gt_e[sample_inds]
	linreg2 = LinearRegression().fit(el_x.reshape(el_x.shape[0],1), el_y.reshape(el_x.shape[0],1))         
	linreg_pred_e = linreg2.predict(pred_e.reshape(gt_e.shape[0],-1))
	linreg_pred_e = np.delete(linreg_pred_e, sample_inds)
	gt_e = np.delete(gt_e, sample_inds)
	err_e = (abs(linreg_pred_e - gt_e)).mean()
	print('Elevation error: ',err_e)
	mae+=err_e

	ct_x = pred_t[sample_inds]
	ct_y = gt_t[sample_inds]
	linreg3 = LinearRegression().fit(ct_x.reshape(ct_x.shape[0],1), ct_y.reshape(ct_x.shape[0],1)) 
	linreg_pred_t = linreg3.predict(pred_t.reshape(gt_t.shape[0],-1))
	linreg_pred_t = np.delete(linreg_pred_t, sample_inds)
	gt_t = np.delete(gt_t, sample_inds)
	err_t = (abs(linreg_pred_t - gt_t)).mean()
	print('Tilt error: ',err_t)
	mae+=err_t

	print('MAE: ',mae/3)
	
	#model = [linreg1, linreg2, linreg3]
	#with open("./model/linreg_coeff_kps_10_new", "wb") as f:
	#	pickle.dump(model, f)
	#	print("done")
	return mae/3
def main(): 
    network = AwareNet(total_maps=10).cuda()
    ckpt = './ckpts/MAFL/model_mafl_kps_10.pth'
    checkpoint = torch.load(ckpt)
    network.load_state_dict(checkpoint['model_state_dict']) 
    eval_angle(network)

if __name__ == "__main__":
   main()


  
