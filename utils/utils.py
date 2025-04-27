import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torchvision.transforms as T 
from PIL import Image 

def _get_smooth_step(n, b):
    x = torch.linspace(-1.0, 1.0, n)
    y = 0.5 + 0.5 * torch.tanh(x / b)
    return y


def _get_smooth_mask(h, w, margin, step):
    b = 0.4
    step_up = _get_smooth_step(step, b)
    step_down = _get_smooth_step(step, -b)
    def create_strip(size):
      return torch.cat(
          [torch.zeros(margin, dtype=torch.float32),
           step_up,
           torch.ones(size - 2 * margin - 2 * step, dtype=torch.float32),
           step_down,
           torch.zeros(margin, dtype=torch.float32)], dim=0)
    mask_x = create_strip(w)
    mask_y = create_strip(h)
    mask2d = mask_y[:, None] * mask_x[None]
    return mask2d

 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def compute_equ_loss(img_xy, img_rotated_xy, theta, image_size=128):
    # compute the rotation matrix 
    # image_xy shape [2, 10, 2]
    # imgae_rotated_xy [2, 10, 2]
    # theta : [0, 180]
    M = cv2.getRotationMatrix2D((image_size/2, image_size/2), theta, 1) #shape [2, 3]
    rotation_matrix = torch.tensor(M, dtype=torch.float32).cuda()
    # convert xy to homogenous coor
    ones = torch.ones((img_xy.shape[0], img_xy.shape[1]), dtype=torch.float32).cuda()
    img_xy_hom = torch.cat((img_xy, torch.unsqueeze(ones,dim=-1)), dim=-1)
    corresponding_xy = img_xy_hom @ rotation_matrix.T
    # compute loss 
    loss = nn.MSELoss(reduction='mean')
    output = loss(img_rotated_xy, corresponding_xy)
    return output
    

def rotate_image(img, theta, image_size=128):
    # img -> tensor 
    transform = T.ToPILImage()
    img = transform(img)
    M = cv2.getRotationMatrix2D((image_size/2, image_size/2), theta, 1) #shape [2, 3]
    image = cv2.warpAffine(np.array(img), M, (image_size, image_size))
    return  Image.fromarray(image) 

    
# batchxn
def normalize_vector(v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch, v.shape[1])
    v = v/v_mag
    return v
    
# u, v batchxn
def cross_product( u, v):
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
#poses batchx6
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batchx3
    y_raw = poses[:,3:6]#batchx3

    x = normalize_vector(x_raw) #batchx3
    z = cross_product(x,y_raw) #batchx3
    z = normalize_vector(z)#batchx3
    y = cross_product(z,x)#batchx3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batchx3x3
    return matrix


#input batchx4x4 or batchx3x3
#output torch batchx3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch=rotation_matrices.shape[0]
    R=rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular= sy<1e-6
    singular=singular.float()
        
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)
    zs=R[:,1,0]*0
        
   
    out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda())
  
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler






 
	
