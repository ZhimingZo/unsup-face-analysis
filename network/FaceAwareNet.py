import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from utils.utils import *

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        if m.bias is not None: 
             nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        nn.init.constant_(m.bias.data, 0)

class Conv2DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size1, kernel_size2, stride1=1, stride2=1, padding1='same', padding2='same', activation="relu", normalize="batch"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size1, stride=stride1, padding=padding1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size2, stride=stride2, padding=padding2)
       
        if activation == "relu":
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU(0.2)

        if normalize == "batch":
            self.norm = nn.BatchNorm2d(out_channel, eps=1e-04, track_running_stats=False)
        elif normalize == "instance":
            self.norm = nn.InstanceNorm2d(out_channel)
        else:
            self.norm = None
    def forward(self, x):
    	y0 = self.conv1(x)
    	if self.norm is not None:
    	    y0 = self.norm(y0)
    	y1 = self.relu(y0)
    	
    	y2 = self.conv2(y1)
    	if self.norm is not None:
    	    y2 = self.norm(y2)
    	out = self.relu(y2)
    	return out

class Encoder(nn.Module): 

    def __init__(self, in_chan=3, filters_=32): 
        super().__init__()
        self.conv1 = Conv2DBlock(in_chan, filters_, kernel_size1=7, kernel_size2=3, stride1=1, stride2=1) # 32 * 128 * 128 
        self.conv2 = Conv2DBlock(filters_, filters_*2, kernel_size1=3, kernel_size2=3, stride1=2, stride2=1, padding1=1) #64 * 64 * 64
        self.conv3 = Conv2DBlock(filters_*2, filters_*4, kernel_size1=3, kernel_size2=3, stride1=2, stride2=1, padding1=1) # 128 * 32*32
        self.conv4 = Conv2DBlock(filters_*4, filters_*8, kernel_size1=3, kernel_size2=3, stride1=2, stride2=1, padding1=1) # 256 * 16 * 16

    def forward(self, x): 
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        out = self.conv4(y3)
        return out

class Decoder(nn.Module): 
    def __init__(self, filters=32, out_channels=3, total_maps=20):
        super().__init__()
        
        self.conv1 = Conv2DBlock(filters*8+total_maps, filters*8, kernel_size1=3, kernel_size2=3, stride1=1, stride2=1) #256* 16 * 16
        self.conv2 = Conv2DBlock(filters*8, filters*4, kernel_size1=3, kernel_size2=3, stride1=1, stride2=1) #128 * 32 * 32
        self.conv3 = Conv2DBlock(filters*4, filters*2, kernel_size1=3, kernel_size2=3, stride1=1, stride2=1) # 64 * 64 * 64
       
        self.out1 = nn.Conv2d(filters*2, filters, kernel_size=3, stride=1, padding='same') # 32 * 128 * 128 
        self.out2 = nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding='same') # 3 * 128 * 128
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(filters, eps=1e-04, track_running_stats=False)
    def forward(self, x):
        y0 = self.conv1(x) # N*256*16*16

        y0 = F.interpolate(y0, scale_factor=2, mode='bilinear', align_corners=False) # N*256*32*32
        y1 = self.conv2(y0) # N*128*32*32
        
        y1 = F.interpolate(y1, scale_factor=2, mode='bilinear', align_corners=False) # N*128*64*64
        y2 = self.conv3(y1) # N*64*64*64 

        y2 = F.interpolate(y2, scale_factor=2, mode='bilinear', align_corners=False) # N*64*128*128 
             
        y3 = self.out1(y2) #N*32*128*128
        y3 = self.norm(y3)
        y3 = self.relu(y3) #N*3*128*128
        out = self.out2(y3)
        return out

class PoseNet(nn.Module): 
     def __init__ (self, in_feature=256, inter_feature=128, out_feature=3):  # input N X 256 x 16 x 16
         super().__init__() 
         
         self.pool = nn.AvgPool2d(2, 2) # k = 3 and stride = 2
         self.rotation = nn.Sequential(
                        nn.Linear(in_feature*8*8, inter_feature), # 512 * 8 * 8 -> 128
                        nn.ReLU(),
                        nn.Dropout(),  
			nn.Linear(inter_feature, inter_feature), 
                        nn.ReLU(),
                        nn.Dropout(),
			nn.Linear(inter_feature, inter_feature),                                
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(inter_feature, out_feature*2)  #256 * 8 * 8 -> 6         
                    )
         self.translation = nn.Sequential(
                        nn.Linear(in_feature*8*8, out_feature)        #256 * 8 * 8 -> 3 
                    )
     def forward(self, x): 
         out = self.pool(x)
         out = torch.flatten(out, start_dim=1)
         rotation = self.rotation(out)
         translation = self.translation(out)
         return rotation, translation  


         
class AwareNet(nn.Module): # NCHW
     def __init__(self, in_channels=3, out_channels=3, filters=32, total_maps=10):
         super().__init__()
         self.encoder_t = Encoder(in_chan=in_channels, filters_=filters)
         self.encoder_s = Encoder(in_chan=in_channels, filters_=filters)
         self.heatmap = nn.Conv2d(filters*8, total_maps, kernel_size=1) # output heatmap feature 
        
         self.pose_predictor = PoseNet() # input feature from encoder, output: a, e, t, and potential translation 
         self.renderer = Decoder(filters=filters, out_channels=out_channels, total_maps=total_maps) #
         self.underlying3DPose =  nn.Parameter(torch.zeros(size=(out_channels, total_maps), dtype=torch.float)) # 3 X nmaps,  X, Y, Z 
         nn.init.xavier_uniform_(self.underlying3DPose.data, gain=1.414)
         
     def forward(self, source_img, target_img):
         img_feature = self.encoder_s(source_img)
         structure_feature = self.encoder_t(target_img)
         
         rotation, translation  = self.pose_predictor(structure_feature)
         rotation_mat = compute_rotation_matrix_from_ortho6d(rotation)
         
         landmarks_3d = torch.matmul(rotation_mat, self.underlying3DPose) # B X 3 X 3 _ 3X10 -> B X 3 X 10
         landmarks_3d = landmarks_3d + translation[..., None]
         landmarks_3d = landmarks_3d.permute(0, 2, 1) # B X 10 X 3

         z = torch.clone(landmarks_3d[:,:,2:])
         z = torch.clamp(z, min=1)
         projected_2d = torch.clamp(landmarks_3d[:, :, :2] / z, min=-1, max=1) 

         ht_map_features = self.heatmap(structure_feature) # NCHW
         ht_map_features = ht_map_features.permute(0, 2, 3, 1) #NHWC
         kps_xy, ht_maps = self.compute_heatmap(ht_map_features) # compute heatmap 
         concatenated_embedding = torch.cat((img_feature, ht_maps), dim=1)
         pred_img = self.renderer(concatenated_embedding) # reconstruction 

         euler = compute_euler_angles_from_rotation_matrices(rotation_mat)*180/np.pi
         return pred_img, kps_xy, ht_maps, projected_2d, rotation, translation, self.underlying3DPose, euler
         
     def compute_heatmap(self, x):
        # get "x-y" coordinates: 
        xshape = list(x.shape) # N, 16, 16, 20
        map_size = 16
        std = 10
       
        g_x, g_x_prob = self.get_coord(x, 1, xshape[2])
        g_y, g_y_prob = self.get_coord(x, 2, xshape[1])
        
        gauss_mu = torch.stack([g_x, g_y], dim=2)
        gauss_yx = self.convert_to_hp(gauss_mu, [map_size, map_size], std)
        return gauss_mu, gauss_yx 
      
     def get_coord(self, x, other_axis, axis_size):
        g_c_prob = torch.mean(x, dim=other_axis) #B, W, NMAP
        softmax = torch.nn.Softmax(dim=1)
        g_c_prob = softmax(g_c_prob) #B, W, NMAP
        coord_pt = torch.linspace(-1.0, 1.0, axis_size) # W
        coord_pt = torch.reshape(coord_pt.float(), (1,axis_size,1)).cuda()
        g_c = torch.sum(g_c_prob*coord_pt, dim=1)
        return g_c, g_c_prob
    
     def convert_to_hp(self, mu, shape_hw, inv_std ):
        mu_x, mu_y = mu[:, :, 0:1], mu[:, :, 1:2]
        x = torch.linspace(-1.0, 1.0, shape_hw[1]).float().to("cuda")
        y = torch.linspace(-1.0, 1.0, shape_hw[0]).float().to("cuda")
        
        mu_x, mu_y = torch.unsqueeze(mu_x, -1), torch.unsqueeze(mu_y, -1)
        x = torch.reshape(x, (1, 1, 1, shape_hw[1]))
        y = torch.reshape(y, (1, 1, shape_hw[0], 1)) 
  
        g_y = (y-mu_y)**2
        g_x = (x-mu_x)**2
        dist = (g_y + g_x) * inv_std**2 
        g_yx = torch.exp(-dist) # NCHW
        return g_yx
        
     def compute_rot_mat(self, euler_angles, convention=["Z", "Y", "X"]):  
        # rot is B X 3 
        """
            Convert rotations given as Euler angles in radians to rotation matrices.

            Args:
               euler_angles: Euler angles in radians as tensor of shape (..., 3).
               convention: Convention string of three uppercase letters from
               {"X", "Y", and "Z"}.

            Returns:
               Rotation matrices as tensor of shape (..., 3, 3).
        """
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = [
            self._axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]
        # return functools.reduce(torch.matmul, matrices)
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])        
        
     def _axis_angle_rotation(self, axis: str, angle: torch.Tensor) -> torch.Tensor:
         """
         Return the rotation matrices for one of the rotations about an axis
         of which Euler angles describe, for each value of the angle given.

         Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

         Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
         """

         cos = torch.cos(angle)
         sin = torch.sin(angle)
         one = torch.ones_like(angle)
         zero = torch.zeros_like(angle)

         if axis == "X":
             R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
         elif axis == "Y":
             R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
         elif axis == "Z":
             R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
         else:
             raise ValueError("letter must be either X, Y or Z.")

         return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
        

