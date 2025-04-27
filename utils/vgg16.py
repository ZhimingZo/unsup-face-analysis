import torch
import torch.nn as nn
from torchvision import models
import torchvision
from torchvision import transforms
import torch.nn.functional as F
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(weights='IMAGENET1K_V1').features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(23, 30):
            self.to_relu_5_3.add_module(str(x), features[x])
            
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        h = self.to_relu_5_3(h)
        h_relu_5_3 = h
        out = (x, h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3, h_relu_5_3)
        return out
        
def vgg_loss_mask(inp1, inp2, vgg16, mask):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inp1 =  transform(inp1)
    inp2 =  transform(inp2)
    oup1 = vgg16(inp1)
    oup2 = vgg16(inp2)
    total_loss = 0
    ws = [100.0, 1.6, 2.3, 1.8, 2.8, 100.0]
    for i in range(len(oup2)):
        mask1 = F.interpolate(mask, size=(oup1[i].shape[2], oup1[i].shape[3]))
        inter_loss = torch.square(oup1[i]-oup2[i])
        inter = torch.clone(inter_loss).detach()
        wl = ws[i] + 0.01*(torch.mean(inter*mask1) - ws[i])
        total_loss += torch.mean((inter_loss/wl*mask1))
    return total_loss*1000
 
 
