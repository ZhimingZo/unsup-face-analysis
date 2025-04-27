import argparse
import numpy as np
import os
import time
import datetime
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms 
from dataset.celeba_dataset_tps import  import_dataset_celeba
from dataset.mafl_dataset_tps import  import_dataset_mafl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import  ExponentialLR
from progress.bar import Bar
from utils.utils import AverageMeter
from matplotlib import pyplot as plt
from utils.vgg16 import vgg_loss_mask, Vgg16
from network.FaceAwareNet import AwareNet, initialize_weights
from eval.eval_mafl import evaluate as evaluate_mafl
from eval.angles_eva_auto import eval_angle
import warnings
warnings.filterwarnings('ignore')
 
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True                                                   

def train(args, data_loader, network, optimizer, scheduler, ckpt=None):
    
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        network.load_state_dict(checkpoint['model_state_dict']) 
        print("load checkpoint")
        for name, param in network.named_parameters():
            if param.requires_grad == True:
                 print(name)  

    loss_l2 = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    vgg_net = Vgg16().cuda()
    save_path = os.path.join("checkpoint/end_to_end_MAFL/", args.model_name, str(args.n_maps))
    if not os.path.exists(save_path):
       os.makedirs(save_path)
    F = open(os.path.join(save_path, "loss.txt"), 'w+')
    best_model_error_mafl=None
    #best_model_error_aflw=None
    for e in range(args.epoch): 
        # Switch to train mode
        torch.set_grad_enabled(True)
        network.train()
        print("epoch: " + str(e))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_loss_AM = AverageMeter()
        kps_loss_AM = AverageMeter()
        end = time.time()
        bar = Bar('Train', max=len(data_loader)) 
        for i, (source_img, target_img, flip_target_img, input_img, mask)  in enumerate(data_loader):
            data_time.update(time.time() - end)
            num_imgs = input_img.size(0)
            source_img = source_img.cuda()
            target_img = target_img.cuda()
	    # Train network
            out_img, out_xy, out_htmap, projected_2d, _, _, _, euler = network(source_img, target_img)
            optimizer.zero_grad() 
            out_xy = out_xy.detach()
            out_xy = (out_xy[:,:,:]+1)/2.0 * 128 
            projected_2d = (projected_2d[:,:,:]+1)/2.0 * 128
            recon_loss = vgg_loss_mask(out_img, target_img, vgg_net, mask.cuda())
            kps_loss = (loss_l2(out_xy[:, :, :], projected_2d) + loss_l1(out_xy[:, :, :], projected_2d))    
            loss = 20*kps_loss  + recon_loss
            kps_loss_AM.update(kps_loss.item(), num_imgs)
            total_loss_AM.update(loss.item(), num_imgs)
            
            loss.backward()
            optimizer.step()
            if args.max_norm:
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)   
            batch_time.update(time.time() - end)
            
            end = time.time()
            bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f} ' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=total_loss_AM.avg)
            bar.next()
        bar.finish()
	# save model 
        scheduler.step()
        if (e + 1) % args.save_interval == 0:
             torch.save({
                'epoch': e,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss_AM.avg,
            }, os.path.join(save_path, 'model_' + str(e) +'.pth'))
         
        # evaluation 
        model_error_mafl = evaluate_mafl(network)
        angle_error = eval_angle(network)
        print("model_error_on_mafl: " + str(model_error_mafl)) 
        if best_model_error_mafl is None or model_error_mafl < best_model_error_mafl:
            best_model_error_mafl = model_error_mafl
            # save best pth
            torch.save({
                'epoch': e,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss_AM.avg,
            }, os.path.join(save_path, 'best_model_mafl'+ '.pth'))
            
        F.write('Epoch: ' +str(e+1) + '\t' + 'Loss: ' +str(total_loss_AM.avg) + '\t ' + "kps_loss: "+ str(kps_loss_AM.avg)  + "\t " + "angle_mae: " + str(angle_error) + "\t" + "mafl_error: " + str(model_error_mafl))
        F.write('\n')
        F.flush()
    F.close()
    
def visualize(data_loader, network, path):
    checkpoint = torch.load(path)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    transform = transforms.ToPILImage()        
    for i, (source_img, target_img, flip_target_img, input_img, mask)  in enumerate(data_loader):
            source_img = rot_img1.cuda()
            target_img = rot_img2.cuda()
            inp_img = input_img.cuda()
	    # Train network
            out_img, out_xy, out_htmap, project2d, R, t, _3d, euler = network(inp_img, inp_img)

            out_xy = (out_xy[:,:,:]+1)/2.0 * 128
            out_xy = out_xy.detach().cpu().numpy()
            
            s_img = source_img.detach().cpu()[0]
            t_img = target_img.detach().cpu()[0]
            oup_img = out_img.detach().cpu()[0] 

            s_img = transform(s_img)
            t_img = transform(t_img)
            oup_img = transform(oup_img)
            
            oup_img.save("./test/oup_img_"+ str(i) + ".png")
            input_img1 = inp_img.detach().cpu() 

            input_img1 = transform(input_img1[0])
            input_img1.save("./test/inp_img_" + str(i) + ".png")
            x = out_xy[0,:,0]
            y = out_xy[0,:,1]
            color=['r','g','b', 'c', 'm', 'y', 'k', 'w', 'r', 'g']
            plt.axis('off')
            plt.scatter(x, y, c=color)
            plt.imshow(input_img1)
            plt.savefig("./test/testing_from_testing_data_" + str(i) +"_.png", bbox_inches='tight', pad_inches=0, transparent=True, dpi=34.9)
            plt.clf()
            
            print(i)
            if i == 50:
                exit()

def main(): 
    # create arg parser
    parser = argparse.ArgumentParser(description='IMMNet')
    parser.add_argument('--model_name', default='awareNet_end_to_end_celeba',type=str, help='model_name')
    
    parser.add_argument('--save_interval', default=1, type=int, help='interval (epoch) to save models')        
    parser.add_argument('--gpus', type=str, default='0', help='GPU numbers used for training')
    parser.add_argument('--num_workers', type=int, default=16, help='num workers for data loader')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    
    parser.add_argument('--epoch', default=20, type=float, help='training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    
    parser.add_argument('--n_maps', default=10, type=int, help='number of keypoints') 
    
    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    network = AwareNet(total_maps=args.n_maps).cuda()
    network.apply(initialize_weights)
    # Set optimizers
    optimizer = optim.Adam(network.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5) 
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = import_dataset_celeba(True) # is_Train 
    ckpt = None
    dataloader_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train(args, dataloader_train, network, optimizer, scheduler, ckpt=ckpt)
if __name__ == '__main__': 
    main()


