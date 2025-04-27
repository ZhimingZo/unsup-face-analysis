import argparse
import numpy as np
import torch
from dataset.mafl_dataset_tps import import_dataset_mafl 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn.linear_model
from network.FaceAwareNet import AwareNet
import pickle 
def network_eval(data_loader, network):
    torch.set_grad_enabled(False)
    network.eval()
    result_tensor={'gauss_xy':[], 'future_landmarks':[]}
    for i, (source_img, target, _, input_img, mask, label)  in enumerate(data_loader):
            out_img, out_xy, out_htmap, project,_,_,_,_ = network(input_img.cuda(), input_img.cuda())
            result_tensor['gauss_xy'].append(out_xy[:,:,:].detach().cpu().numpy())
            result_tensor['future_landmarks'].append(label)
    result_tensor = {k: np.concatenate(v) for k, v in result_tensor.items()}      
    return result_tensor


def evaluate(network, batch_size=50, bias=False, im_size=128): 
    
    #load dataset
    train_dataset = import_dataset_mafl(is_train=True)
    test_dataset = import_dataset_mafl(is_train=False)
    
    dataset_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dataset_test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    
    # eval
    train_tensors = network_eval(dataset_train_loader, network)   
    test_tensors = network_eval(dataset_test_loader, network)
    
    def convert_landmarks(tensors, im_size=128):
        landmarks = tensors['gauss_xy']
        landmarks_gt = tensors['future_landmarks']
        im_size = np.array(im_size)
        landmarks = ((landmarks + 1) / 2.0) * im_size
        n_samples = landmarks.shape[0]
        landmarks = landmarks.reshape((n_samples, -1))
        landmarks_gt = landmarks_gt.reshape((n_samples, -1))
        return landmarks, landmarks_gt

    X_train, y_train = convert_landmarks(train_tensors, im_size)
    X_test, y_test = convert_landmarks(test_tensors, im_size)
    # regression
    regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=bias)
    _ = regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    
    landmarks_gt = test_tensors['future_landmarks'].astype(np.float32)
    landmarks_regressed = y_pred.reshape(landmarks_gt.shape)    
    eyes = landmarks_gt[:, :2, :]
    occular_distances = np.sqrt(np.sum((eyes[:, 0, :] - eyes[:, 1, :])**2, axis=-1))
    distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
    mean_error = np.mean(distances / occular_distances[:, None])
    
    
    #model = [regr]
    #with open("./model/MAFL_reg", "wb") as f:
    #    pickle.dump(model, f)
    #    print("done")
    return mean_error



def main(args): 
    batch_size = args.batch_size 
    bias = args.bias
    kps = args.keypoints
    network = AwareNet(total_maps=kps).cuda()
    network.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    mean_error = evaluate(network, 50, False, args.im_size)
    print('')
    print('========================= RESULTS =========================')
    print('model trained in unsupervised way on %s dataset' % "mafl")
    print('regressor trained on %s training set' % args.train_dataset)
    print('error on %s datset : %.5f (%.3f percent)' % (
       args.test_dataset,
       mean_error, mean_error * 100.0))
    print('===========================================================')

if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser(description='Test model on face datasets.')
   parser.add_argument('--experiment-name', type=str, default='mafl_evl', help='Name of the experiment to evaluate.')
   parser.add_argument('--train_dataset', type=str, default='mafl', help='Training dataset for regressor (mafl|aflw).')
   parser.add_argument('--test_dataset', type=str, default='mafl', help='Testing dataset for regressed landmarks (mafl|aflw).')
   parser.add_argument('--model_path', type=str, default= './ckpts/MAFL/model_mafl_kps_10.pth')
   parser.add_argument('--paths-config', type=str, default='configs/paths/default.yaml', required=False, help='Path to the paths config.')
   parser.add_argument('--iteration', type=int, default=None, required=False, help='Checkpoint iteration to evaluate.')
   parser.add_argument('--buffer-name', type=str, default=None, required=False, help='Name of the buffer when using matlab data pipeline.')
   parser.add_argument('--im_size', type=int, default=128, required=False, help='Image size.')
   parser.add_argument('--bias', action='store_true', required=False, help='Use bias in the regressor.')
   parser.add_argument('--batch_size', type=int, default=50, required=False, help='batch_size')  
   parser.add_argument('--keypoints', type=int, default=10, required=False, help='num_kps') 
   args = parser.parse_args()
   main(args)


























