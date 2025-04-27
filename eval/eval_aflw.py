import argparse
import numpy as np
import torch
from dataset.aflw_dataset_tps import import_dataset_aflw
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn.linear_model
from network.FaceAwareNet import AwareNet
import pickle

def network_eval(data_loader, network):
    torch.set_grad_enabled(False)
    network.eval()
    result_tensor={'gauss_xy':[], 'future_landmarks':[]}
    for i, (source_img, target, input_img, mask, label)  in enumerate(data_loader):
            out_img, out_xy, out_htmap, _, _, _, _, _  = network(input_img.cuda(), input_img.cuda())
            result_tensor['gauss_xy'].append(out_xy.detach().cpu().numpy())
            result_tensor['future_landmarks'].append(label)
 
    result_tensor = {k: np.concatenate(v) for k, v in result_tensor.items()}      
    return result_tensor


def evaluate_aflw(network, batch_size=50, bias=False, im_size=128): 
    
    #load dataset
    train_dataset = import_dataset_aflw(is_train=True)
    test_dataset = import_dataset_aflw(is_train=False)
    
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
    #with open("./model/AFLW_reg", "wb") as f:
    #    pickle.dump(model, f)
    #    print("done")
    return mean_error


def main(): 
    network = AwareNet(total_maps=10).cuda()
    checkpoint = "./ckpts/AFLW/model_aflw_kps_10.pth"
    network.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    mean_error = evaluate_aflw(network,bias=False)
    print('')
    print('========================= RESULTS =========================')
    print('model trained in unsupervised way on %s dataset' % "celeba")
    print('regressor trained on aflw')
    print('error on aflw_test datset : %.5f (%.3f percent)' % (
       mean_error, mean_error * 100.0))
    print('===========================================================')

if __name__=='__main__':
    main()




















