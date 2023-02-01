import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../")
# import uci_gas.utils as utils
from uci_gas.model import Gas_Model
import numpy as np
import argparse
import json
import os
import pandas as pd
# import copy
# from pathlib import Path
import torch
import torch.utils.data as data
from data.transforms import SensorTransforms

from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    OpenMax,
    ViM,
)
from pytorch_ood.utils import OODMetrics, ToUnknown

class OodDataset(data.Dataset):
    def __init__(self, data_arr, ood_flag=False, transforms=None, time_start=140, time_end=200, time_length=20, method='cnn'):
        self.transforms = transforms
        self.data_lst, self.labels = data_arr
        self.min = time_start
        self.max = time_end - time_length
        self.length = time_length
        self.method = method
        self.ood = ood_flag
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.method == 'cnn' or self.method == 'crnn':
            if self.min >= self.max:
                start_idx = self.min
            else:
                start_idx = np.random.randint(self.min, self.max)
            data = self.data_lst[index][:, start_idx:start_idx+self.length, :]
        else:
            if self.min >= self.max:
                start_idx = self.min
            else:
                start_idx = np.random.randint(self.min, self.max)
            data = self.data_lst[index][start_idx:start_idx+self.length, :]

        if self.ood:
            label = -1
        else:
            label = self.labels[index]

        if self.transforms is not None:
            t_data = self.transforms.apply(data)
            return t_data, label

        return data, label
    
class OodDataManager(object):

    def __init__(self, config):        
        self.class_num = config['class_num']
        self.method = config['method']
        config_data = config['data']
        self.data_path = config_data['path']
        self.name = config_data['name']
        self.time_start = config_data['start']
        self.time_end = config_data['end']
        self.time_length = config_data['length']
        self.batch_size = config_data['loader']['batch_size']
        self.datasets = {}
        
        print(self.data_path)
        
        npz_train_x_file_name = os.path.join(self.data_path, 'train_sensor_x.npy')
        npz_train_y_file_name = os.path.join(self.data_path, 'train_sensor_y.npy')

        npz_test_x_file_name = os.path.join(self.data_path, 'test_sensor_x.npy')
        npz_test_y_file_name = os.path.join(self.data_path, 'test_sensor_y.npy')

        X_train = np.load(npz_train_x_file_name)
        Y_train = np.load(npz_train_y_file_name)

        X_test = np.load(npz_test_x_file_name)
        Y_test = np.load(npz_test_y_file_name)
        # Y_test = np.array(Y_test, dtype=int)

        data_len = (Y_test.shape[0] // self.batch_size) * self.batch_size
        if data_len < 1:
            self.batch_size = Y_test.shape[0]
            data_len = self.batch_size
        print('data_len:', Y_test.shape[0], data_len)
        X_test = X_test[:data_len]
        Y_test = Y_test[:data_len]
        
        X_test = torch.FloatTensor(X_test)
        Y_test = torch.LongTensor(Y_test)
        self.test_data = (X_test, Y_test)

        X_train = torch.FloatTensor(X_train)
        Y_train = torch.LongTensor(Y_train)
        self.train_data = (X_train, Y_train)

        tsf_args = config['transforms']['args']
        
        with open(os.path.join(self.data_path, 'sensor_info.json')) as f:
            json_object = json.load(f)
            self.mean = json_object['sensor_data_mean_{}_{}'.format(self.time_start, self.time_end)]
            self.std = json_object['sensor_data_std_{}_{}'.format(self.time_start, self.time_end)]

        # print('mean:', mean)
        # print('std:', std)
        train_transfs = SensorTransforms('test', tsf_args, self.mean, self.std)
        test_transfs = SensorTransforms('test', tsf_args, self.mean, self.std)

        dataset_in_train = OodDataset(self.train_data, ood_flag=False, transforms=train_transfs, 
                                    time_start = self.time_start, time_end = self.time_end, time_length = self.time_length, method = self.method)
        self.train_loader = data.DataLoader(dataset=dataset_in_train, batch_size = self.batch_size)
        
        for ood_start in range(0, 100):
            ood_end = ood_start + self.time_length
            dataset_in_test = OodDataset(self.test_data, ood_flag=False, transforms=test_transfs, 
                                        time_start = self.time_start, time_end = self.time_end, time_length = self.time_length, method = self.method)
            dataset_out_test = OodDataset(self.test_data, ood_flag=True, transforms=test_transfs, 
                                        time_start = ood_start, time_end = ood_end, time_length = self.time_length, method = self.method)

            d_name = '{}_{}'.format(ood_start, ood_end)
            self.datasets[d_name] = data.DataLoader(dataset=dataset_in_test + dataset_out_test, batch_size = self.batch_size, drop_last=True)

class OodModelManager(object):
    def __init__(self, config, resume_path):
        self.method = config['method']
        
        self.resume_path = resume_path
        self.model = Gas_Model(config)
        if config['parallel']:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        opt_name = config['optimizer']['type']
        opt_args = config['optimizer']['args']
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
        
        self.resume_checkpoint()
        
        if self.method == "brnn_att" or self.method == 'crnn':
            torch.backends.cudnn.enabled=False
        
    def resume_checkpoint(self):
        print("Loading checkpoint: {} ...".format(os.path.join(self.resume_path, 'checkpoints/model_best.pth')))
        checkpoint = torch.load(os.path.join(self.resume_path, 'checkpoints/model_best.pth'))
        # print(checkpoint)
        self.start_epoch = checkpoint['epoch']
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        self.optimizer.load_state_dict(checkpoint['optimizer'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    # parser.add_argument('--resume_path', default='/hdd1/ischang/2022/gas_ood/saved_cv/L_all_cnn_result/sensor_npz-cnn-arcface', type=str, help='path')
    # parser.add_argument('--resume_path', default='/hdd1/ischang/2022/gas_ood/saved_cv/L_all_cnn_result/sensor_npz-cnn-NllLoss', type=str, help='path')
    # parser.add_argument('--resume_path', default='/hdd1/ischang/2022/gas_ood/saved_cv/L_n_n_result/sensor_npz-L1_1-brnn_att-cosface', type=str, help='path')
    parser.add_argument('--resume_path', default='/hdd1/ischang/2022/gas_ood/saved_cv/L_n_n_result/sensor_npz-L1_1-brnn_att-NllLoss', type=str, help='path')
    # parser.add_argument('--resume_path', default='/hdd1/ischang/2022/gas_ood/saved_cv/L_n_n_result/sensor_npz-L1_1-sp_bi_dir-cosface', type=str, help='path')
    # parser.add_argument('--resume_path', default='/hdd1/ischang/2022/gas_ood/saved_cv/L_n_n_result/sensor_npz-L1_1-sp_bi_dir-NllLoss', type=str, help='path')
    
    args = parser.parse_args()
    config = json.load(open(os.path.join(args.resume_path, 'checkpoints/config.json')))

    device = "cuda:0"
    
    ood_data_manager = OodDataManager(config)
    ood_model_manager = OodModelManager(config, args.resume_path)
    
    ood_model_manager.model.eval()

    # print(ood_model_manager.model)
    # exit()
    
    detectors = {}
    cfg_loss = config["train"]["loss"]
    if cfg_loss == 'NllLoss' or cfg_loss == 'FocalLoss':
        w = ood_model_manager.model.linear.weight
        b = ood_model_manager.model.linear.bias
    else:
        w = ood_model_manager.model.adms_loss.fc.weight
        b = ood_model_manager.model.adms_loss.fc.bias
        
    detectors["ViM"] = ViM(ood_model_manager.model.features, d=32, w=w, b=b)
    detectors["Mahalanobis"] = Mahalanobis(ood_model_manager.model.features, norm_std=ood_data_manager.std, eps=0.002)
    detectors["KLMatching"] = KLMatching(ood_model_manager.model)
    detectors["MaxSoftmax"] = MaxSoftmax(ood_model_manager.model)
    detectors["EnergyBased"] = EnergyBased(ood_model_manager.model)
    detectors["MaxLogit"] = MaxLogit(ood_model_manager.model)
    detectors["ODIN"] = ODIN(ood_model_manager.model, norm_std=ood_data_manager.std, eps=0.002)
    detectors["OpenMax"] = OpenMax(ood_model_manager.model)

    for name, detector in detectors.items():
        print(f"--> Fitting {name}")
        detector.fit(ood_data_manager.train_loader, device=device)

    results = []
    with torch.no_grad():
        for detector_name, detector in detectors.items():
            print(detector_name)
            for dataset_name, loader in ood_data_manager.datasets.items():
                metrics = OODMetrics()
                
                for x, y in loader:
                    x, target = x.cuda(), y.cuda()
                    # x, target = Variable(x), Variable(y)
                    # metrics.update(detector(x.to(device)), y.to(device))
                    # print('====>', x.shape, y.shape)
                    metrics.update(detector(x), y)
                    # print(ood_model_manager.model)
                    # result = ood_model_manager.model(x)
                    # result = ood_model_manager.model.features(x)
                    # print('\n result:{}'.format(result.shape))
                    # result = ood_model_manager.model(x, embed=True)
                    # print('\n result:{}'.format(result.shape))
                    # exit()
                    
                r = {"Detector": detector_name, "Dataset": dataset_name}
                r.update(metrics.compute())
                results.append(r)
    
    # print(results)         
    df = pd.DataFrame(results)
    print(df)
    # mean_scores = df.groupby("Detector").mean() * 100
    # print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))

    df.to_csv(os.path.join(args.resume_path, 'result_ood.csv'), float_format="%.2f", index = False)
    
