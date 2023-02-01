# import sys
# sys.path.append('..')

import os
from re import S
import pandas as pd
import numpy as np
import collections

import torch
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split

from data.data_sets import NpzDataset

class Gas_Sensor_DataManager(object):

    def __init__(self, config, epoch):
        self.class_num = config['class_num']
        self.method = config['method']

        config_data = config['data']
        self.data_path = config_data['path']
        self.name = config_data['name']
        self.loader_params = config_data['loader']
        self.time_start = config_data['start']
        self.time_end = config_data['end']
        self.time_length = config_data['length']
        self.imbalance = config_data['imbalance']
        self.train_test_split = config_data['split']
        
        self.epoch = epoch
        self.data_splits = {}
        
        if self.train_test_split:
            npz_train_x_file_name = os.path.join(self.data_path, 'train_sensor_x.npy')
            npz_train_y_file_name = os.path.join(self.data_path, 'train_sensor_y.npy')
            npz_test_x_file_name = os.path.join(self.data_path, 'test_sensor_x.npy')
            npz_test_y_file_name = os.path.join(self.data_path, 'test_sensor_y.npy')

            X_train = np.load(npz_train_x_file_name)
            Y_train = np.load(npz_train_y_file_name)
            Y_train = np.array(Y_train, dtype=int)

            X_test = np.load(npz_test_x_file_name)
            Y_test = np.load(npz_test_y_file_name)
            Y_test = np.array(Y_test, dtype=int)
        else:
            npz_x_file_name = os.path.join(self.data_path, 'sensor_x.npy')
            npz_y_file_name = os.path.join(self.data_path, 'sensor_y.npy')

            X_data = np.load(npz_x_file_name)
            Y_data = np.load(npz_y_file_name)
            Y_data = np.array(Y_data, dtype=int)

            X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data,
                                                                test_size=0.2, 
                                                                shuffle=True, 
                                                                random_state=1004)

        self.class_counts = self.get_class_counts(Y_train, Y_test)
        # print(Y_test)
        
        X_train = torch.FloatTensor(X_train)
        Y_train = torch.LongTensor(Y_train)

        X_test = torch.FloatTensor(X_test)
        Y_test = torch.LongTensor(Y_test)
        
        self.data_splits['train'] = (X_train, Y_train)
        self.data_splits['test'] = (X_test, Y_test)

    def get_class_counts(self, y_train, y_test):
        class_counts = {}
        
        class_counts['train'] = np.zeros(self.class_num)
        for cls_num in range(self.class_num):
            class_counts['train'][cls_num] = collections.Counter(y_train)[cls_num]

        class_counts['test'] = np.zeros(self.class_num)
        for cls_num in range(self.class_num):
            class_counts['test'][cls_num] = collections.Counter(y_test)[cls_num]

        print('class_counts train:', class_counts['train'])
        print('class_counts test:', class_counts['test'])
        return class_counts

    def get_loader(self, name, transfs):
        data_split = self.data_splits[name]
        dataset = NpzDataset(data_split, transforms=transfs, time_start = self.time_start, time_end = self.time_end, time_length = self.time_length, method = self.method)

        if name == 'train':
            if self.imbalance == 'over':
                print('=====================> over')
                class_counts = self.class_counts[name] #43200, 4800
                num_samples = sum(class_counts)
                class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
                weights = [class_weights[data_split[1][i]] for i in range(int(num_samples))] #해당 레이블마다의 가중치 비율
                sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
                return data.DataLoader(dataset=dataset, **self.loader_params, sampler = sampler)
            else:
                return data.DataLoader(dataset=dataset, **self.loader_params)
        else:
            return data.DataLoader(dataset=dataset, **self.loader_params)

if __name__ == '__main__':
    pass    
