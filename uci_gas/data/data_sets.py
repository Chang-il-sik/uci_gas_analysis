"""
Dataset in charge of knowing where the source of data, labels and transforms.
Should provide access to the data by indexing.
"""

# import os
# import pandas as pd
import numpy as np
import torch.utils.data as data

class NpzDataset(data.Dataset):
    def __init__(self, data_arr, transforms=None, time_start=150, time_end=250, time_length=20, method='cnn'):
        self.transforms = transforms
        self.data_lst, self.labels = data_arr
        self.min = time_start
        self.max = time_end - time_length
        self.length = time_length
        self.method = method
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
            # data = self.data_lst[index][0:20, :]
        
        # print('====> data : ', self.data_lst[index].shape)
        # print('==> data : ', data.shape)
        
        label = self.labels[index]

        if self.transforms is not None:
            t_data = self.transforms.apply(data)
            return t_data, label

        return data, label
    
if __name__ == '__main__':
    pass










