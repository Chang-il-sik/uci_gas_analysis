import numpy as np
import torch
from torchvision import transforms
import random
import torch.nn as nn

class SensorTransforms(object):
    def __init__(self, name, args, mean, std):
        self.args = args
        self.mean = mean
        self.std = std
        self.transfs = {
            'test': transforms.Compose([
                # transforms.ToTensor(), 
                # transforms.Normalize(mean, std),
                transfer(args, mean, std)
            ]),
            'train': transforms.Compose([
                # transforms.ToTensor(), 
                # transforms.Normalize(mean, std),
                transfer(args, mean, std)
                # AddNoise(*args['noise']),
            ])
        }[name]

    def apply(self, data):
        # print('data', data.shape)
        # print(data)
        # audio -> (time, channel)
        return self.transfs(data)
        
    def __repr__(self):
        return self.transfs.__repr__()

class AddNoise(nn.Module):
    def __init__(self, prob, std=1e-4):
        """Add Gaussian noise.
        Args:
        snr: float
        """
        super(AddNoise, self).__init__()

        self.prob = prob
        self.std = std

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        if torch.rand(1)[0] < self.prob:
            input = input + self.std*torch.randn(input.shape[0],input.shape[1],input.shape[2]).cuda()
            # print('===> input', input.shape)

        return input

class transfer(object):
    def __init__(self, args, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)
        
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def __call__(self, tensor):
        # print('========tensor: ', tensor.shape)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.cuda()

        tensor = (tensor - self.mean)/self.std
        
        return tensor


