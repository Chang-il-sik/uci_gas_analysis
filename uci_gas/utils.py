import torch
from torchvision import datasets, transforms
import os
import sys
import data.data_manager as data_module
import data.transforms as data_transfoms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

def _get_transform(config, name):
    tsf_name = config['transforms']['type']
    tsf_args = config['transforms']['args']
    start = config['data']['start']
    end = config['data']['end']
    
    with open(os.path.join(config['data']['path'], 'sensor_info.json')) as f:
        json_object = json.load(f)
        mean = json_object['sensor_data_mean_{}_{}'.format(start, end)]
        std = json_object['sensor_data_std_{}_{}'.format(start, end)]

    # print('mean:', mean)
    # print('std:', std)
    return getattr(data_transfoms, tsf_name)(name, tsf_args, mean, std)

def data_generator(config, epoch):
    # print(os.getcwd())
    
    data_manager = getattr(data_module, config['data']['type'])(config, epoch)

    train_transforms = _get_transform(config, 'train')
    test_transforms = _get_transform(config, 'test')
    
    train_loader = data_manager.get_loader('train', train_transforms)
    test_loader = data_manager.get_loader('test', test_transforms)
    
    return train_loader, test_loader, data_manager

class NllLoss(nn.Module):
    def __init__(self, weight=None, self_paced=False):
        super(NllLoss, self).__init__()
        self.weight = torch.tensor(weight).cuda()
        if self_paced:
            self.reduction = 'none'
        else:
            self.reduction = 'mean'
            
    def forward(self, output, target):
        # print('self_paced:', self.reduction)
        nll_loss = F.nll_loss(output, target, weight=self.weight, reduction=self.reduction)
        return nll_loss

# def nll_loss(output, target):
#     # loss for log_softmax
#     return F.nll_loss(output, target)

# def cross_entropy(output, target):
#     return F.cross_entropy(output, target)

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean', self_paced=False):
        if weight != None:
            weight = torch.tensor(weight).cuda()
        if self_paced:
            reduction='none'
            
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, output, target):
        # ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        ce_loss = F.nll_loss(output, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss

# def focal_loss(output, target):
#     # print('output:', output)
#     f_loss = FocalLoss(gamma=2)
#     return f_loss.forward(output, target)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, weight=None, smoothing=0.1, dim=-1, self_paced=False):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = torch.tensor(weight).cuda()
        self.self_paced = self_paced
        
    def forward(self, output, target):
        # output = output.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(output)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.weight != None:
                # print('[0]', true_dist[0], self.weight, self.cls)
                true_dist = torch.mul(true_dist, self.weight)
                # print('[1]', true_dist[0])
        
        if self.self_paced:
            ls_loss = torch.sum(-true_dist * output, dim=self.dim)
        else:
            ls_loss = torch.mean(torch.sum(-true_dist * output, dim=self.dim))
        # print('ls_loss:', ls_loss)
        return ls_loss

# def label_smoothing_loss(output, target):
#     l_loss = LabelSmoothingLoss(smoothing = 0.1)
#     return l_loss.forward(output, target)

def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score


def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce

def get_priority(ptype, logits, labels):
    if ptype == 'score':
        ws = 1 - logits2score(logits, labels)
    elif ptype == 'entropy':
        ws = logits2entropy(logits)
    elif ptype == 'CE':
        ws = logits2CE(logits, labels)
    
    return ws

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, self_paced=False):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = torch.tensor(weight).cuda()
        self.self_paced = self_paced

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        if self.self_paced:
            loss = F.cross_entropy(self.s*output, target, weight=self.weight, reduction='none')
        else:
            loss = F.cross_entropy(self.s*output, target, weight=self.weight)
        return loss

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        # self.fc = nn.Linear(in_features, out_features, bias=False)
        self.fc = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=W.dim()-1)

        # for W in self.fc.parameters():
        #     W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    