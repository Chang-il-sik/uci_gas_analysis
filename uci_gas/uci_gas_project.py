import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../")
import uci_gas.utils as utils
from uci_gas.model import Gas_Model
import numpy as np
import argparse
import json
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from uci_gas.decoupling import KNNClassifier
import copy
from pathlib import Path

class GAS_Proj():
    def __init__(self, config, action):
        self.config = config
        
        self.epoch = 1
        self.steps = 0
        self.epochs = config['train']['epochs']
        self.cuda = config['cuda']
        self.clip = config['train']['clip']

        self.batch_size = config['data']['loader']['batch_size']
        
        self.class_num = config['class_num']
            
        self.model = Gas_Model(config)
        if config['parallel']:
            self.model = torch.nn.DataParallel(self.model)

        if self.cuda:
            self.model = self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        opt_name = config['optimizer']['type']
        opt_args = config['optimizer']['args']
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
        self.lr = opt_args['lr']

        lr_name = config['lr_scheduler']['type']
        lr_args = config['lr_scheduler']['args']
        if lr_name == 'None':
            self.lr_scheduler = None
        else:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(self.optimizer, **lr_args)

        self.loss_func = config['train']['loss']
        
        if self.loss_func == 'arcface' or self.loss_func == 'cosface' or self.loss_func == 'sphereface':
            pass
        elif self.loss_func == 'CenterLoss':
            self.loss = getattr(utils, 'NllLoss')(**config['train']['NllLoss_args'])
            self.optimizer_centloss = torch.optim.SGD(self.model.module.criterion_cent.parameters(), lr=0.5)
        else:
            self.loss_args = config['train'][self.loss_func+'_args']
            self.loss = getattr(utils, self.loss_func)(**self.loss_args)
        
        if self.loss_func=='LabelSmoothingLoss':
            self.loss.cls = self.class_num

        self.mnt_best = 0

        self.log_interval = config['log_interval']
        self.save_period = config['train']['save_p']
        self.early_stop = config['train']['early_stop']
        self.best = False
        self.not_improved_count = 0
        
        self.train_loader = None
        self.test_loader = None
        self.test_accuracy = 0
        self.train_loss = 0
        self.test_loss = 0
        self.train_accuracy = 0

        self.imbalance = config['data']['imbalance']
        self.dist_type = config['train']['MetricLoss_arg']['dist_type']
        
    def trainer_init(self, args, config, resume):
        cfg_trainer = config['train']
        basename = os.path.basename(args.config)
        self.log_dir = os.path.join(cfg_trainer['save_dir'], os.path.splitext(basename)[0])
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir) 

        # setup directory for checkpoint saving
        # start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        self.save_list = []

        # Save configuration file into checkpoint directory:
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        if self.config.get('cfg', None) is not None:
            cfg_save_path = os.path.join(self.checkpoint_dir, 'model.cfg')
            with open(cfg_save_path, 'w') as fw:
                fw.write(open(self.config['cfg']).read())
            self.config['cfg'] = cfg_save_path

        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)
                    
    def train(self):
        train_loss = 0
        correct = 0
        self.model.train()

        self.total_logits = torch.empty((0, self.class_num)).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()

        # y_label = np.array([])
        # y_pred = np.array([])
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # print('ep:', self.epoch)
            # print('target:', target)
            # print('==========>', data.shape, self.batch_size, batch_idx)
            
            if self.cuda: 
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            if self.loss_func == 'CenterLoss':
                self.optimizer_centloss.zero_grad()
            self.optimizer.zero_grad()
            # print(target)

            if self.loss_func == 'arcface' or self.loss_func == 'cosface' or self.loss_func == 'sphereface':
                out = self.model(data, embed=True)
                loss = self.model.adms_loss.forward(out, target)
            elif self.loss_func == 'CenterLoss':
                features, out = self.model(data, embed=True, out_flag=True)
                output = F.log_softmax(out, dim=1)
                loss_xent = self.loss.forward(output, target)
                loss_cent = self.model.criterion_cent(features, target)
                # loss_cent *= args.weight_cent
                loss = loss_xent + loss_cent
                # print('loss:', loss_xent.item(), loss_cent.item())
            else:
                out = self.model(data)
                output = F.log_softmax(out, dim=1)
                    
                # print('self.loss_func:', self.loss_func)
                # print('out:', out.shape)
                print('output:', output.shape)
                print('target:', target.shape)
                loss = self.loss.forward(output, target)
                        
            loss.backward()
            
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            if self.loss_func == 'CenterLoss':
                self.optimizer_centloss.step()
            self.optimizer.step()

            train_loss += loss
            self.steps += 1
            
            if self.loss_func == 'arcface' or self.loss_func == 'cosface' or self.loss_func == 'sphereface':
                pass
            else:
                self.total_logits = torch.cat((self.total_logits, output))
                self.total_labels = torch.cat((self.total_labels, target))
                    
                pred = output.data.max(1, keepdim=True)[1].cpu()
                label = target.cpu()
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()
                
                if batch_idx > 0 and batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                        self.epoch, batch_idx * self.batch_size, len(self.train_loader.dataset),
                        100. * batch_idx / self.train_loader_len, train_loss.item()/self.log_interval, self.steps))
                    # train_loss = 0

        self.train_loss = train_loss.detach().cpu().numpy()/self.train_loader_len

        if self.loss_func == 'arcface' or self.loss_func == 'cosface' or self.loss_func == 'sphereface':
            print('=====> train_loss:', self.train_loss)
        else:
            print('\nTrain Accuracy: {}/{} ({:.0f}%)\n'.format(correct, self.train_loader_len,
                100. * correct / self.train_loader_len))
            self.train_accuracy = correct.numpy() / self.train_loader_len

    def test(self, save_flag=False):
        self.model.eval()
        if self.loss_func == 'arcface' or self.loss_func == 'cosface' or self.loss_func == 'sphereface':
            self.init_NCM()
            self.test_NCM(save_flag=True, method=self.loss_func)
        else:
            test_loss = 0
            correct = 0
            correct2 = 0
            
            with torch.no_grad():
                y_pred = np.array([])
                y_label = np.array([])

                for data, target in self.test_loader:
                    if self.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)
                    # print('target:', target)
                    # print('output:', output)
                    if self.loss_func == 'arcface' or self.loss_func == 'cosface' or self.loss_func == 'sphereface':
                        out = self.model(data, embed=True)
                        test_loss += self.model.adms_loss.forward(out, target)
                    else:
                        out = self.model(data)
                        output = F.log_softmax(out, dim=1)
                        
                        loss = self.loss.forward(output, target)
                        test_loss += loss

                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                    
                    pred = output.data.max(1, keepdim=False)[1].cpu().numpy()
                    y_pred = np.concatenate((y_pred, pred), axis=None)
                    label = target.cpu().numpy()
                    y_label = np.concatenate((y_label, label), axis=None)
                    # print('y_pred:', y_pred)
                    # print('label:', label)
                    
                self.test_accuracy = correct.cpu().numpy() / self.test_loader_len
                self.test_loss = test_loss.cpu().numpy() / self.test_loader_len

                self.save_result(y_label, y_pred, save_flag=save_flag)
            
    def run_lr_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        else:
            if self.epoch % 10 == 0:
                self.lr /= 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def pre_run(self, epoch):
        self.epoch = epoch
        if epoch == 1:
            self.train_loader, self.test_loader, self.data_manager = utils.data_generator(self.config, epoch)
            self.train_loader_len = len(self.train_loader.dataset)
            self.test_loader_len = len(self.test_loader.dataset)
    
    def save_result(self, y_label, y_pred, fig_save=False, save_flag=False, method='None'):
        pre_text = 'single'
        
        target_names = []
        for id in range(self.class_num):
            target_names.append('class {}'.format(id+1))
            
        if save_flag:
            try:
                result_metrics = metrics.classification_report(y_label, y_pred, target_names=target_names)
                with open(os.path.join(self.log_dir, '{}_report_{}.txt'.format(pre_text, self.epoch)), "w") as f:
                    f.write(result_metrics)
            except:
                print('metrics.classification_report error')
        
        c_mat = confusion_matrix(y_label, y_pred)
        sum_rows = np.expand_dims(c_mat.astype(np.float64).sum(axis=1), axis=1)
        c_mat = np.round(c_mat / sum_rows, 2)
        if fig_save:
            disp = ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels=target_names)
            disp.plot()
            # plt.show()
            plt.savefig(os.path.join(self.log_dir, '{}_cm_{}.png'.format(pre_text, self.epoch)))
            plt.clf()

        result = {}

        # print('c_mat:', c_mat.shape)
        
        c_mat_sum = np.sum(c_mat)
        sum = 0
        for idx in range(len(target_names)):
            sum += c_mat[idx][idx]
        result['test_cm_acc'] = sum/c_mat_sum
        self.weighted_accuracy = result['test_cm_acc']
        
        # print('Epoch : {}'.format(self.epoch))
        print('Weighted Accuracy : {:.3f}'.format(self.weighted_accuracy))
        print('Unweighted Accuracy : {:.3f}'.format(self.test_accuracy))
        print('Train Accuracy : {:.3f}'.format(self.train_accuracy))
        print('Average Train loss: {:.3f}'.format(self.train_loss))
        print('Average Test loss: {:.3f}'.format(self.test_loss))

        if save_flag:
            if method == 'None':
                self.save_list.append( {'epoch' : self.epoch, 
                                            'WA':self.weighted_accuracy, 
                                            'UA':self.test_accuracy,
                                            'TA':self.train_accuracy,
                                            'Train Loss':self.train_loss,
                                            'Test Loss':self.test_loss} )
            else:
                self.save_list.append( {'method' : method,
                                            'epoch' : self.epoch, 
                                            'WA':self.weighted_accuracy, 
                                            'UA':self.test_accuracy,
                                            'TA':self.train_accuracy,
                                            'Train Loss':self.train_loss,
                                            'Test Loss':self.test_loss} )
                
    def update(self):
        improved = False
        if self.weighted_accuracy > self.mnt_best:
            improved = True

        if improved:
            self.mnt_best = self.weighted_accuracy
            self.not_improved_count = 0
            self.best = True
        else:
            self.not_improved_count += 1

        if self.not_improved_count > self.early_stop:
            return False

        if self.epoch % self.save_period == 0:
            self._save_checkpoint(self.epoch, save_best=self.best)
            self.best = False
        
        return True

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
        }
            
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-current.pth')
        # filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        # torch.save(state, filename)
        # print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: {} ...".format('model_best.pth'))
            print("[IMPROVED]")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch']
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def get_knncentroids(self):
        self.model.eval()

        feats_all, labels_all = [], []
        # model = tx.Extractor(self.model, ["f_linear"])

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for data, target in self.train_loader:
                inputs, labels = data.cuda(), target.cuda()

                # Calculate Features of each training data
                inputs = Variable(inputs)
                features = self.model(inputs, embed=True)

                feats_all.append(features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        
        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all)

        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids = []        
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_==i], axis=0))
            return np.stack(centroids)
        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)
    
        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        # print('featmean', featmean.shape)
        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,   
                'cl2ncs': cl2n_centers}

    def test_NCM(self, save_flag=False, method='NCM'):
        self.model.eval()
        test_loss = 0
        correct = 0

        # model = tx.Extractor(self.model, ["f_linear"])
        
        with torch.no_grad():
            y_pred = np.array([])
            y_label = np.array([])

            for data, target in self.test_loader:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                
                features = self.model(data, embed=True)
                # print('===> features["f_linear"]:', features["f_linear"].shape)
                output, _ = self.knn_classifier(features)

                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                
                pred = output.data.max(1, keepdim=False)[1].cpu().numpy()
                y_pred = np.concatenate((y_pred, pred), axis=None)
                label = target.cpu().numpy()
                y_label = np.concatenate((y_label, label), axis=None)

            self.test_accuracy = correct.cpu().numpy() / self.test_loader_len

            self.save_result(y_label, y_pred, save_flag=save_flag, method=method)

    def init_NCM(self):
        with torch.no_grad():
            cfeats = self.get_knncentroids()
            # print('len(cfeats[].featmean)', len(cfeats['mean']))
            self.knn_classifier = KNNClassifier.create_model(len(cfeats['mean']), num_classes=self.class_num, dist_type=self.dist_type)
            self.knn_classifier.update(cfeats)

    def pnorm(self, weights, p):
        normB = torch.norm(weights, 2, 1)
        ws = weights.clone()
        for i in range(weights.size(0)):
            ws[i] = ws[i] / torch.pow(normB[i], p)
        return ws

    def dotproduct_similarity(self, A, B):
        feat_dim = A.size(1)
        AB = torch.mm(A, B.t())

        return AB

    def ood_init(self, args):
        self.save_list = []

        self.train_loader, self.test_loader, self.data_manager = utils.data_generator(self.config, epoch)
        
        # self.pre_run(1)

        gas_proj.log_dir = Path(args.resume).parent.parent
        self._resume_checkpoint(args.resume)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
    parser.add_argument('action', type=str, help='what action to take (train, test, eval)')
    #'one_way, bi_dir, sp_bi_dir, dialated_conv, h_dialated_conv, wavenet, crnn, rnn, brnn, cnn'
    parser.add_argument('-c', '--config', default=None, type=str,
                            help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                            help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    # Resolve config vs. resume
    checkpoint = None
    if args.config:
        config = json.load(open(args.config))
    elif args.resume:
        checkpoint = torch.load(args.resume)
        config = checkpoint['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    gas_proj = GAS_Proj(config, args.action)
    
    if args.action == 'train':
        gas_proj.trainer_init(args, config, False)
        
        for epoch in range(1, gas_proj.epochs+1):
            print('epoch:{}/{}'.format(epoch, gas_proj.epochs))
            gas_proj.pre_run(epoch)
            gas_proj.train()
            gas_proj.test(True)
            gas_proj.run_lr_scheduler()
            gas_proj.update()
        
        save_df = pd.DataFrame(gas_proj.save_list)
        save_df.to_csv(os.path.join(gas_proj.log_dir, 'result.csv'), index = False)

    elif args.action == 'eval':
        gas_proj.ood_init(args)
        
        # gas_proj.init_NCM()
        # gas_proj.test_NCM(save_flag=True)
        
        # save_df = pd.DataFrame(gas_proj.save_list)
        # save_df.to_csv(os.path.join(gas_proj.log_dir, 'ood_result.csv'), index = False)
        
    elif args.action == 'test':
        gas_proj._resume_checkpoint(args.resume)
        gas_proj.pre_run(1)
        gas_proj.test(False)