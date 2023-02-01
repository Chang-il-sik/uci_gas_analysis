import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import numpy as np
from matplotlib import rc, rcParams

rc('font', weight='bold')
title_flag = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--path', type=str, default='./', help='path')
    parser.add_argument('--zo_flag', type=str2bool, default=False, help='zo_flag')
    parser.add_argument('--linewidth', type=int, default=3, help='linewidth')
    args = parser.parse_args()

    base_dir_name = str(Path(args.path)).split('/')[-1]
    cs_file_name = os.path.join(args.path, 'best_result.csv')

    cs_df = pd.read_csv(cs_file_name)
    cs_drop_df = cs_df.dropna(axis=0)

    net_lst = cs_drop_df['Network'].unique()
    loss_lst = cs_drop_df['Loss'].unique()
    loss_lst = ['NllLoss', 'FocalLoss', 'arcface', 'cosface', 'sphereface']
    loss_label_lst = ['SoftmaxLoss', 'FocalLoss', 'arcface', 'cosface', 'sphereface']

    detector_lst = ['ViM', 'Mahalanobis', 'KLMatching', 'MaxSoftmax', 'EnergyBased', 'MaxLogit', 'ODIN', 'OpenMax']
    metric_str_lst = ['AUROC', 'AUPR-IN', 'AUPR-OUT', 'ACC95TPR', 'FPR95TPR']
    dataset_str = '0_20'
    
    net_disp_lst = []
    for idx, net in enumerate(net_lst):
        if net == 'sp_bi_dir':
            net_disp_lst.append('bi_tcn')
        else:
            net_disp_lst.append(net)

    if args.zo_flag:
        base_dir = 'zero_offset'
        prefix_name = 'sensor_npz_zo'
    else:
        base_dir = 'raw_signal'
        prefix_name = 'sensor_npz'

    # plt.figure(figsize=(10,8))
    
    for net in net_lst:
        for idx, loss in enumerate(loss_lst):
            for detector in detector_lst:
                x_label = []
                y_lst = []
                for metric in metric_str_lst:
                    loc_dir = '{}-{}-{}'.format(prefix_name, net, loss)
                    
                    ood_csv_name = os.path.join(os.path.join(args.path, loc_dir), 'result_ood.csv')
                    ood_df = pd.read_csv(ood_csv_name)
                    ood_df = ood_df.dropna(axis=0)

                    ood_metric_df = ood_df[['Detector', 'Dataset', metric]]
                    
                    sel_drop_df = ood_metric_df[(ood_metric_df['Detector'] == detector) & (ood_metric_df['Dataset'] == dataset_str)]
                    value = list(sel_drop_df[metric])[0]
                    
                    x_label.append(metric)
                    y_lst.append(value)
                plt.plot(x_label, y_lst, marker='o', linewidth=args.linewidth)

            if title_flag:
                plt.title('{}_{}_OOD_Result'.format(net, loss))
            plt.xlabel('OOD Metric', weight='bold')
            # plt.ylabel('Accuracy')
            plt.legend(detector_lst, framealpha=0.2)
            
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            png_file_name = '{}_{}_{}_plot.png'.format(base_dir_name, net, loss)
            if os.path.isfile(os.path.join(base_dir, png_file_name)):
                os.remove(os.path.join(base_dir, png_file_name))
            plt.savefig(os.path.join(base_dir, png_file_name))
            # plt.show()
            plt.clf()
            # exit()
