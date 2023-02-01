import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import numpy as np
import seaborn as sns
import json
from matplotlib import rc, rcParams

rc('font', weight='bold')
title_flag = False

Width = 6
Height = 9

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
    parser.add_argument('--zo_flag', type=str2bool, default=False, help='zo_flag')
    parser.add_argument('--mode', type=str, default='Ln', help='mode')
    parser.add_argument('--linewidth', type=int, default=3, help='linewidth')
    args = parser.parse_args()

    mode = args.mode.strip()

    loss_lst = ['NllLoss', 'cosface']
    loss_label_lst = ['SoftmaxLoss', 'cosface' ]
    # detector_lst = ['ViM', 'Mahalanobis', 'KLMatching', 'MaxSoftmax', 'EnergyBased', 'MaxLogit', 'ODIN', 'OpenMax']
    detector_lst = ['ViM', 'Mahalanobis']
    # metric_str_lst = ['AUROC', 'AUPR-IN', 'AUPR-OUT', 'ACC95TPR', 'FPR95TPR']
    col_str_lst = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']

    start_time = 0
    end_time = 100
    time_len = 20

    metric = 'AUROC'
    
    if mode == 'Ln' or mode == 'Lcnn':
        # plt.rcParams['figure.figsize'] = [10, 2]
        if args.zo_flag:
            base_dir = 'ood_L_n_time_result/zero_offset'
        else:
            base_dir = 'ood_L_n_time_result/raw_signal'
        
        if mode == 'Ln':
            net_lst = ['brnn_att', 'sp_bi_dir']
            net_disp_lst = ['brnn_att', 'bi_tcn']
            m_str = 'rnn'
            if args.zo_flag:
                prefix_name = 'L_n_zo_result'
            else:
                prefix_name = 'L_n_result'
        else:
            net_lst = ['cnn', 'crnn']
            net_disp_lst = ['cnn', 'crnn']
            m_str = 'cnn'
            if args.zo_flag:
                prefix_name = 'L_cnn_zo_result'
            else:
                prefix_name = 'L_cnn_result'

        legend_str_lst = []
        
        for w in range(Width):
            for net_idx, net in enumerate(net_lst):
                for loss_idx, loss in enumerate(loss_lst):
                    for detector in detector_lst:
                        json_file = '{}_{}_{}_{}_L{}.json'.format(prefix_name, net, loss, detector, w+1)
                        with open(os.path.join(base_dir, json_file)) as f:
                            json_object = json.load(f)
                            plt.plot(json_object['x_label'], json_object['y_lst'], marker='.', linewidth=args.linewidth)
                            if title_flag:
                                plt.title('L{}_0-100_OOD Result'.format(w+1))
                        legend_str_lst.append('{}_{}_{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx], detector))

            png_file_name = '{}_L{}_plot.png'.format(m_str, w+1)
            plt.xlabel('OOD {}'.format(metric), weight='bold')
            plt.legend(legend_str_lst, framealpha=0.2)
            plt.ylim(json_object['min_value'], json_object['max_value'])
            if os.path.isfile(os.path.join(base_dir, png_file_name)):
                os.remove(os.path.join(base_dir, png_file_name))
            plt.savefig(os.path.join(base_dir, png_file_name))
            # plt.show()
            plt.clf()
            # exit()

    elif mode == 'Lnn':
        if args.zo_flag:
            base_dir = 'ood_L_n_n_time_result/zero_offset'
            prefix_name = 'L_n_n_zo_result'
        else:
            base_dir = 'ood_L_n_n_time_result/raw_signal'
            prefix_name = 'L_n_n_result'

        net_lst = ['brnn_att', 'sp_bi_dir']
        net_disp_lst = ['brnn_att', 'bi_tcn']
        print('zo_flag:', args.zo_flag)
        legend_str_lst = []
        
        for h in range(Height):
            for w in range(Width):
                for net_idx, net in enumerate(net_lst):
                    for loss_idx, loss in enumerate(loss_lst):
                        for detector in detector_lst:
                            json_file = '{}_{}_{}_{}_L{}_{}.json'.format(prefix_name, net, loss, detector, w+1, h+1)
                            with open(os.path.join(base_dir, json_file)) as f:
                                json_object = json.load(f)
                                plt.plot(json_object['x_label'], json_object['y_lst'], marker='.', linewidth=args.linewidth)
                                if title_flag:
                                    plt.title('L{}_{}_0-100_OOD Result'.format(w+1, h+1))
                            legend_str_lst.append('{}_{}_{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx], detector))
                            
                png_file_name = 'L{}_{}_plot.png'.format(w+1, h+1)
                plt.xlabel('OOD {}'.format(metric), weight='bold')
                plt.legend(legend_str_lst, framealpha=0.2)
                plt.ylim(json_object['min_value'], json_object['max_value'])
                if os.path.isfile(os.path.join(base_dir, png_file_name)):
                    os.remove(os.path.join(base_dir, png_file_name))
                plt.savefig(os.path.join(base_dir, png_file_name))
                # plt.show()
                plt.clf()
                # exit()

