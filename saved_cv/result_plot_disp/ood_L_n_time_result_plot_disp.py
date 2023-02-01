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
    parser.add_argument('--path', type=str, default='./', help='path')
    parser.add_argument('--zo_flag', type=str2bool, default=False, help='zo_flag')
    parser.add_argument('--mode', type=str, default='Ln', help='mode')
    parser.add_argument('--linewidth', type=int, default=3, help='linewidth')
    
    args = parser.parse_args()

    base_dir_name = str(Path(args.path)).split('/')[-1]

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
    dataset_lst = []
    for t in range(start_time, end_time):
        dataset_lst.append('{}_{}'.format(t, t+time_len))

    if args.zo_flag:
        base_dir = 'zero_offset'
        prefix_name = 'sensor_npz_zo'
    else:
        base_dir = 'raw_signal'
        prefix_name = 'sensor_npz'

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    metric = 'AUROC'
    
    if mode == 'Ln' or mode == 'Lcnn':
        plt.rcParams['figure.figsize'] = [10, 2]
        
        if mode == 'Ln':
            net_lst = ['brnn_att', 'sp_bi_dir']
            net_disp_lst = ['brnn_att', 'bi_tcn']
            m_str = 'rnn'
        else:
            net_lst = ['cnn', 'crnn']
            net_disp_lst = ['cnn', 'crnn']
            m_str = 'cnn'

        for net_idx, net in enumerate(net_lst):
            for loss_idx, loss in enumerate(loss_lst):
                for detector in detector_lst:
                    dataset_df_lst = []
                    for dataset_str in dataset_lst:
                        heatmap_dict = {}
                        for w in range(1, Width+1):
                            loc_dir = '{}-L{}_{}-{}-{}'.format(prefix_name, w, m_str, net, loss)
                            
                            ood_csv_name = os.path.join(os.path.join(args.path, loc_dir), 'result_ood.csv')
                            ood_df = pd.read_csv(ood_csv_name)
                            ood_df = ood_df.dropna(axis=0)
                            
                            ood_metric_df = ood_df[['Detector', 'Dataset', metric]]
                            
                            sel_drop_df = ood_metric_df[(ood_metric_df['Detector'] == detector) & (ood_metric_df['Dataset'] == dataset_str)]
                            value = list(sel_drop_df[metric])[0]
                            heatmap_dict[col_str_lst[w-1]] = value

                        df = pd.DataFrame([heatmap_dict], index=[metric])
                        dataset_df_lst.append(df)
                        
                    min_value = 1
                    max_value = 0
                    for d_idx, dataset_df in enumerate(dataset_df_lst):
                        dataset_np = dataset_df.to_numpy()
                        if min_value > dataset_np.min():
                            min_value = dataset_np.min()
                        if max_value < dataset_np.max():
                            max_value = dataset_np.max()

                    for w in range(Width):
                        x_label = []
                        y_lst = []
                        for d_idx, dataset_df in enumerate(dataset_df_lst):
                            dataset_np = dataset_df.to_numpy().squeeze()
                            # print(dataset_np.shape)
                            y_lst.append(dataset_np[w])
                            x_label.append(d_idx)
                        
                        plt.plot(x_label, y_lst, marker='o', linewidth=args.linewidth)
                        if title_flag:
                            plt.title('{}_{}_{}_L{}_0-100_OOD Result'.format(net, loss, detector, w+1))
                        plt.xlabel('OOD {}'.format(metric), weight='bold')
                        plt.ylim(min_value, max_value)

                        json_file_name = '{}_{}_{}_{}_L{}.json'.format(base_dir_name, net, loss, detector, w+1)
                        json_data = {}
                        json_data['x_label'] = x_label
                        json_data['y_lst'] = y_lst
                        json_data['min_value'] = min_value
                        json_data['max_value'] = max_value
                        # print(json_data)
                        with open(os.path.join(base_dir, json_file_name), "w") as json_file:
                            json.dump(json_data, json_file)
                        
                        png_file_name = '{}_{}_{}_{}_L{}_plot.png'.format(base_dir_name, net, loss, detector, w+1)
                        if os.path.isfile(os.path.join(base_dir, png_file_name)):
                            os.remove(os.path.join(base_dir, png_file_name))
                        plt.savefig(os.path.join(base_dir, png_file_name))
                        # plt.show()
                        plt.clf()
                        # exit()

    elif mode == 'Lnn':
        net_lst = ['brnn_att', 'sp_bi_dir']
        net_disp_lst = ['brnn_att', 'bi_tcn']
        print('zo_flag:', args.zo_flag)

        for net_idx, net in enumerate(net_lst):
            for loss_idx, loss in enumerate(loss_lst):
                for detector in detector_lst:
                    dataset_df_lst = []
                    for dataset_str in dataset_lst:
                        heatmap_lst = []
                        for h in range(1, Height+1):
                            heatmap_dict = {}
                            for w in range(1, Width+1):
                                loc_dir = '{}-L{}_{}-{}-{}'.format(prefix_name, w, h, net, loss)
                                
                                ood_csv_name = os.path.join(os.path.join(args.path, loc_dir), 'result_ood.csv')
                                ood_df = pd.read_csv(ood_csv_name)
                                ood_df = ood_df.dropna(axis=0)
                                
                                ood_metric_df = ood_df[['Detector', 'Dataset', metric]]
                                
                                sel_drop_df = ood_metric_df[(ood_metric_df['Detector'] == detector) & (ood_metric_df['Dataset'] == dataset_str)]
                                value = list(sel_drop_df[metric])[0]
                                heatmap_dict[col_str_lst[w-1]] = value
                            heatmap_lst.append(heatmap_dict)
                            
                        # print(heatmap_lst)
                        df = pd.DataFrame(heatmap_lst, index=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9'])
                        dataset_df_lst.append(df)

                    min_value = 1
                    max_value = 0
                    for d_idx, dataset_df in enumerate(dataset_df_lst):
                        dataset_np = dataset_df.to_numpy().squeeze()
                        if min_value > dataset_np.min():
                            min_value = dataset_np.min()
                        if max_value < dataset_np.max():
                            max_value = dataset_np.max()

                    for h in range(Height):
                        for w in range(Width):
                            x_label = []
                            y_lst = []
                            for d_idx, dataset_df in enumerate(dataset_df_lst):
                                dataset_np = dataset_df.to_numpy()
                                # print(dataset_np.shape)
                                y_lst.append(dataset_np[h][w])
                                x_label.append(d_idx)
                            
                            plt.plot(x_label, y_lst, marker='o', linewidth=args.linewidth)
                            if title_flag:
                                plt.title('{}_{}_{}_L{}_{}_0-100_OOD Result'.format(net, loss, detector, w+1, h+1))
                            plt.xlabel('OOD {}'.format(metric), weight='bold')
                            plt.ylim(min_value, max_value)

                            json_file_name = '{}_{}_{}_{}_L{}_{}.json'.format(base_dir_name, net, loss, detector, w+1, h+1)
                            json_data = {}
                            json_data['x_label'] = x_label
                            json_data['y_lst'] = y_lst
                            json_data['min_value'] = min_value
                            json_data['max_value'] = max_value
                            # print(json_data)
                            with open(os.path.join(base_dir, json_file_name), "w") as json_file:
                                json.dump(json_data, json_file, indent=4)
    
                            png_file_name = '{}_{}_{}_{}_L{}_{}_plot.png'.format(base_dir_name, net, loss, detector, w+1, h+1)
                            if os.path.isfile(os.path.join(base_dir, png_file_name)):
                                os.remove(os.path.join(base_dir, png_file_name))
                            plt.savefig(os.path.join(base_dir, png_file_name))
                            # plt.show()
                            plt.clf()
                            # exit()

