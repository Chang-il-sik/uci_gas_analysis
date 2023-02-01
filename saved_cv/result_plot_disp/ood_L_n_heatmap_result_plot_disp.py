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
    parser.add_argument('--line_flag', type=str2bool, default=False, help='line_flag')
    parser.add_argument('--mode', type=str, default='Ln', help='mode')
    parser.add_argument('--linewidth', type=int, default=3, help='linewidth')
    args = parser.parse_args()

    base_dir_name = str(Path(args.path)).split('/')[-1]

    mode = args.mode.strip()

    loss_lst = ['NllLoss', 'cosface']
    loss_label_lst = ['SoftmaxLoss', 'cosface' ]
    detector_lst = ['ViM', 'Mahalanobis', 'KLMatching', 'MaxSoftmax', 'EnergyBased', 'MaxLogit', 'ODIN', 'OpenMax']
    metric_str_lst = ['AUROC', 'AUPR-IN', 'AUPR-OUT', 'ACC95TPR', 'FPR95TPR']
    col_str_lst = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    dataset_str = '0_20'

    if mode == 'Ln' or mode == 'Lcnn':
        if args.zo_flag:
            base_dir = 'ood_L_n_heatmap_result_plot_disp_run/zero_offset'
            prefix_name = 'sensor_npz_zo'
        else:
            base_dir = 'ood_L_n_heatmap_result_plot_disp_run/raw_signal'
            prefix_name = 'sensor_npz'
    else:
        if args.zo_flag:
            base_dir = 'ood_L_n_n_heatmap_result_plot_disp_run/zero_offset'
            prefix_name = 'sensor_npz_zo'
        else:
            base_dir = 'ood_L_n_n_heatmap_result_plot_disp_run/raw_signal'
            prefix_name = 'sensor_npz'
        
    json_data_dic = {}
    for metric in metric_str_lst:
        json_data_dic[metric] = {}
    
    if mode == 'Ln' or mode == 'Lcnn':
        if args.line_flag==False:
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
                    x_label = []
                    y_lst = []
                    for metric in metric_str_lst:
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
                        print('metric:', metric)
                        json_str = '{}-{}-{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx], detector)
                        ood_value = round(df.to_numpy().mean(), 2)
                        json_data_dic[metric][json_str] = ood_value
                        
                        if args.line_flag == False:
                            ax = sns.heatmap(df, annot=True, vmin = 0, vmax = 1.0)
                            if title_flag:
                                plt.title('Heatmap_{}_{}_{}_{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx], detector, metric), fontsize=10)
                            
                            if not os.path.exists(os.path.join(base_dir, metric)):
                                os.makedirs(os.path.join(base_dir, metric))

                            png_file_name = os.path.join(os.path.join(base_dir, metric), '{}_{}_{}_{}_{}_heatmap_plot.png'.format(base_dir_name, net, loss, detector, metric))
                            print(png_file_name)

                            if os.path.isfile(png_file_name):
                                os.remove(png_file_name)
                            plt.savefig(png_file_name)
                            # plt.show()
                            plt.clf()
                            # exit()
                            
                        if args.line_flag:
                            x_label.append(metric)
                            y_lst.append(ood_value)
                            
                    if args.line_flag:
                        plt.plot(x_label, y_lst, marker='o', linewidth=args.linewidth)
                        
                if args.line_flag:
                    if title_flag:
                        plt.title('{}_{}_OOD_Result'.format(net, loss))
                    plt.xlabel('OOD Metric', weight='bold')
                    plt.legend(detector_lst, framealpha=0.2)
                    
                    png_file_name = '{}_{}_{}_plot.png'.format(base_dir_name, net, loss)
                    if os.path.isfile(os.path.join(base_dir, png_file_name)):
                        os.remove(os.path.join(base_dir, png_file_name))
                    plt.savefig(os.path.join(base_dir, png_file_name))
                    # plt.show()
                    plt.clf()
                    # exit()
                        
        if args.line_flag == False:
            for metric in metric_str_lst:
                print(json_data_dic[metric])
                with open(os.path.join(os.path.join(base_dir, metric), m_str + '_ood_metric.json'), 'w') as handle:
                    json.dump(json_data_dic[metric], handle, indent=4, sort_keys=False)

    elif mode == 'Lnn':
        net_lst = ['brnn_att', 'sp_bi_dir']
        net_disp_lst = ['brnn_att', 'bi_tcn']
        print('zo_flag:', args.zo_flag)

        for net_idx, net in enumerate(net_lst):
            for loss_idx, loss in enumerate(loss_lst):
                for detector in detector_lst:
                    x_label = []
                    y_lst = []
                    for metric in metric_str_lst:
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
                        print('metric:', metric)
                        json_str = '{}-{}-{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx], detector)
                        ood_value = round(df.to_numpy().mean(), 2)
                        json_data_dic[metric][json_str] = ood_value
                            
                        if args.line_flag == False:
                            ax = sns.heatmap(df, annot=True, vmin = 0, vmax = 1.0)
                            if title_flag:
                                plt.title('Heatmap_{}_{}_{}_{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx], detector, metric), fontsize=10)
                            
                            if not os.path.exists(os.path.join(base_dir, metric)):
                                os.makedirs(os.path.join(base_dir, metric))

                            png_file_name = os.path.join(os.path.join(base_dir, metric), '{}_{}_{}_{}_{}_heatmap_plot.png'.format(base_dir_name, net, loss, detector, metric))
                            print(png_file_name)

                            if os.path.isfile(png_file_name):
                                os.remove(png_file_name)
                            plt.savefig(png_file_name)
                            # plt.show()
                            plt.clf()
                            # exit()
                        
                        if args.line_flag:
                            x_label.append(metric)
                            y_lst.append(ood_value)
                            
                    if args.line_flag:
                        plt.plot(x_label, y_lst, marker='o', linewidth=args.linewidth)
                        
                if args.line_flag:
                    if title_flag:
                        plt.title('{}_{}_OOD_Result'.format(net, loss))
                    plt.xlabel('OOD Metric', weight='bold')
                    plt.legend(detector_lst, framealpha=0.2)
                    
                    png_file_name = '{}_{}_{}_plot.png'.format(base_dir_name, net, loss)
                    if os.path.isfile(os.path.join(base_dir, png_file_name)):
                        os.remove(os.path.join(base_dir, png_file_name))
                    plt.savefig(os.path.join(base_dir, png_file_name))
                    # plt.show()
                    plt.clf()
                    # exit()
        
        if args.line_flag == False:
            # print(json_data_dic)
            for metric in metric_str_lst:
                print(json_data_dic[metric])
                with open(os.path.join(os.path.join(base_dir, metric), 'ood_metric.json'), 'w') as handle:
                    json.dump(json_data_dic[metric], handle, indent=4, sort_keys=False)


