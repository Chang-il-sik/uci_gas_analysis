import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import numpy as np
import seaborn as sns
import json

Width = 6
Height = 9

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--path', type=str, default='./', help='path')
    parser.add_argument('--mode', type=str, default='Ln', help='mode')
    args = parser.parse_args()

    base_dir_name = str(Path(args.path)).split('/')[-1]
    cs_file_name = os.path.join(args.path, 'best_result.csv')

    cs_df = pd.read_csv(cs_file_name)
    cs_drop_df = cs_df.dropna(axis=0)

    mode = args.mode.strip()
    
    if mode == 'Ln' or mode == 'Lcnn':
        plt.rcParams['figure.figsize'] = [10, 2]
        
        if mode == 'Ln':
            net_lst = ['brnn_att', 'sp_bi_dir']
            net_disp_lst = ['brnn_att', 'bi_tcn']
        else:
            net_lst = ['cnn', 'crnn']
            net_disp_lst = ['cnn', 'crnn']
        loss_lst = ['NllLoss', 'cosface']
        loss_label_lst = ['SoftmaxLoss', 'cosface' ]
        json_data = {}
        
        for net_idx, net in enumerate(net_lst):
            for loss_idx, loss in enumerate(loss_lst):
                heatmap_dict = {}
                for w in range(1, Width+1):
                    loc_str = 'L{}'.format(w)
                    sel_drop_df = cs_drop_df[(cs_drop_df['Location'] == loc_str) &
                                            (cs_drop_df['Network'] == net) &
                                            (cs_drop_df['Loss'] == loss)]
                    # net_lst = sel_drop_df['Network'].unique()
                    # loss_lst = sel_drop_df['Loss'].unique()
                    ua = list(sel_drop_df['UA'])[0]
                    # heatmap_dict[loc_str+'({})'.format(ua)] = ua
                    heatmap_dict[loc_str] = ua
                print(heatmap_dict)
                df = pd.DataFrame([heatmap_dict], index=['accuracy'])
                print(df)
                json_str = '{}-{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx])
                json_data[json_str] = round(df.to_numpy().mean(), 2)
                
                ax = sns.heatmap(df, annot=True, vmin = 0, vmax = 1.0)
                plt.title('Heatmap_{}_{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx]), fontsize=10)
                
                png_file_name = '{}_{}_{}_heatmap_plot.png'.format(base_dir_name, net, loss)
                if os.path.isfile(png_file_name):
                    os.remove(png_file_name)
                plt.savefig(png_file_name)
                # plt.show()
                plt.clf()
                
        with open('{}_accuracy.json'.format(base_dir_name), 'w') as handle:
            json.dump(json_data, handle, indent=4, sort_keys=False)

    elif mode == 'Lnn':
        net_lst = ['brnn_att', 'sp_bi_dir']
        net_disp_lst = ['brnn_att', 'bi_tcn']
        loss_lst = ['NllLoss', 'cosface']
        loss_label_lst = ['SoftmaxLoss', 'cosface' ]
        col_str_lst = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        json_data = {}
        
        for net_idx, net in enumerate(net_lst):
            for loss_idx, loss in enumerate(loss_lst):
                heatmap_lst = []
                for h in range(1, Height+1):
                    heatmap_dict = {}
                    for w in range(1, Width+1):
                        loc_str = 'L{}_{}'.format(w, h)
                        sel_drop_df = cs_drop_df[(cs_drop_df['Location'] == loc_str) &
                                                (cs_drop_df['Network'] == net) &
                                                (cs_drop_df['Loss'] == loss)]
                        # net_lst = sel_drop_df['Network'].unique()
                        # loss_lst = sel_drop_df['Loss'].unique()
                        ua = list(sel_drop_df['UA'])[0]
                        heatmap_dict[col_str_lst[w-1]] = ua
                    heatmap_lst.append(heatmap_dict)
                    
                # print(heatmap_lst)
                df = pd.DataFrame(heatmap_lst, index=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9'])
                print(df)
                json_str = '{}-{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx])
                json_data[json_str] = round(df.to_numpy().mean(), 2)
                
                ax = sns.heatmap(df, annot=True, vmin = 0, vmax = 1.0)
                plt.title('Heatmap_{}_{}'.format(net_disp_lst[net_idx], loss_label_lst[loss_idx]), fontsize=10)
                
                png_file_name = '{}_{}_{}_heatmap_plot.png'.format(base_dir_name, net, loss)
                if os.path.isfile(png_file_name):
                    os.remove(png_file_name)
                plt.savefig(png_file_name)
                # plt.show()
                plt.clf()

        with open('{}_accuracy.json'.format(base_dir_name), 'w') as handle:
            json.dump(json_data, handle, indent=4, sort_keys=False)
            

