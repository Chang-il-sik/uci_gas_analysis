import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import numpy as np
from matplotlib import rc, rcParams

rc('font', weight='bold')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--path', type=str, default='./', help='path')
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

    net_disp_lst = []
    for idx, net in enumerate(net_lst):
        if net == 'sp_bi_dir':
            net_disp_lst.append('bi_tcn')
        else:
            net_disp_lst.append(net)

    # plt.figure(figsize=(10,8))
    
    for net in net_lst:
        x_label = []
        y_lst = []

        for idx, loss in enumerate(loss_lst):
            x_label.append(loss_label_lst[idx])
            ua = cs_drop_df[(cs_drop_df['Loss'] == loss) & (cs_drop_df['Network'] == net)]['UA']
            y_lst.append(list(ua)[0])
            # print(type(ua))
            # print(list(ua)[0])
            # print(ua[0])
    
        # print(type(y_lst[0]))
        # print(net_disp_lst)
        
        plt.plot(x_label, y_lst, marker='o', linewidth=args.linewidth)

    plt.xlabel('Loss Function', weight='bold')
    plt.ylabel('Accuracy', weight='bold')
    plt.legend(net_disp_lst, framealpha=0.2)
    # plt.ylim(0.4, 1.0)
    png_file_name = '{}_plot.png'.format(base_dir_name)
    if os.path.isfile(png_file_name):
        os.remove(png_file_name)
    plt.savefig(png_file_name)
    # plt.show()
    plt.clf()
