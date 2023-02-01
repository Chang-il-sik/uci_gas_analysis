import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import numpy as np
from matplotlib import rc, rcParams

rc('font', weight='bold')

Width = 6
Height = 9

loss_lst = ['NllLoss', 'cosface']
loss_label_lst = ['SoftmaxLoss', 'cosface' ]

def result_plot_disp(mode, sel_drop_df, w, h, linewidth):
    net_lst = sel_drop_df['Network'].unique()
    # loss_lst = sel_drop_df['Loss'].unique()
    
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
            ua = sel_drop_df[(sel_drop_df['Loss'] == loss) & (sel_drop_df['Network'] == net)]['UA']
            y_lst.append(list(ua)[0])
            # print(type(ua))
            # print(list(ua)[0])
            # print(ua[0])
    
        # print(type(y_lst[0]))
        # print(net_disp_lst)
        
        plt.plot(x_label, y_lst, marker='o', linewidth=linewidth)

    plt.xlabel('Loss Function', weight='bold')
    plt.ylabel('Accuracy', weight='bold')
    plt.legend(net_disp_lst, framealpha=0.2)
    # plt.ylim(0.4, 1.0)
    if mode == 'Ln':
        png_file_name = '{}_L{}_plot.png'.format(base_dir_name, w)
    elif mode == 'Lnn':
        png_file_name = '{}_L{}_{}_plot.png'.format(base_dir_name, w, h)
        
    if os.path.isfile(png_file_name):
        os.remove(png_file_name)
    plt.savefig(png_file_name)
    # plt.show()
    plt.clf()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--path', type=str, default='./', help='path')
    parser.add_argument('--mode', type=str, default='Ln', help='mode')
    parser.add_argument('--linewidth', type=int, default=3, help='linewidth')
    args = parser.parse_args()

    base_dir_name = str(Path(args.path)).split('/')[-1]
    cs_file_name = os.path.join(args.path, 'best_result.csv')

    cs_df = pd.read_csv(cs_file_name)
    cs_drop_df = cs_df.dropna(axis=0)

    mode = args.mode.strip()
    
    if mode == 'Ln':
        for w in range(1, Width+1):
            sel_drop_df = cs_drop_df[cs_drop_df['Location'] == 'L{}'.format(w)]
            result_plot_disp(mode, sel_drop_df, w, 0, args.linewidth)
    elif mode == 'Lnn':
        for w in range(1, Width+1):
            for h in range(1, Height+1):
                sel_drop_df = cs_drop_df[cs_drop_df['Location'] == 'L{}_{}'.format(w, h)]
                result_plot_disp(mode, sel_drop_df, w, h, args.linewidth)


