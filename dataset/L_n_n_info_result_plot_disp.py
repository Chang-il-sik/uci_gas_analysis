import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import numpy as np
import seaborn as sns
import json
from distutils.util import strtobool
from matplotlib import rc, rcParams

rc('font', weight='bold')
title_flag = False

Width = 6
Height = 9

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--path', type=str, default='./sensor_info', help='path')
    parser.add_argument('--zo_flag', default=False, type=lambda x:bool(strtobool(x.rstrip())), help='zo_flag')
    parser.add_argument('--ylim_flag', default=False, type=lambda x:bool(strtobool(x.rstrip())), help='ylim_flag')
    parser.add_argument('--normalize_flag', default=False, type=lambda x:bool(strtobool(x.rstrip())), help='ylim_flag')
    parser.add_argument('--linewidth', default=3, type=int, help='linewidth')
    args = parser.parse_args()

    class_name_lst = ['Acetaldehyde_500', 'Acetone_2500', 'Ammonia_10000', 'Benzene_200', 'Ethylene_500', 'Methane_1000', 'Methanol_200', 'Toluene_200']
    info_lst = ['m']
    # info_lst = ['m', 'v', 's', 'c']
    time_length = 250
    x = range(time_length)
    
    result_path = 'plot'
    if not os.path.exists(os.path.join(args.path, result_path)):
        os.makedirs(os.path.join(args.path, result_path))

    for h in range(1, Height+1):
        for w in range(1, Width+1):

            if args.normalize_flag:
                json_file = 'sensor_L{}_{}_info.json'.format(w, h)
                with open(os.path.join(args.path, json_file)) as f:
                    json_object = json.load(f)
                    f_mean = json_object['sensor_data_mean_0_250']
                    f_std = json_object['sensor_data_std_0_250']

            if args.ylim_flag:
                # calcaulate min max
                min_dic = {}
                max_dic = {}
                for info in info_lst:
                    min_dic[info] = 10000.0
                    max_dic[info] = -100000.0

                for f in range(1, 9):
                    for info in info_lst:
                        info_name = '{}_f{}'.format(info, f)
                        for c_idx, c_name in enumerate(class_name_lst):
                            if args.zo_flag:
                                csv_name = '{}_sensor_L{}_{}_info_zo_all.csv'.format(c_name, w, h)
                            else:
                                csv_name = '{}_sensor_L{}_{}_info_all.csv'.format(c_name, w, h)
                            df = pd.read_csv(os.path.join(args.path, csv_name))
                            df = df.dropna(axis=0)                

                            sel_info = list(df[info_name])
                            sel_info = np.array(sel_info)
                            if args.normalize_flag:
                                sel_info = (sel_info - f_mean[f-1])/f_std[f-1]
                            
                            if min_dic[info] > min(sel_info):
                                min_dic[info] = min(sel_info)
                            if max_dic[info] < max(sel_info):
                                max_dic[info] = max(sel_info)
                    
            for f in range(1, 9):
                for info in info_lst:
                    legend_lst = []
                    info_name = '{}_f{}'.format(info, f)
                    for c_idx, c_name in enumerate(class_name_lst):
                        if args.zo_flag:
                            csv_name = '{}_sensor_L{}_{}_info_zo_all.csv'.format(c_name, w, h)
                        else:
                            csv_name = '{}_sensor_L{}_{}_info_all.csv'.format(c_name, w, h)
                        print(csv_name)
                        df = pd.read_csv(os.path.join(args.path, csv_name))
                        df = df.dropna(axis=0)                

                        sel_info = list(df[info_name])
                        sel_info = np.array(sel_info)
                        legend_lst.append('{}'.format(c_name))
                        if args.normalize_flag:
                            sel_info = (sel_info - f_mean[f-1])/f_std[f-1]
                        plt.plot(x, sel_info, linewidth=args.linewidth)

                    if title_flag:
                        plt.title('L{}_{}_{}'.format(w, h, info_name))
                    plt.legend(legend_lst, loc='upper right', framealpha=0.2)
                    if args.ylim_flag:
                        plt.ylim(min_dic[info], max_dic[info])
                    
                    if args.zo_flag:
                        if args.ylim_flag:
                            if args.normalize_flag:
                                png_file_name = '{}/ylim_normalize_L{}_{}_{}_zo.png'.format(result_path, w, h, info_name)
                            else:
                                png_file_name = '{}/ylim_L{}_{}_{}_zo.png'.format(result_path, w, h, info_name)
                        else:
                            if args.normalize_flag:
                                png_file_name = '{}/normalize_L{}_{}_{}_zo.png'.format(result_path, w, h, info_name)
                            else:
                                png_file_name = '{}/raw_L{}_{}_{}_zo.png'.format(result_path, w, h, info_name)
                    else:
                        if args.ylim_flag:
                            if args.normalize_flag:
                                png_file_name = '{}/ylim_normalize_L{}_{}_{}.png'.format(result_path, w, h, info_name)
                            else:
                                png_file_name = '{}/ylim_L{}_{}_{}.png'.format(result_path, w, h, info_name)
                        else:
                            if args.normalize_flag:
                                png_file_name = '{}/normalize_L{}_{}_{}.png'.format(result_path, w, h, info_name)
                            else:
                                png_file_name = '{}/raw_L{}_{}_{}.png'.format(result_path, w, h, info_name)
                        
                    if os.path.isfile(os.path.join(args.path, png_file_name)):
                        os.remove(os.path.join(args.path, png_file_name))
                    plt.savefig(os.path.join(args.path, png_file_name))
                    # plt.show()
                    plt.clf()
                    # exit()
