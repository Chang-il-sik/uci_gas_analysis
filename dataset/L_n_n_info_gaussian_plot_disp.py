import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import json
from distutils.util import strtobool
from scipy.stats import norm
from matplotlib import rc, rcParams

rc('font', weight='bold')
title_flag = False

Width = 6
Height = 9

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--path', type=str, default='./sensor_info', help='path')
    parser.add_argument('--zo_flag', default=False, type=lambda x:bool(strtobool(x.rstrip())), help='zo_flag')
    parser.add_argument('--start', default=0, type=int, help='start')
    parser.add_argument('--end', default=20, type=int, help='end')
    parser.add_argument('--linewidth', default=3, type=int, help='linewidth')
    args = parser.parse_args()

    class_name_lst = ['Acetaldehyde_500', 'Acetone_2500', 'Ammonia_10000', 'Benzene_200', 'Ethylene_500', 'Methane_1000', 'Methanol_200', 'Toluene_200']
    info_lst = ['m']
    # info_lst = ['m', 'v', 's', 'c']
    time_length = 250
    x = range(time_length)

    result_path = 'gaussian_plot'
    if not os.path.exists(os.path.join(args.path, result_path)):
        os.makedirs(os.path.join(args.path, result_path))
            
    for h in range(1, Height+1):
        for w in range(1, Width+1):
            json_file = 'sensor_L{}_{}_info.json'.format(w, h)
            with open(os.path.join(args.path, json_file)) as f:
                json_object = json.load(f)
                f_mean = json_object['sensor_data_mean_{}_{}'.format(args.start, args.end)]
                f_std = json_object['sensor_data_std_{}_{}'.format(args.start, args.end)]

                y_lst = []
                for f_idx in range(8):
                    x = np.arange(f_mean[f_idx]-5, f_mean[f_idx]+5, 0.01)
                    plt.plot(x, norm(f_mean[f_idx], f_std[f_idx]).pdf(x), linewidth=args.linewidth)
                    y_lst.append('m:{},s:{},sensor_{}'.format(round(f_mean[f_idx],2), round(f_std[f_idx], 2), f_idx+1))

                if title_flag:
                    plt.title('L{}_{}_{}s-{}s_feature_pdf'.format(w, h, args.start, args.end))
                plt.legend(y_lst, framealpha=0.2)
                
                png_file_name = '{}/L{}_{}_{}s-{}s_feature_pdf.png'.format(result_path, w, h, args.start, args.end)
                if os.path.isfile(os.path.join(args.path, png_file_name)):
                    os.remove(os.path.join(args.path, png_file_name))
                plt.savefig(os.path.join(args.path, png_file_name))
                # plt.show()
                plt.clf()
                    
