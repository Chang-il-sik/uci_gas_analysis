import os
import argparse
import pandas as pd
import natsort
import numpy as np
from distutils.util import strtobool

Height = 9
Width = 6

def get_max_idx(path, file):
    result_file = os.path.join(path, file, 'result.csv')
    df = pd.read_csv(result_file)
    argmax = df['UA'].argmax()
    df_idx = df.iloc[argmax]
    return df_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('-p', '--path', default=None, type=str, help='directory path (default: None)')
    parser.add_argument('--mode', default='all', type=str, help='all')

    args = parser.parse_args()

    file_lst = os.listdir(args.path)
    file_lst = natsort.natsorted(file_lst)

    save_list = []
    mode = args.mode.strip()
    
    if mode=='all':
        for file in file_lst:
            dir_name = os.path.join(args.path, file)
            if os.path.isdir(dir_name):
                f_info = file.split('-')
                sensor_name = f_info[0]
                model_name = f_info[1]
                loss_name = f_info[2]

                df_idx = get_max_idx(args.path, file)

                save_list.append( { 'Network' : model_name, 
                                    'Loss' : loss_name,
                                    'WA' : round(df_idx['WA'], 3),
                                    'UA' : round(df_idx['UA'], 3),
                                    'TA' : round(df_idx['TA'], 3),
                                    'epoch' : df_idx['epoch'] } )
    elif mode=='Ln':
        for ln in range(1, Width+1):
            find_str = '-L{}_'.format(ln)
            matching_lst = [s for s in file_lst if find_str in s]

            for file in matching_lst:
                dir_name = os.path.join(args.path, file)
                if os.path.isdir(dir_name):
                    f_info = file.split('-')
                    sensor_name = f_info[0]
                    model_name = f_info[2]
                    loss_name = f_info[3]

                    df_idx = get_max_idx(args.path, file)

                    save_list.append( { 'Location' : 'L{}'.format(ln),
                                        'Network' : model_name,
                                        'Loss' : loss_name,
                                        'WA' : round(df_idx['WA'], 3),
                                        'UA' : round(df_idx['UA'], 3),
                                        'TA' : round(df_idx['TA'], 3),
                                        'epoch' : df_idx['epoch'] } )
    elif mode=='Lnn':
        for h in range(1, Height+1):
            for w in range(1, Width+1):
                find_str = '-L{}_{}'.format(w, h)
                matching_lst = [s for s in file_lst if find_str in s]

                for file in matching_lst:
                    dir_name = os.path.join(args.path, file)
                    if os.path.isdir(dir_name):
                        f_info = file.split('-')
                        sensor_name = f_info[0]
                        model_name = f_info[2]
                        loss_name = f_info[3]

                        df_idx = get_max_idx(args.path, file)

                        save_list.append( { 'Location' : 'L{}_{}'.format(w, h),
                                            'Network' : model_name,
                                            'Loss' : loss_name,
                                            'WA' : round(df_idx['WA'], 3),
                                            'UA' : round(df_idx['UA'], 3),
                                            'TA' : round(df_idx['TA'], 3),
                                            'epoch' : df_idx['epoch'] } )

    save_df = pd.DataFrame(save_list)

    save_df.to_csv(os.path.join(args.path, 'best_result.csv'), index = False)
    save_df.to_excel(os.path.join(args.path, 'best_result.xlsx'), index = False)            
