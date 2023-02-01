import os
import argparse
import numpy as np
import natsort
import json

# mode raw, zo, dm
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--mode', type=str, default='dm', help='mode')

    args = parser.parse_args()

    mode = args.mode.strip()
    
    if args.mode == 'raw':
        base_dir = 'sensor_npz'
        dst_path = 'sensor_info'
    elif args.mode == 'zo':
        base_dir = 'sensor_npz_zo'
        dst_path = 'sensor_info_zo'
    elif args.mode == 'dm':
        base_dir = 'sensor_npz'
        dst_path = 'sensor_info_dm'
    
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    file_lst = os.listdir(base_dir)
    file_lst = natsort.natsorted(file_lst)
    
    for file in file_lst:
        data_path = os.path.join(base_dir, file)

        print(data_path)
        
        npz_x_file_name = os.path.join(data_path, 'sensor_x.npy')
        X_data = np.load(npz_x_file_name)
        type_len = len(X_data.shape)
        
        if type_len == 3:
            json_dict = {}
            json_dict['sensor_data_shape'] = X_data.shape
            save_json_file_name = '{}_info.json'.format(file)

            if args.mode == 'dm':
                start = 0
                end = 20
                tmp_data = X_data[:, start:end, :]
                f_mean = tmp_data.mean(axis=(0,1)).tolist()
                # print('f_mean', f_mean)
                X_data = X_data - X_data[19] + f_mean

                if False:
                    start = 0
                    end = 20
                    tmp_data = X_data[:, start:end, :]
                    print('mean 0-20:', tmp_data.mean(axis=(0,1)).tolist())
                    print('std 0-20:', tmp_data.std(axis=(0,1)).tolist())

                    start = 0
                    end = 250
                    tmp_data = X_data[:, start:end, :]
                    print('mean 0-250:', tmp_data.mean(axis=(0,1)).tolist())
                    print('std 0-250:', tmp_data.std(axis=(0,1)).tolist())
                    exit()
                
            start = 0
            end = 250
            tmp_data = X_data[:, start:end, :]
            json_dict['sensor_data_mean_{}_{}'.format(start, end)] = tmp_data.mean(axis=(0,1)).tolist()
            json_dict['sensor_data_std_{}_{}'.format(start, end)] = tmp_data.std(axis=(0,1)).tolist()

            start = 0
            end = 20
            tmp_data = X_data[:, start:end, :]
            json_dict['sensor_data_mean_{}_{}'.format(start, end)] = tmp_data.mean(axis=(0,1)).tolist()
            json_dict['sensor_data_std_{}_{}'.format(start, end)] = tmp_data.std(axis=(0,1)).tolist()

            with open(os.path.join(dst_path, save_json_file_name), "w") as json_file:
                json.dump(json_dict, json_file, indent=4)

    print('finished')
