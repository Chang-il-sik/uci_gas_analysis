import os
import argparse
import numpy as np
import natsort
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--ratio', default=0.2, type=float, help='ratio')

    args = parser.parse_args()

    base_dir_list = [ 'sensor_npz', 'sensor_npz_zo' ]
    # base_dir_list = [ 'sensor_3class_npz', 'sensor_3class_npz_zo' ]
    
    for base_dir in base_dir_list:
        file_lst = os.listdir(base_dir)
        file_lst = natsort.natsorted(file_lst)
        
        for file in file_lst:
            data_path = os.path.join(base_dir, file)

            npz_x_file_name = os.path.join(data_path, 'train_sensor_x.npy')
            save_json_file_name = os.path.join(data_path, 'sensor_info.json')
            
            X_data = np.load(npz_x_file_name)

            json_dict = {}
            json_dict['sensor_data_shape'] = X_data.shape
            type_len = len(X_data.shape)
            # print(X_data.shape)
            print(save_json_file_name)
            
            if type_len == 3:
                end = 20
                tmp_data = X_data[:, 0:end, :]
                json_dict['sensor_data_mean_{}_{}'.format(0, end)] = tmp_data.mean(axis=(0,1)).tolist()
                json_dict['sensor_data_std_{}_{}'.format(0, end)] = tmp_data.std(axis=(0,1)).tolist()

                end = 250
                for idx in range(0, 151, 50):
                    tmp_data = X_data[:, idx:end, :]
                    # print('mean_{}_{}:'.format(idx, end), tmp_data.mean(axis=(0,1)))
                    # print('std_{}_{}:'.format(idx, end), tmp_data.std(axis=(0,1)))
                    json_dict['sensor_data_mean_{}_{}'.format(idx, end)] = tmp_data.mean(axis=(0,1)).tolist()
                    json_dict['sensor_data_std_{}_{}'.format(idx, end)] = tmp_data.std(axis=(0,1)).tolist()

                end = 200
                for idx in range(0, 161, 20):
                    tmp_data = X_data[:, idx:end, :]
                    # print('mean_{}_{}:'.format(idx, end), tmp_data.mean(axis=(0,1)))
                    # print('std_{}_{}:'.format(idx, end), tmp_data.std(axis=(0,1)))
                    json_dict['sensor_data_mean_{}_{}'.format(idx, end)] = tmp_data.mean(axis=(0,1)).tolist()
                    json_dict['sensor_data_std_{}_{}'.format(idx, end)] = tmp_data.std(axis=(0,1)).tolist()

            elif type_len == 4:
                end = 20
                tmp_data = X_data[:, :, 0:end, :]
                json_dict['sensor_data_mean_{}_{}'.format(0, end)] = tmp_data.mean(axis=(0,1,2)).tolist()
                json_dict['sensor_data_std_{}_{}'.format(0, end)] = tmp_data.std(axis=(0,1,2)).tolist()

                end = 250
                for idx in range(0, 201, 50):
                    tmp_data = X_data[:, :, idx:end, :]
                    # print('mean_{}_{}:'.format(idx, end), tmp_data.mean(axis=(0,1,2)))
                    # print('std_{}_{}:'.format(idx, end), tmp_data.std(axis=(0,1,2)))
                    json_dict['sensor_data_mean_{}_{}'.format(idx, end)] = tmp_data.mean(axis=(0,1,2)).tolist()
                    json_dict['sensor_data_std_{}_{}'.format(idx, end)] = tmp_data.std(axis=(0,1,2)).tolist()

                end = 200
                for idx in range(0, 161, 20):
                    tmp_data = X_data[:, :, idx:end, :]
                    # print('mean_{}_{}:'.format(idx, end), tmp_data.mean(axis=(0,1,2)))
                    # print('std_{}_{}:'.format(idx, end), tmp_data.std(axis=(0,1,2)))
                    json_dict['sensor_data_mean_{}_{}'.format(idx, end)] = tmp_data.mean(axis=(0,1,2)).tolist()
                    json_dict['sensor_data_std_{}_{}'.format(idx, end)] = tmp_data.std(axis=(0,1,2)).tolist()
            
            with open(save_json_file_name, "w") as json_file:
                json.dump(json_dict, json_file, indent=4)
