import os
import argparse
import numpy as np
import natsort
from sklearn.model_selection import train_test_split

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

            npz_x_file_name = os.path.join(data_path, 'sensor_x.npy')
            npz_y_file_name = os.path.join(data_path, 'sensor_y.npy')

            X_data = np.load(npz_x_file_name)
            Y_data = np.load(npz_y_file_name)
            Y_data = np.array(Y_data, dtype=int)

            X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data,
                                                                test_size=args.ratio, 
                                                                shuffle=True, 
                                                                random_state=1004)

            npz_train_x_file_name = os.path.join(data_path, 'train_sensor_x.npy')
            npz_train_y_file_name = os.path.join(data_path, 'train_sensor_y.npy')
            npz_test_x_file_name = os.path.join(data_path, 'test_sensor_x.npy')
            npz_test_y_file_name = os.path.join(data_path, 'test_sensor_y.npy')

            np.save(npz_train_x_file_name, X_train)
            np.save(npz_train_y_file_name, Y_train)
            np.save(npz_test_x_file_name, X_test)
            np.save(npz_test_y_file_name, Y_test)

            # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
            print(npz_train_x_file_name)
