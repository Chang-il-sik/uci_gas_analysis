# uci_gas_analysis
UCI Gas sensor arrays in open sampling settings Data Set Analysis

## Dataset preparation
http://archive.ics.uci.edu/ml/datasets/gas+sensor+arrays+in+open+sampling+settings   
Dataset is created as an npy file in 1 second increments without considering heater and fan speed

Capacity when created with sensor_npz.tar.gz: 15G   
Capacity when created with sensor_npz_zo.tar.gz: 15G   

You need to create a dataset and put it in uci_gas_analysis/dataset/sensor_npz and uci_gas_analysis/dataset/sensor_npz_zo.   

## Setup
    git clone https://github.com/Chang-il-sik/uci_gas_analysis.git
    cd uci_gas_analysis/uci_gas

## Create config file for model training
    sh script/change_config.sh

config file is created in uci_gas_analysis/auto_config folder.

## Model train
    sh script/L_all_cnn_run.sh
    sh script/L_all_cnn_zo_run.sh
    sh script/L_all_run.sh
    sh script/L_all_zo_run.sh
    sh script/L_cnn_run.sh
    sh script/L_n_n_run.sh
    sh script/L_n_n_zo_run.sh
    sh script/L_n_run.sh
    sh script/L_n_zo_run.sh

L_all: Time series data by collecting all data.   
L_all_cnn : Collect all data and use 2D CNN (Convolutional Neural Network) method.   
L_cnn: Uses the 2-dimensional CNN (Convolutional Neural Network) method for L1 ~ L6.   
L_n: time series data by location for L1 ~ L6.   
L_n_n: Time series data by 2-dimensional spatial location.   
zo : stands for zero-offset subtraction.   

The trained result is saved in the uci_gas_analysis/saved_cv folder.

## The maximum result among the trained models is saved as a csv file.
    sh script/model_best_L_all_run.sh
    sh script/model_best_L_n_n_run.sh
    sh script/model_best_L_cnn_run.sh
    sh script/model_best_L_n_run.sh

## Analysis through OOD(Out Of Distribution) method
    sh script/ood_L_all_cnn_run.sh
    sh script/ood_L_all_cnn_zo_run.sh
    sh script/ood_L_all_run.sh
    sh script/ood_L_all_zo_run.sh 
    sh script/ood_L_cnn_run.sh
    sh script/ood_L_n_n_run.sh
    sh script/ood_L_n_n_zo_run.sh
    sh script/ood_L_n_run.sh
    sh script/ood_L_n_zo_run.sh

## hange folder location
    cd uci_gas_analysis/saved_cv/result_plot_disp

## Plot the results for the the trained model
    sh script/L_all_result_plot_disp_run.sh
    sh script/L_n_n_result_plot_disp_run.sh
    sh script/L_n_n_heatmap_result_plot_disp_run.sh
    sh script/L_n_result_plot_disp_run.sh
    sh script/L_n_heatmap_result_plot_disp_run.sh

## Plot the results for the OOD(Out Of Distribution) method
    sh script/ood_L_all_result_plot_disp_run.sh
    sh script/ood_L_n_n_heatmap_result_plot_disp_run.sh
    sh script/ood_L_n_heatmap_result_plot_disp_run.sh
    sh script/ood_L_n_n_time_result_plot_disp_run.sh
    sh script/ood_L_n_time_result_plot_disp_run.sh
    sh script/ood_L_n_n_multi_time_result_plot_disp_run.sh
    sh script/ood_L_n_multi_time_result_plot_disp_run.sh
