#!/bin/bash
count=0
for entry in `ls ../auto_config/L_all_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_all_config/$entry
done
count=0
for entry in `ls ../auto_config/L_all_cnn_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_all_cnn_config/$entry
done
count=0
for entry in `ls ../auto_config/L_cnn_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_cnn_config/$entry
done
count=0
for entry in `ls ../auto_config/L_n_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_n_config/$entry
done
count=0
for entry in `ls ../auto_config/L_n_n_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_n_n_config/$entry
done
python model_best_L.py --path ../saved_cv/L_all_result --mode all
python model_best_L.py --path ../saved_cv/L_all_cnn_result --mode all
python model_best_L.py --path ../saved_cv/L_cnn_result --mode Ln
python model_best_L.py --path ../saved_cv/L_n_result --mode Ln
python model_best_L.py --path ../saved_cv/L_n_n_result --mode Lnn
count=0
for entry in `ls ../saved_cv/L_all_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_all_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_all_cnn_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_all_cnn_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_cnn_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_cnn_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_n_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_n_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_n_n_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_n_n_result/$entry
done
