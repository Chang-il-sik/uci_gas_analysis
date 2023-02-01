#!/bin/bash
count=0
for entry in `ls ../auto_config/L_cnn_zo_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_cnn_zo_config/$entry
done
count=0
for entry in `ls ../auto_config/L_n_n_zo_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_n_n_zo_config/$entry
done
python model_best_L.py --path ../saved_cv/L_cnn_zo_result --mode Ln
python model_best_L.py --path ../saved_cv/L_n_n_zo_result --mode Lnn
