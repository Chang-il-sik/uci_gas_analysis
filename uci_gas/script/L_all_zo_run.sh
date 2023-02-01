#!/bin/bash
# zo
count=0
for entry in `ls ../auto_config/L_all_zo_config`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python uci_gas_project.py train --config ../auto_config/L_all_zo_config/$entry
done
