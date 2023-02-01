#!/bin/bash
# zo
count=0
for entry in `ls ../saved_cv/L_all_zo_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_all_zo_result/$entry
done
