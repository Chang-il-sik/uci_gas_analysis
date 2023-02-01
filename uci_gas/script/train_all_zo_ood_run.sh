count=0
for entry in `ls ../saved_cv/L_all_cnn_zo_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_all_cnn_zo_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_cnn_zo_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_cnn_zo_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_n_zo_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_n_zo_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_n_n_zo_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_n_n_zo_result/$entry
done
count=0
for entry in `ls ../saved_cv/L_all_zo_result`; do
    count=$(($count+1))
    echo $entry
    # echo $count
    python ood_data_loader.py --resume_path ../saved_cv/L_all_zo_result/$entry
done
