python ood_L_n_heatmap_result_plot_disp.py --zo_flag false --line_flag false --path ../L_n_result --mode Ln
python ood_L_n_heatmap_result_plot_disp.py --zo_flag true --line_flag false --path ../L_n_zo_result --mode Ln
python ood_L_n_heatmap_result_plot_disp.py --zo_flag false --line_flag false --path ../L_cnn_result/ --mode Lcnn
python ood_L_n_heatmap_result_plot_disp.py --zo_flag true --line_flag false --path ../L_cnn_zo_result/ --mode Lcnn

python ood_L_n_heatmap_result_plot_disp.py --zo_flag false --line_flag true --path ../L_n_result --mode Ln
python ood_L_n_heatmap_result_plot_disp.py --zo_flag true --line_flag true --path ../L_n_zo_result --mode Ln
python ood_L_n_heatmap_result_plot_disp.py --zo_flag false --line_flag true --path ../L_cnn_result/ --mode Lcnn
python ood_L_n_heatmap_result_plot_disp.py --zo_flag true --line_flag true --path ../L_cnn_zo_result/ --mode Lcnn
