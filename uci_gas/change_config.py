import os
import argparse
import json
import copy
from distutils.util import strtobool

os.environ['KMP_DUPLICATE_LIB_OK']='True'

Three_Class_flag = False

rnn_model_lst = [ 'brnn_att', 'sp_bi_dir' ]
cnn_model_lst = [ 'crnn', 'cnn' ]

# n_loss_lst = [ 'NllLoss', 'FocalLoss', 'arcface', 'cosface', 'sphereface', 'CenterLoss' ]

mode_lst = [ 'L_n_n', 'L_n', 'L_cnn', 'L_all', 'L_all_cnn' ]
# mode_lst = [ 'L_all', 'L_all_cnn' ]
# mode_lst = [ 'L_n_n', 'L_n', 'L_cnn' ]
# mode_lst = [ 'L_cnn' ]
# zo_flag

if Three_Class_flag:
    Width = 5
else:
    Width = 6
Height = 9

def save_json(config, save_path, save_cfg_name):
    config_save_path = os.path.join(save_path, save_cfg_name)
    with open(config_save_path, 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=False)

def pre_func(config, model, name):
    config['method'] = model
    if Three_Class_flag:
        if args.zo_flag:
            b_name = 'sensor_3class_8f_npz_zo'
            # b_name = 'sensor_3class_npz_zo'
        else:
            b_name = 'sensor_3class_8f_npz'
            # b_name = 'sensor_3class_npz'
    else:
        if args.zo_flag:
            b_name = 'sensor_npz_zo'
        else:
            b_name = 'sensor_npz'
    b_path = os.path.join(args.path, b_name)
    if args.zo_flag:
        save_path = os.path.join(args.tpath, name + 'zo_config')
    else:
        save_path = os.path.join(args.tpath, name + 'config')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    return b_path, b_path, b_name, save_path

def L_n_n_Func(args, config, model, loss):
    b_path, b_path, b_name, save_path = pre_func(config, model, 'L_n_n_')
    
    for h in range(1, Height+1):
        for w in range(1, Width+1):
            p_name = 'sensor_L{}_{}'.format(w, h)
            p_all_name = os.path.join(b_path, p_name)

            config['data']['path'] = p_all_name
            config['data']['name'] = p_name

            save_cfg_name = '{}-L{}_{}-{}-{}.json'.format(b_name, w, h, model, loss)
            save_json(config, save_path, save_cfg_name)
            
def L_n_Func(args, config, model, loss):
    b_path, b_path, b_name, save_path = pre_func(config, model, 'L_n_')
    
    for w in range(1, Width+1):
        p_name = 'sensor_L{}_rnn'.format(w)
        p_all_name = os.path.join(b_path, p_name)

        config['data']['path'] = p_all_name
        config['data']['name'] = p_name

        save_cfg_name = '{}-L{}_rnn-{}-{}.json'.format(b_name, w, model, loss)
        save_json(config, save_path, save_cfg_name)

def L_cnn_Func(args, config, model, loss):
    b_path, b_path, b_name, save_path = pre_func(config, model, 'L_cnn_')
    
    for w in range(1, Width+1):
        p_name = 'sensor_L{}_cnn'.format(w)
        p_all_name = os.path.join(b_path, p_name)

        config['data']['path'] = p_all_name
        config['data']['name'] = p_name

        save_cfg_name = '{}-L{}_cnn-{}-{}.json'.format(b_name, w, model, loss)
        save_json(config, save_path, save_cfg_name)

def L_all_Func(args, config, model, loss):
    b_path, b_path, b_name, save_path = pre_func(config, model, 'L_all_')
    
    p_name = 'sensor_all_rnn'
    p_all_name = os.path.join(b_path, p_name)

    config['data']['path'] = p_all_name
    config['data']['name'] = p_name

    save_cfg_name = '{}-{}-{}.json'.format(b_name, model, loss)

    save_json(config, save_path, save_cfg_name)

def L_all_cnn_Func(args, config, model, loss):
    b_path, b_path, b_name, save_path = pre_func(config, model, 'L_all_cnn_')
    
    p_name = 'sensor_all_cnn'
    p_all_name = os.path.join(b_path, p_name)

    config['data']['path'] = p_all_name
    config['data']['name'] = p_name

    save_cfg_name = '{}-{}-{}.json'.format(b_name, model, loss)

    save_json(config, save_path, save_cfg_name)

def set_config_loss_func(config):
    config['train']['loss'] = loss
    if loss=='NllLoss' or loss=='FocalLoss':
        # config['parallel'] = True
        config['parallel'] = False
    else:
        config['parallel'] = False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequence Modeling')
    parser.add_argument('--config', default='config/gas_sensor_rnn_config.json', type=str, help='config file path')
    parser.add_argument('--path', default='../dataset', type=str, help='path')
    parser.add_argument('--mode', default='L_n_n', type=str, help='num_mels (default: 64)')
    parser.add_argument('--batch', default=1024, type=int, help='batch (default: 512)')
    parser.add_argument('--epochs', default=50, type=int, help='epochs (default: 50)')
    parser.add_argument('--start', default=140, type=int, help='epochs (default: 140)')
    parser.add_argument('--end', default=200, type=int, help='epochs (default: 200)')
    parser.add_argument('--length', default=20, type=int, help='epochs (default: 20)')
    parser.add_argument('--tpath', default='../auto_config', type=str, help='target path')
    parser.add_argument('--zo_flag', default=False, type=lambda x:bool(strtobool(x.rstrip())), help='zo_flag')

    args = parser.parse_args()
    
    config = json.load(open(args.config))

    config['data']['loader']['batch_size'] = args.batch
    config['train']['epochs'] = args.epochs
    config['data']['start'] = args.start
    config['data']['end'] = args.end
    config['data']['length'] = args.length
    if Three_Class_flag:
        config['class_num'] = 3
        config['data']['feature_num'] = 8
        # config['data']['feature_num'] = 6
        config['train']['NllLoss_args']['weight'] = [1.0, 1.0, 1.0]
        config['train']['FocalLoss_args']['weight'] = [1.0, 1.0, 1.0]
        config['train']['LabelSmoothingLoss_args']['weight'] = [1.0, 1.0, 1.0]
    
    print('=================')
    
    for mode in mode_lst:
        if args.zo_flag:
            config['train']["save_dir"] = '../saved_cv/'+mode+'_zo_result'
        else:
            config['train']["save_dir"] = '../saved_cv/'+mode+'_result'
            
        if mode == 'L_n_n':
            n_loss_lst = [ 'NllLoss', 'cosface' ]
            for loss in n_loss_lst:
                set_config_loss_func(config)
                for model in rnn_model_lst:
                    L_n_n_Func(args, config, model, loss)
        elif mode == 'L_n':
            n_loss_lst = [ 'NllLoss', 'cosface' ]
            for loss in n_loss_lst:
                set_config_loss_func(config)
                for model in rnn_model_lst:
                    L_n_Func(args, config, model, loss)
        elif mode == 'L_cnn':
            n_loss_lst = [ 'NllLoss', 'cosface' ]
            for loss in n_loss_lst:
                set_config_loss_func(config)
                for model in cnn_model_lst:
                    L_cnn_Func(args, config, model, loss)
        elif mode == 'L_all':
            n_loss_lst = [ 'NllLoss', 'FocalLoss', 'arcface', 'cosface', 'sphereface' ]
            for loss in n_loss_lst:
                set_config_loss_func(config)
                for model in rnn_model_lst:
                    L_all_Func(args, config, model, loss)
        elif mode == 'L_all_cnn':
            n_loss_lst = [ 'NllLoss', 'FocalLoss', 'arcface', 'cosface', 'sphereface' ]
            for loss in n_loss_lst:
                set_config_loss_func(config)
                for model in cnn_model_lst:
                    L_all_cnn_Func(args, config, model, loss)
