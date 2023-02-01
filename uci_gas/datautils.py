import os
import numpy as np

def load_SMR(dataset):
    num_mels = 16
    normalize = False
    
    BASE_DIR = '../dataset'
    
    if normalize:
        npz_file = os.path.join(BASE_DIR, dataset + "_normalize_{}.npz".format(num_mels))
    else:
        npz_file = os.path.join(BASE_DIR, dataset + "_{}.npz".format(num_mels))
    # train_file = os.path.join(BASE_DIR, dataset + "_train_{}.npz".format(num_mels))
    # train_label_file = os.path.join(BASE_DIR, dataset + "_train_label_{}.npz".format(num_mels))
    # test_file = os.path.join(BASE_DIR, dataset + "_test_{}.npz".format(num_mels))
    # test_label_file = os.path.join(BASE_DIR, dataset + "_test_label_{}.npz".format(num_mels))
    # if os.path.isfile(train_file) and os.path.isfile(train_label_file) and os.path.isfile(test_file) and os.path.isfile(test_label_file):
    if os.path.isfile(npz_file):
        data = np.load(npz_file)
        train = data['train']
        train_labels = data['train_labels']
        test = data['test']
        test_labels = data['test_labels']

    return train, train_labels, test, test_labels
