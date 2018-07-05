import numpy
import os

from main import train

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    train(
    saveto           = './{}.npz'.format(model_name),
    LoadFrom         = './{}.npz_49088'.format(model_name),
    reload_          = False,
    dim_word         = 300,
    dim              = 300,
    patience         = 7,
    n_words          = 100140,
    decay_c          = 0.,
    clip_c           = 10.,
    lrate            = 0.0004,
    optimizer        = 'adam', 
    maxlen           = 450,
    batch_size       = 32,
    valid_batch_size = 32,
    dispFreq         = 100,
    validFreq        = int(392702/32+1),
    saveFreq         = int(392702/32+1),
    use_dropout      = True,
    test          = int(392702/32+1),
    datasets         = ['../../data/binary_tree/premise_multinli_0.9_train.txt',
                        '../../data/binary_tree/hypothesis_multinli_0.9_train.txt',
                        '../../data/binary_tree/label_multinli_0.9_train.txt'],
    valid_datasets   = ['../../data/binary_tree/premise_multinli_0.9_dev_matched.txt',
                        '../../data/binary_tree/hypothesis_multinli_0.9_dev_matched.txt',
                        '../../data/binary_tree/label_multinli_0.9_dev_matched.txt'],
    test_datasets    = ['../../data/binary_tree/premise_multinli_0.9_dev_mismatched.txt',
                        '../../data/binary_tree/hypothesis_multinli_0.9_dev_mismatched.txt',
                        '../../data/binary_tree/label_multinli_0.9_dev_mismatched.txt'],
    test_matched_datasets   = ['../../data/binary_tree/premise_multinli_0.9_test_matched_unlabeled.txt', 
                        '../../data/binary_tree/hypothesis_multinli_0.9_test_matched_unlabeled.txt',
                        '../../data/binary_tree/label_multinli_0.9_test_matched_unlabeled.txt'],
    test_mismatched_datasets    = ['../../data/binary_tree/premise_multinli_0.9_test_mismatched_unlabeled.txt', 
                        '../../data/binary_tree/hypothesis_multinli_0.9_test_mismatched_unlabeled.txt',
                        '../../data/binary_tree/label_multinli_0.9_test_mismatched_unlabeled.txt'],
    dictionary       = '../../data/binary_tree/multinli_vocab_cased.pkl',
    embedding        = '../../data/glove/glove.840B.300d.txt',
    #embedding=None
    )

