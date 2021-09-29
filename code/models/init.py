#encoding:utf-8
# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

import os
import argparse
import yaml
from .vocab import deserialize_vocab

def parser_options(prefix_path):
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSICD_GaLR.yaml', type=str,
                        help='path to a yaml options file')
    # parser.add_argument('--text_sim_path', default='data/ucm_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
    opt = parser.parse_args()

    # load model options
    with open(os.path.join(prefix_path,opt.path_opt), 'r') as handle:
        options = yaml.load(handle)

    return options

def model_init(prefix_path):
    options = parser_options(prefix_path)

    # choose model
    if options['model']['name'] == "GaLR":
        from .layers import GaLR as models
    else:
        raise NotImplementedError

    # make vocab
    vocab = deserialize_vocab(os.path.join(prefix_path,options['dataset']['vocab_path']))
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    model = models.factory(options['model'],
                           vocab_word,
                           cuda=True,
                           data_parallel=False)
    return model, vocab
