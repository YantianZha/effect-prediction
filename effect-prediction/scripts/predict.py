# coding: utf-8
# Written by Yantian Zha at Yochan lab, ASU, on April 2016
# Reference: https://github.com/kracwarlock/action-recognition-visual-attention/blob/master/util/data_handler.py

import numpy
import sys
import argparse

import util.gpu_util
board = util.gpu_util.LockGPU()
print 'GPU Lock Acquired'

import theano
import theano.tensor as tensor
theano.config.floatX = 'float32'
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import os
import time

from scipy import optimize, stats
from collections import OrderedDict

import warnings

from util.data_handler import DataHandler
from util.data_handler import TrainProto
from util.data_handler import TestTrainProto
from util.data_handler import TestValidProto
from util.data_handler import TestTestProto

import src.actrec as actrec

if __name__ == '__main__':
    valid_batch_size = 24
    maxlen = 1
    testing_stride = 1
    dataset = 'ucf11'
    fps = 60
    saveto = '/home/yochan/DL/effect-recognition/models/action_model.npz'
    last_n = 4
    reload_ = True

    # reload options
    if reload_ and os.path.exists(saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Building model'
    params = actrec.init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = actrec.load_params(saveto, params)

    tparams = actrec.init_tparams(params)

    trng, use_noise, \
          inps, alphas, \
          cost, \
          opts_out, preds = \
          actrec.build_model(tparams, model_options)
    f_preds = theano.function(inps, preds, profile=False, on_unused_input='ignore')

    data_test_test_pb = TestTestProto(valid_batch_size,maxlen,testing_stride,dataset,fps)
    dh_test_test = DataHandler(data_test_test_pb)
    test_test_dataset_size = dh_test_test.GetDatasetSize()

    num_test_test_batches = test_test_dataset_size / valid_batch_size
    if test_test_dataset_size % valid_batch_size != 0:
        num_test_test_batches += 1
    print 'Data handlers ready'
    print '-----'
    test_err = actrec.pred_acc_label_multi_slices(saveto, valid_batch_size, f_preds, maxlen, data_test_test_pb, dh_test_test, test_test_dataset_size, num_test_test_batches, last_n, test=True)
    print "Full dataset FINAL effect test error is %f" %test_err




