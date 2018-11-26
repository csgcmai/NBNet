import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import logging
import cv2
from datetime import datetime
import argparse

def make_dcgan_sym(ngf, ndf, nc, predict=False,no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    scale=1.0507009873554804934193349852946;
    alpha=1.6732632423543772848170429916717;
    BatchNorm = mx.sym.BatchNorm
    rand = mx.sym.Variable('rand')
    if predict:
        rand = mx.sym.Variable('data')

    data = mx.sym.Reshape(rand, shape=(0,0,1,1))
    g1 = mx.sym.Deconvolution(data, name='g1', kernel=(5,5), num_filter=ngf*8, no_bias=no_bias)
    #gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    #gact = mx.sym.Activation(gbn1, name='gact1', act_type='relu')
    gact = mx.sym.LeakyReLU(g1,name='gact1', act_type='elu', slope=alpha)*scale

    ngg = ngf*4
    for i in range(2,7):
        g = mx.sym.Deconvolution(gact, name='g%d'%i, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngg, no_bias=no_bias)
    	#gbn = BatchNorm(g, name='gbn%d'%i, fix_gamma=fix_gamma, eps=eps)
    	#gact = mx.sym.Activation(gbn, name='gact%d'%i, act_type='relu')
        gact = mx.sym.LeakyReLU(g,name='gact%d'%i, act_type='elu', slope=alpha)*scale
        ngg //= 2
	

    g7 = mx.sym.Convolution(gact, name='g7', kernel=(3,3), stride=(1,1), pad=(1,1), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g7, name='gact7', act_type='tanh')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    ngg *= 2
    d1 = mx.sym.Convolution(data, name='d1', kernel=(3,3), stride=(1,1), pad=(1,1), num_filter=ngg, no_bias=no_bias)
    #dbn1 = BatchNorm(d1, name='dbn1', fix_gamma=fix_gamma, eps=eps)
    #dact = mx.sym.LeakyReLU(dbn1, name='dact1', act_type='leaky', slope=0.2)
    dact = mx.sym.LeakyReLU(d1,name='gact%d'%i, act_type='elu', slope=alpha)*scale

    for i in range(2,7):
        ngg *= 2
        d= mx.sym.Convolution(dact, name='d%d'%i, kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngg, no_bias=no_bias)
        #dbn= BatchNorm(d, name='dbn%d'%i, fix_gamma=fix_gamma, eps=eps)
        #dact = mx.sym.LeakyReLU(dbn, name='dact%d'%i, act_type='leaky', slope=0.2)
        dact = mx.sym.LeakyReLU(d,name='dact%d'%i, act_type='elu', slope=alpha)*scale

    d7 = mx.sym.Convolution(dact, name='d7', kernel=(5,5), num_filter=1, no_bias=no_bias)
    d7 = mx.sym.Flatten(d7)

    dloss = mx.sym.LogisticRegressionOutput(data=d7, label=label, name='dloss')
    return gout, dloss

