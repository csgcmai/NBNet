# Adapted from https://github.com/Nicatio/Densenet/blob/master/mxnet/symbol_densenet.py

"""References:
Guangcan Mai, Kai Cao, Pong C. Yuen and Anil K. Jain. 
"On the Reconstruction of Face Images from Deep Face Templates." 
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2018)

Gao Huang, Zhuang Liu, Laurens van der Maaten and Kilian Weinberger.
"Densely Connected Convolutional Networks." CVPR2017
"""

import mxnet as mx
import numpy as np
from util.dcgan.symbol_dcgan160 import make_dcgan_sym

def add_layer(
    x,
    num_channel,
    name,
    dconv=False,
    pad=1,
    kernel_size=3,
    dropout=0.,
    l2_reg=1e-4):

    x = mx.symbol.BatchNorm(x, eps = l2_reg, name = name + '_batch_norm')
    x = mx.symbol.Activation(data = x, act_type='relu', name = name + '_relu')
    if not dconv:
        x = mx.symbol.Convolution(
            name=name + '_conv',
            data=x,
            num_filter=num_channel,
            kernel=(kernel_size, kernel_size),
            stride=(1, 1),
            pad=(pad, pad),
            no_bias=True
        )
    else:
        x = mx.symbol.Deconvolution(
            name=name + '_dconv',
            data=x,
            num_filter=num_channel,
            kernel=(kernel_size, kernel_size),
            stride=(2, 2),
            pad=(pad, pad),
            no_bias=True
        )
    if dropout > 0:
        x = mx.symbol.Dropout(x, p = dropout, name = name + '_dropout')
    return x

def nb_block(
    x,
    num_layers,
    growth_rate,
    name,
    net_switch = 'nbnetb',
    dropout=0.,
    l2_reg=1e-4):

    num_layers = np.int_(num_layers)
    if net_switch == 'nbneta':
        out = [x,]
        for i in range(num_layers):
            out.append(add_layer(out[i], growth_rate, name=name+'_layer_'+str(i), dropout=dropout, l2_reg=l2_reg))
        x = mx.symbol.concat(*out, name=name+'_concat_'+str(i))
    elif net_switch == 'nbnetb':
        for i in range(num_layers):
            out = add_layer(x, growth_rate, name=name+'_layer_'+str(i), dropout=dropout, l2_reg=l2_reg)
            x = mx.symbol.concat(x, out, name=name+'_concat_'+str(i))
    return x

def transition_block(
    x,
    num_channel,
    name,
    dropout=0.,
    l2_reg=1e-4):

    x = add_layer(x, num_channel, name = name, dconv=True, pad=1, kernel_size=4, dropout=dropout, l2_reg=l2_reg)
    #x = mx.symbol.Pooling(x, name = name + '_pool', global_pool = False, kernel = (2,2), stride = (2,2), pool_type = 'avg')
    return x

def get_symbol(
    num_block,
    num_layer=int(32),
    growth_rate=int(8),
    dropout=0.,
    l2_reg=1e-4,
    net_switch='nbnetb',
    init_channels=int(256),
):
    if net_switch == 'dcnn':
        net,_ = make_dcgan_sym(64,64,3,True)
        return net
    n_channels = init_channels

    data = mx.symbol.Variable(name='data')
    data = mx.symbol.Reshape(data,shape=(0,0,1,1))
    conv = mx.symbol.Deconvolution(name="conv0",data=data,num_filter=n_channels,kernel=(5, 5),no_bias=True)
    #conv = mx.symbol.Reshape(conv,shape=(0,0,6,6))

    for i in range(num_block - 1):
        conv = nb_block(conv, num_layer, growth_rate, name = 'nb'+str(i)+'_',
                        dropout=dropout, l2_reg=l2_reg, net_switch=net_switch)
        n_channels /= 2
        num_layer /= 2
        conv = transition_block(conv, np.int_(n_channels), name = 'trans'+str(i)+'_', dropout=dropout, l2_reg=l2_reg)

    conv = nb_block(conv, num_layer, growth_rate, name = 'last_', dropout=dropout, l2_reg=l2_reg)
    conv = mx.symbol.BatchNorm(conv, eps = l2_reg, name = 'batch_norm_last')
    conv = mx.symbol.Activation(data = conv, act_type='relu', name='relu_last')
    conv = mx.symbol.Convolution(data=conv, kernel=(1,1), num_filter=3, no_bias=True,name='conv_last')
    conv = mx.symbol.Activation(data=conv, act_type='tanh', name='tanh_last')

    return conv
