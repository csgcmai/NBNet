"""References:
Guangcan Mai, Kai Cao, Pong C. Yuen and Anil K. Jain. 
"On the Reconstruction of Face Images from Deep Face Templates." 
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2018)
"""

from symbol_nbnet import get_symbol
from symbol_perceptual import descriptor_symbol
from scipy.spatial.distance import cosine
from util import load_args as la
from util.util import *
import tensorflow as tf
import threading
import os,errno
import mxnet as mx
import numpy as np
import argparse
import logging
import time
import pdb

from datetime import datetime

# TensorBorad Flag
UseTensorBoard = True

if UseTensorBoard: 
    from TensorBoardLogging import LogMetricsCallback

file_path = os.path.dirname(os.path.abspath(__file__))
VGGPATH = file_path+'/../model/vgg/vgg19.params'

def main(args):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
    log_file_full_name = '%s%s.log'%(args.model_save_prefix,stamp)
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    size_p = 160
    
    if UseTensorBoard: 
        tb_folder = args.model_save_prefix+'_tblog/train'
        try:
            os.makedirs(tb_folder)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(tb_folder):
                pass
            else:
                raise
        batch_end_tb_callback = LogMetricsCallback(tb_folder,score_store=True)
    
    #net,_ = make_dcgan_sym(64,64,3)
    #net = get_symbol(6)
    net = get_symbol(6, net_switch=args.net_switch)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    devsid = [int(i) for i in args.gpus.split(',')]
    checkpoint = mx.callback.do_checkpoint(args.model_save_prefix)
    speedmeter = mx.callback.Speedometer(args.batch_size, 50)
    kv = mx.kvstore.create(args.kv_store)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)
        
    img_gen = img_generator(ctx=devs, batch_size=args.batch_size, 
                        model_prefix=args.face_gen_prefix, load_epoch=args.face_gen_epoch)
    fea_ext = feature_extractor(devsid)
    train = RandFaceIter(batch_size=args.batch_size, img_gen=img_gen, fea_ext=fea_ext)
    train = mx.io.PrefetchingIter(train)

    model = mx.mod.Module(symbol=net, context=devs)
    model.bind(data_shapes=train.provide_data,label_shapes=None, for_training=True)
    model.init_params(
        initializer=mx.init.Normal(0.02),
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True
        )

    model.init_optimizer(
        optimizer          = 'adam',
        optimizer_params = {
            "learning_rate" : 2e-4,
            "wd" : 0,
            "beta1" : 0.5,
            "lr_scheduler" : mx.lr_scheduler.FactorScheduler(step=5000, factor=0.94)
        })

    descriptor = descriptor_symbol(args.perceptual_layer)
    model_desc = mx.mod.Module(symbol=descriptor, context=devs)
    model_desc.bind(data_shapes=[('data',(args.batch_size,3,size_p,size_p))], 
                                        inputs_need_grad=True, for_training=True)
    arg_shapes,output_shapes,aux_shapes=descriptor.infer_shape(data=(args.batch_size,3,size_p,size_p))
    arg_names = descriptor.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    pretrained = mx.nd.load(VGGPATH)
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
    model_desc.set_params(arg_params=arg_dict,aux_params=None)

    logging.info('start with arguments %s', args)

    def cosine_score(label,pred):
        return 1.0-((label-pred)**2.0).mean()*label.shape[1]/2.

    #tpl_score = mx.metric.CustomMetric(feval=cosine_score,name='tpl_score')
    pixel_mae = mx.metric.MSE(name=args.perceptual_layer+'_mse')
    eval_metrics = CustomMetric()
    #eval_metrics.add(tpl_score)
    eval_metrics.add(pixel_mae)

    for epoch in range(20):#train at most 20 epoches
        tic = time.time()
        num_batch = 0

        end_of_batch = False
        train.reset()
        data_iter = iter(train)
        pixel_mae.reset()
        next_data_batch = next(data_iter)
        while not end_of_batch:
            data_batch = next_data_batch
            num_batch += 1
            model._exec_group.forward(data_batch,is_train=True)
            rec_img = model._exec_group.get_outputs()

            try:# pre fetch  next batch
                next_data_batch = next(data_iter)
                model.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True


            #ori_img = mx.nd.pad(data_batch.label[0],mode='reflect',pad_width=(0,0,0,0,pd,pd,pd,pd))
            ori_img = data_batch.label[0]
            model_desc._exec_group.forward(mx.io.DataBatch([ori_img]))
            desc_ori = model_desc._exec_group.get_outputs()[0].copy()

            #rec_img = mx.nd.pad(rec_img,mode='reflect',pad_width=(0,0,0,0,pd,pd,pd,pd))
            model_desc._exec_group.forward(mx.io.DataBatch(rec_img),is_train=True)
            desc_rec = model_desc._exec_group.get_outputs()[0]

            eval_metrics.update(desc_ori,desc_rec)
            desc_grad = desc_rec - desc_ori
            model_desc._exec_group.backward([desc_grad])

            img_grad = model_desc._exec_group.get_input_grads()[0]
            #img_grad=mx.nd.slice(img_grad,begin=(0,0,pd,pd),end=(img_grad.shape[0],img_grad.shape[1],pd+96,pd+96))
            
            model._exec_group.backward([img_grad])
            model.update()

            if num_batch % 1000 == 0:
                rep_ori = data_batch.data[0].asnumpy()
                rec_img_tmp = np.clip((rec_img[0].asnumpy().transpose((0,2,3,1))+1.0)/2.0, 0, 1)
                rec_img_tmp = [np.expand_dims(prewhiten(x),axis=0) for x in rec_img_tmp]
                rep_rec = fea_ext(np.concatenate(rec_img_tmp))
                facenet_scores = np.array([cosine(x,y) for x,y in zip(rep_ori, rep_rec)])

            batch_end_params = BatchEndParam(epoch=epoch,
                nbatch=num_batch,
                eval_metric=eval_metrics,
                locals=locals())

            if UseTensorBoard: 
                batch_end_tb_callback(batch_end_params) 
            speedmeter(batch_end_params)


        arg_params, aux_params = model.get_params()
        checkpoint(epoch,model.symbol,arg_params, aux_params)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training openface-generator-cnn")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model-save-prefix', type=str, default='../model/nbnet/of2img-percept',
                            help='the prefix of the model to save')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch size')
    parser.add_argument('--net-switch', type=str, default='nbnetb', choices=['nbneta','nbnetb','dcnn'],
                            help='the nbnet architecture to be used')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--model-load-prefix', type=str, default=None, help='the prefix of the model to load')
    parser.add_argument('--model-load-epoch', type=int, default=0, help='indicate the epoch for model-load-prefix')
    parser.add_argument('--face-gen-prefix', type=str, default='../model/dcgan/vgg-160-pt-slG', 
                            help='the prefix of the face generation model to load')
    parser.add_argument('--face-gen-epoch', type=int, default=0, help='epoch for loading face generation model')
    parser.add_argument('--perceptual-layer', type=str, default='relu3_2', help='layer for perceptual loss')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    args = parser.parse_args()
    args.model_save_prefix += '-'+args.net_switch+'-'+args.face_gen_prefix.split('/')[-1].split('-')[0]
    main(args)
