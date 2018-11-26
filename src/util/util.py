import tensorflow as tf
import threading
import mxnet as mx
import numpy as np
import sys
import os
from facenet_tf.src import facenet
from collections import namedtuple
from dcgan.symbol_dcgan160 import make_dcgan_sym
import load_args as la

file_path = os.path.dirname(os.path.abspath(__file__))
facenet_path = file_path+'/facenet_tf/model/20170512-110547/20170512-110547.pb'

def feature_extractor(devs,model_path=facenet_path):
    iph = []; ebds = []; ptph = []; ss = []
    for i_dev,dev in enumerate(devs):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,
                                            allow_growth=True,visible_device_list=str(dev))
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                ss.append(sess)
                sess.as_default()
                # Load the model
                facenet.load_model(model_path)
                #pdb.set_trace()
                
                # Get input and output tensors
                iph.append(tf.get_default_graph().get_tensor_by_name("input:0"))
                ebds.append(tf.get_default_graph().get_tensor_by_name("embeddings:0"))
                ptph.append(tf.get_default_graph().get_tensor_by_name("phase_train:0"))

    def single_feature_extract (i,image,out_list=range(len(devs))):
        with ptph[i].graph.as_default():
            feed_dict = { iph[i].graph.get_tensor_by_name("input:0"):image, ptph[i]:False }
            out_list[i] = ss[i].run(ebds[i], feed_dict=feed_dict)
                
    if len(devs) <= 1:
        def f_extract(images):
            out_list = [0]
            single_feature_extract(0,images,out_list)
            return out_list[0]
        return f_extract

    def multi_feature_extract(images):
        out_list = range(len(devs))
        ims = np.split(images,len(devs))
        ths = [threading.Thread(target=single_feature_extract,args=(i,ims[i],out_list)) 
                                        for i in range(len(devs))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        return np.concatenate(out_list)
    return multi_feature_extract

def img_generator(ctx=mx.gpu(1), batch_size=64, model_prefix=None, load_epoch=0, rand_crop=False):
    tmp_arg_params,tmp_aux_params = la.load_args(model_prefix, load_epoch)
    net, _ = make_dcgan_sym(64,64,3,True)

    model_trained = mx.mod.Module(symbol=net, context=ctx)
    model_trained.bind(data_shapes=[('data',(batch_size,100))],label_shapes=None, for_training=False)

    model_trained.set_params(
        arg_params=tmp_arg_params,
        aux_params=tmp_aux_params,
        allow_missing=False
        #allow_extra=True
        )

    def face_gen(data):
        model_trained._exec_group.forward(mx.io.DataBatch([data],None),is_train=False)
        f_gen = model_trained._exec_group.get_outputs()
        return f_gen[0].asnumpy()

    return face_gen

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

class RandFaceIter(mx.io.DataIter):
    def __init__(self, batch_size, img_gen, fea_ext, ndim=100):
        self.batch_size = batch_size
        self.ndim = ndim
        self.img_gen = img_gen
        self.fea_ext = fea_ext
        self.counter = 0
        self.maxcount = np.int_(5e3) # for calling the epoch_end
        self.provide_data = [('data', (batch_size, 128))]
        self.provide_label = [('image_label', (batch_size, 3, 160, 160))]

    def iter_next(self):
        if(self.counter>self.maxcount):
            self.counter = 0
            return False
        self.counter += 1
        rand_data = mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim))
        self.cur_img = self.img_gen(rand_data)
        data = np.clip((self.cur_img.transpose((0,2,3,1))+1.0)/2.0, 0, 1)
        data = [np.expand_dims(prewhiten(x),axis=0) for x in data]
        self.cur_reps = self.fea_ext(np.concatenate(data))
        return True

    def getdata(self):
        return [mx.nd.array(self.cur_reps)]

    def getlabel(self):
        return [mx.nd.array(self.cur_img)]

class CustomMetric(mx.metric.CompositeEvalMetric):
    def __init__(self,*args,**kwargs):
        super(CustomMetric,self).__init__(*args,**kwargs)

    def update(self,labels,preds):
        for i_mt in range(len(self.metrics)):
            self.metrics[i_mt].update([labels[i_mt]],[preds[i_mt]])

BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'eval_metric',
                            'locals'])
