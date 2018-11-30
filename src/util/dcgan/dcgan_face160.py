"""References:
Guangcan Mai, Kai Cao, Pong C. Yuen and Anil K. Jain. 
"On the Reconstruction of Face Images from Deep Face Templates." 
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2018)

Alec Radford, Luke Metz and Soumith Chintala.
"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks."
ICLR2016
"""

import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2
from datetime import datetime
from symbol_dcgan160 import make_dcgan_sym
from collections import namedtuple
import argparse
import pdb

mx.random.seed(128)
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'eval_metric',
                            'locals'])

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

class ImagenetIter(mx.io.DataIter):
    def __init__(self, path, batch_size, data_shape):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec = path,
            data_shape  = data_shape,
            batch_size  = batch_size,
            )
        self.provide_data = [('data', (batch_size,) + data_shape)]
        self.provide_label = []

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        data = self.internal.getdata()
        data_tmp = data.asnumpy().transpose((0,2,3,1))/255.
        data_tmp = np.concatenate([np.expand_dims(cv2.resize(x[20:140,20:140,:],(160,160)),axis=0) for x in data_tmp])
        data_tmp = data_tmp.transpose((0,3,1,2))
        data = data_tmp*2.0 - 1.0
        return [mx.nd.array(data)]

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    plt.imshow(title, buff)
    #cv2.waitKey(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command for training dcgan")
    parser.add_argument('--gpus', type=str, help='the gpu will be used, e.g "2"')
    parser.add_argument('--data-path', type=str, help='the rec file for training')
    parser.add_argument('--model-save-prefix', type=str, help='prefix for model saving')
    args = parser.parse_args()
    #logging.basicConfig(level=logging.DEBUG)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
    log_file_full_name = '%s%s.log'%(args.model_save_prefix,stamp)
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # =============setting============
    dataset = args.data_path.split('/')[-1].split('.')[0]
    imgnet_path = args.data_path
    ndf = 64
    ngf = 64
    nc = 3
    batch_size = 64
    Z = 100
    lr_G = 0.0002
    lr_D = 0.00005
    beta1 = 0.5
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    check_point = True

    symG, symD = make_dcgan_sym(ngf, ndf, nc)
    speedmeter = mx.callback.Speedometer(batch_size, 100)
    ckp_G = mx.callback.do_checkpoint(args.model_save_prefix+'G')
    ckp_D = mx.callback.do_checkpoint(args.model_save_prefix+'D')

    # ==============data==============
    train_iter = ImagenetIter(imgnet_path, batch_size, (3, 160, 160))
    rand_iter = RandIter(batch_size, Z)
    label = mx.nd.zeros((batch_size,), ctx=mx.cpu(0))

    # =============module G=============
    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr_G,
            'wd': 0.,
            'beta1': beta1,
        })
    mods = [modG]

    # =============module D=============
    modD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
    modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr_D,
            'wd': 0.,
            'beta1': beta1,
        })
    mods.append(modD)


    # ============printing==============
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    mon = None
    if mon is not None:
        for mod in mods:
            pass

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == (label>0.5)).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

    mG = mx.metric.CustomMetric(fentropy,name='Gfentropy')
    mD = mx.metric.CustomMetric(fentropy,name='Dfentropy')
    mACC = mx.metric.CustomMetric(facc)

    print 'Training...'
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(100):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            rbatch = rand_iter.next()

            if mon is not None:
                mon.tic()

            modG._exec_group.forward(rbatch, is_train=True)
            outG = modG._exec_group.get_outputs()

            # update discriminator on fake
            mx.nd.random_uniform(low=0.,high=0.3,out=label)
            #label[:] = 0
            modD._exec_group.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD._exec_group.backward()
            #modD.update()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update discriminator on real
            mx.nd.random_uniform(low=0.7,high=1.2,out=label)
            #label[:] = 1
            batch.label = [label]
            modD._exec_group.forward(batch, is_train=True)
            modD._exec_group.backward()
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update()

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update generator
            mx.nd.random_uniform(low=0.7,high=1.2,out=label)
            #label[:] = 1
            modD._exec_group.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD._exec_group.backward()
            diffD = modD._exec_group.get_input_grads()
            modG._exec_group.backward(diffD)
            modG.update()

            mG.update([label], modD.get_outputs())


            if mon is not None:
                mon.toc_print()

            t += 1
            for mtc in [mACC,mG,mD]:
                batch_end_params = BatchEndParam(epoch=epoch,
                    nbatch=t,
                    eval_metric=mtc,
                    locals=locals())
                speedmeter(batch_end_params)
        arg_params, aux_params = modG.get_params()
        ckp_G(epoch,modG.symbol,arg_params, aux_params)

        arg_params, aux_params = modD.get_params()
        ckp_D(epoch,modD.symbol,arg_params, aux_params)
            #if t % 10 == 0:
            #    print 'epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get()
            #    mACC.reset()
            #    mG.reset()
            #    mD.reset()

                #visual('gout', outG[0].asnumpy())
                #diff = diffD[0].asnumpy()
                #diff = (diff - diff.mean())/diff.std()
                #visual('diff', diff)
                #visual('data', batch.data[0].asnumpy())

    #    if check_point:
    #        print 'Saving...'
    #        modG.save_params('model_dcgan160/%s_G_%s-%04d.params'%(dataset, stamp, epoch))
    #        modD.save_params('model_dcgan160/%s_D_%s-%04d.params'%(dataset, stamp, epoch))



