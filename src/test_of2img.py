import mxnet as mx
import numpy as np
import cv2
import argparse

from util.util import feature_extractor
from util.util import prewhiten

def main(args):
    gpuids = [int(i) for i in args.gpus.split(',')]
    ctx = [mx.gpu(i) for i in gpuids]
    fea_ext = feature_extractor(gpuids)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    model = mx.mod.Module(symbol=sym, context=ctx)
    model.bind(data_shapes=[('data',(args.batch_size,128))],label_shapes=None, for_training=False)
    model.set_params(arg_params, aux_params)

    # read the image and extract the facenet feature
    im = cv2.imread(args.input_image)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im_pw = np.expand_dims(prewhiten(im),axis=0)
    rep = fea_ext(im_pw)

    # reconstruct the facenet feature to image and save
    repIter = mx.io.NDArrayIter(rep,batch_size=args.batch_size)
    im_rec = model.predict(repIter).asnumpy()
    im_rec = im_rec.transpose((0, 2, 3, 1))
    im_rec = np.uint8(np.clip((im_rec[0]+1.0)*127.5, 0, 255))
    im_rec2 = np.float_(im_rec)
    im_rec = cv2.cvtColor(im_rec,cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_image,im_rec)

    # extract the facenet feature of the reconstructed image
    imrec_pw = np.expand_dims(prewhiten(im_rec2),axis=0)
    reprec = fea_ext(imrec_pw)

    print('matching score is ', np.dot(rep,reprec.T))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for face reconstruction demo")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--prefix', type=str, default='../model/nbnet/of2img-nbnetb-vgg',
                        help='the prefix of the model to load')
    parser.add_argument('--epoch', type=int, default=0, help='the epoch number to load')
    parser.add_argument('--batch-size', type=int, default=1, help='the batch size')
    parser.add_argument('--input-image', type=str, default='../data/1.jpg', help='')
    parser.add_argument('--output-image', type=str, default='../output/1.jpg', help='')
    args = parser.parse_args()
    main(args)
