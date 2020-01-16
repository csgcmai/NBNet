import numpy as np
import cv2
import argparse

from util.util import feature_extractor
from util.util import prewhiten
from tqdm import tqdm
from scipy.io import loadmat,savemat

def main(args):
    gpuids = [int(i) for i in args.gpus.split(',')]
    ctx = [mx.gpu(i) for i in gpuids]
    data = loadmat(args.input_image_mat)['data']
    data = np.transpose(data,(3,2,1,0))
    fea_ext = feature_extractor(gpuids)

    # read the image and extract the facenet feature
    reps = []
    for img in tqdm(data):

        im_pw = np.expand_dims(prewhiten(np.squeeze(img)),axis=0)
        rep = fea_ext(im_pw)
        reps.append(rep)
    savemat(args.output_feature_mat,{'data':reps})

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for face reconstruction demo")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--input-image-mat', type=str)
    parser.add_argument('--output-feature-mat', type=str)
    args = parser.parse_args()
    main(args)
