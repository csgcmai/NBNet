#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# this file is coming from openface: https://github.com/cmusatyalab/openface, with some changes
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import pdb
import numpy as np
import mxnet as mx
import os,errno
import random
import shutil
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
mtcnn_dir = file_dir+'/mxnet_mtcnn/'
sys.path.append(mtcnn_dir)
from mtcnn_detector import MtcnnDetector

def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Image:
    """Object containing image metadata."""

    def __init__(self, cls, name, path):
        """
        Instantiate an 'Image' object.

        :param cls: The image's class; the name of the person.
        :type cls: str
        :param name: The image's name.
        :type name: str
        :param path: Path to the image on disk.
        :type path: str
        """
        assert cls is not None
        assert name is not None
        assert path is not None

        self.cls = cls
        self.name = name
        self.path = path

    def getBGR(self):
        """
        Load the image from disk in BGR format.

        :return: BGR image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        try:
            bgr = cv2.imread(self.path)
        except:
            bgr = None
        return bgr

    def getRGB(self):
        """
        Load the image from disk in RGB format.

        :return: RGB image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        bgr = self.getBGR()
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = None
        return rgb

    def __repr__(self):
        """String representation for printing."""
        return "({}, {})".format(self.cls, self.name)


def iterImgs(directory):
    u"""
    Iterate through the images in a directory.

    The images should be organized in subdirectories
    named by the image's class (who the person is)::

       $ tree directory
       person-1
       ── image-1.jpg
       ├── image-2.png
       ...
       └── image-p.png

       ...

       person-m
       ├── image-1.png
       ├── image-2.jpg
       ...
       └── image-q.png


    :param directory: The directory to iterate through.
    :type directory: str
    :return: An iterator over Image objects.
    """
    assert directory is not None

    exts = [".jpg", ".png"]

    if args.frgc_flag:
        im_list = open(args.frgc_blufr)
        im_list = [l for l in im_list]
        for path in im_list:
            path = path[0:-1]
            fName = path.split('/')[1]
            imageClass = fName.split('d')[0]
            (imageName, ext) = os.path.splitext(fName)
            yield Image(imageClass, imageName, os.path.join(directory, path))
        return

    for subdir, dirs, files in os.walk(directory):
        for path in files:
            (imageClass, fName) = (os.path.basename(subdir), path)
            (imageName, ext) = os.path.splitext(fName)
            if ext in exts:
                yield Image(imageClass, imageName, os.path.join(subdir, fName))

def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")

def alignMain(args):
    mkdirP(args.outputDir)

    imgs = list(iterImgs(args.inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    align = MtcnnDetector(model_folder=mtcnn_dir+'model', 
            ctx=mx.gpu(int(args.gpus.split(',')[0])), 
            num_worker = 4 , 
            minsize = 50,
            accurate_landmark = True)

    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.outputDir, imgObject.cls)
        mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + "." + args.ext

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getBGR()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                detect = align.detect_face(rgb)
                if detect is not None:
                    bb = detect[0]
                    pts = detect[1]
                    if bb.shape[0] > 1:
                        bb_size = (bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
                        i_max = np.argmax(bb_size)
                        bb = bb[i_max:i_max+1]
                        pts = pts[i_max:i_max+1]
                    outBgr = align.extract_image_chips(rgb, pts, args.size, args.pad)
                    outBgr = outBgr[0]
                else:
                    if args.verbose:
                        print("  + Unable to align.")

            if args.fallbackLfw and outRgb is None:
                nFallbacks += 1
                deepFunneled = "{}/{}.jpg".format(os.path.join(args.fallbackLfw,
                                                               imgObject.cls),
                                                  imgObject.name)
                shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(args.outputDir,
                                                                          imgObject.cls),
                                                             imgObject.name))

            if outBgr is not None:
                #if args.verbose:
                    #print("  + Writing aligned file to disk.")
                #outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                #pdb.set_trace()
                cv2.imwrite(imgName, outBgr)

    if args.fallbackLfw:
        print('nFallbacks:', nFallbacks)


if __name__ == '__main__':
    alignmentParser = argparse.ArgumentParser() 
    alignmentParser.add_argument('inputDir', type=str, help="Input image directory.")
    alignmentParser.add_argument(
        'outputDir', type=str, help="Output directory of aligned images.")
    alignmentParser.add_argument('--pad', type=float, help="pad for face detection region",
                                 default=0.156)
    alignmentParser.add_argument('--size', type=int, help="Default image size.",
                                 default=160)
    alignmentParser.add_argument('--gpus', type=str, help="gpu to be used",
                                 default='0,1')
    alignmentParser.add_argument('--ext', type=str, help="Default image extension.",
                                 default='jpg')
    alignmentParser.add_argument('--fallbackLfw', type=str,
                                 help="If alignment doesn't work, fallback to copying the deep funneled version from this directory..")
    alignmentParser.add_argument('--verbose', action='store_true')
    alignmentParser.add_argument('--frgc-flag', action='store_true')
    alignmentParser.add_argument('--frgc-blufr', type=str, help="specify im list if FRGC dataset",
                                 default=None)

    args = alignmentParser.parse_args()
    alignMain(args)
