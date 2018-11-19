#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 23:25:05 2018

@author: bony
"""

"""Split the SIGNS dataset into train/val/test and resize images to 64x64.

The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (512, 512).
Resizing to (224, 224) or (229, 229) for IncenptionV3 
Makes data size compatible with standard Deep NN architectures 

"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 224  #input size for most DNN arch

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_dir', default='/home/bony/Deep_Learning_Stanford_CS230/Project/Data', help="Directory with the training dataset")
#parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new resized data")
parser.add_argument('--datatype', default='test', help="train set or val set")
parser.add_argument('--output_size', default=299, help="Size of the resize images")

def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()
    
    data_dir = os.path.join(args.data_root_dir, args.datatype)
    assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)

    print("data root dir = ", data_dir)
    print("output size = ", args.output_size)

    # Get the filenames in train directory 
    file_dir = os.listdir(data_dir)
    filenames = [ os.path.join(data_dir, f) for f in file_dir if f.endswith('.png')]

    new_sub_dir = args.datatype+"_"+str(args.output_size)
    
    output_dir = os.path.join(args.data_root_dir, new_sub_dir)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print("Warning: output dir {} already exists".format(output_dir))
        

    print("Processing {} data, saving preprocessed data to {}".format(args.datatype, output_dir))
    for filename in tqdm(filenames):
    
        resize_and_save(filename, output_dir, int(args.output_size))

    print("Done building dataset")
