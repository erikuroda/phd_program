#!usr/bin/python
#-*- coding: utf-8 -*-

'''
This calculates the average of MSE between 2 frames in a row.
(RGB value / 255)

Usage:
     python MSE.py <arg0> <arg1> <arg2> ...
Arguments:
     -d <image list>
     -b <beginning id> (0 <= b < #row(sequence list))
     -e <end id> (0 < e <= #row(sequence list))
     -r <root directory>
     -c <# channels>
     -s <size of each image>
'''

import argparse
import math
import numpy as np
import os
import time

from PIL import Image

import chainer
from chainer.functions.loss.mean_squared_error import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('-data', '-d', default = '',
                    help = 'Data you want to calculate MSE.')
parser.add_argument('-begin_id', '-b', default = 0,
                    type = int, help = 'Beginning frame ID.')
parser.add_argument('-end_id', '-e', default = 0,
                    type = int, help = 'End frame ID.')
parser.add_argument('--root', '-r',
                    default = '/home/fujiyama/PredictiveCoding/',
                    help = 'Root directory path.')
parser.add_argument('--channels', '-c', default = 3, help = 
                    '(#channels): 3 for RGB or 1 for grey scale.')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of each image. width,height (pixels).')
args = parser.parse_args()

if (not args.data):
    print ('Please specify image sequence list.')
    exit()

if (not args.begin_id) or (not args.end_id):
    print ('Please specify both a beginning id and an end id.')
    exit()

if (not args.root):
    print ('Please specify the root directory.')

args.size = args.size.split(',')
for i in range(len(args.size)):
    args.size[i] = int(args.size[i])

def load_list(path, root):
    tuples = []
    for line in open(root + path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples
    
def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    image = image[:, :, :].astype(np.float32)
    image /= 255
    return image

image_list = load_list(args.data, args.root) 

loss = 0.0

t_image = np.ndarray((1, args.channels, args.size[1], args.size[0]),
                     dtype=np.float32)
tplus1_image = np.ndarray((1, args.channels, args.size[1], args.size[0]),
                          dtype=np.float32)

begin_time = time.time()
t_image[0] = read_image(image_list[args.begin_id])
for nth in range(args.begin_id, args.end_id):
    tplus1_image[0] = read_image(image_list[nth + 1])
    loss += mean_squared_error(chainer.Variable(t_image),
                               chainer.Variable(tplus1_image)).data
    t_image[0] = tplus1_image[0]

mean_MSE = loss / (args.end_id - args.begin_id)
end_time = time.time()

print('mean_MSE: {}'.format(mean_MSE))
print('elapsed time: {}'.format(end_time - begin_time))
