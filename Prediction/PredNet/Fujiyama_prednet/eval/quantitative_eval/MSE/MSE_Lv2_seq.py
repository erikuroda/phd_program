#!usr/bin/python
#-*- coding: utf-8 -*-

'''
This program is for comparison between (input, predict) and (predict, correct)
in a specific sequence.
If (input, predict) > (predict, correct), then the learned model can generate
predictive frames which are closer to the correct frames than they are to the
last input frames.

This calculates and outputs the average of MSE at (input, predict) and
(predict, correct).
(RGB value / 255)

Usage:
     python MSE_Lv2_seq.py <arg0> <arg1> <arg2> ...
Arguments:
     -d <sequence name>
     -r <sequence range>
     -p <image directory path>
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
parser.add_argument('--data', '-d', default='',
                    help='A sequence name you want to evaluate.')
parser.add_argument('--seq_range', '-r',
                    help='Range of the sequence (eg: 0,50)')
parser.add_argument('--path', '-p', default='', help='Image directory path.')
parser.add_argument('--channels', '-c', default=3,
                    help='(#channels): 3 for RGB or 1 for grey scale.')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of each image. width,height (pixels).')
args = parser.parse_args()

if (not args.data):
    print ('Please specify image sequence name.')
    exit()

if not args.seq_range:
    print('Please specify the sequence range.')
    exit()

if (not args.path):
    print ('Please specify the image directory path.')

seq_name = args.data
image_dir = args.path

args.seq_range = args.seq_range.split(',')
begin_id = int(args.seq_range[0])
end_id = int(args.seq_range[1])

args.size = args.size.split(',')
for i in range(len(args.size)):
    args.size[i] = int(args.size[i])

print("begin_id : {}".format(begin_id))
print("end_id : {}".format(end_id))

def read_image(path):
    image = np.asarray(Image.open(image_dir + path)).transpose(2, 0, 1)
    image = image[:, :, :].astype(np.float32)
    image /= 255
    return image

input_predict_loss = 0.0
predict_correct_loss = 0.0

t_x = np.ndarray((1, args.channels, args.size[1], args.size[0]),
                     dtype=np.float32)
tplus1_x = np.ndarray((1, args.channels, args.size[1], args.size[0]),
                          dtype=np.float32)
t_y = np.ndarray((1, args.channels, args.size[1], args.size[0]),
                     dtype=np.float32)

begin_time = time.time()

t_x[0] = read_image('test_' + str(begin_id) + '_x.jpg') # input
t_y[0] = read_image('test_' + str(begin_id) + '_y.jpg') # prediction

check = 0

for nth in range(begin_id, end_id):
    tplus1_x[0] = read_image('test_' + str(nth + 1) + '_x.jpg') # correct
    input_predict_loss += mean_squared_error(chainer.Variable(t_x),
                               chainer.Variable(t_y)).data
    predict_correct_loss += mean_squared_error(chainer.Variable(t_y),
                               chainer.Variable(tplus1_x)).data
    t_x[0] = tplus1_x[0]
    t_y[0] = read_image('test_' + str(nth + 1) + '_y.jpg')
    check += 1

num_pair = end_id - begin_id
mean_input_predict_MSE = input_predict_loss / num_pair
mean_predict_correct_MSE = predict_correct_loss / num_pair
end_time = time.time()

if check == num_pair:
    print("Successfully checked!!")
print('mean_input_predict_MSE: {}'.format(mean_input_predict_MSE))
print('mean_predict_correct_MSE: {}'.format(mean_predict_correct_MSE))

print('input_predict_loss: {}'.format(input_predict_loss))
print('predict_correct_loss: {}'.format(predict_correct_loss))
print('num_pair: {}'.format(num_pair))

if mean_input_predict_MSE > mean_predict_correct_MSE:
    print("The model can successfully generate predictive frames which are " +
          "closer to the correct frames than they are to the last input frames!!")
else:
    print("The model generates predictive frames which are closer to " +
          "the last input frames than they are to the correct frames...")

print('elapsed time: {}'.format(end_time - begin_time))

