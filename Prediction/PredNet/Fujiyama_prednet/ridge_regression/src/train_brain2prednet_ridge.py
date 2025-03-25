#!/usr/bin/python
# coding: utf-8
"""For training a ridge regression which takes brain activity as input,
   and predicts internal representation in a layer of PredNet."""
#
# Description:
#   Given brain activity (= input) and internal representation (= target),
#   trains ridge regression with train data, and evaluate it with test data.
#   After training, predicts internal representation for test brain activity
#   using trained ridge regression.
#
# Usage:
#   python ./src/train_brain2prednet_ridge.py \
#   --alpha <parameter for L2 regularization> \
#   --depth <R0 | R1 | R2 | R3> \
#   --brain_train_path <absolute path to brain train file> \
#   --brain_test_path <absolute path to brain test file> \
#   --internal_representation_path <absolute path to internal representation directory> \
#   --model_save_path <absolute path to directory to save trained model> \
#   --predicted_save_path <absolute path to directory to save predicted internal representation>
#
from __future__ import print_function
import argparse
import cPickle as pickle

import matplotlib.image as mpimg
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Ridge regression')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='Parameter for L2 regularization.')
parser.add_argument('--depth', default='R0',
                    help='Depth of Representation module(y) in PredNet '
                         '(R0 | R1 | R2 | R3). R0 indicates the input layer.')
parser.add_argument('--brain_train_path',
                    default='/home/fujiyama/PredictiveCoding/brain_activity/cortex/'
                            'z_brain_train_preprocessed.pickle',
                    help='Path to the brain activity file for training.')
parser.add_argument('--brain_test_path', 
                    default='/home/fujiyama/PredictiveCoding/brain_activity/cortex/'
                            'z_brain_val_preprocessed.pickle',
                    help='Path to the brain activity file for test.')
parser.add_argument('--internal_representation_path',
                    default='/home/fujiyama/PredictiveCoding/internal_representation/',
                    help='Path to the internal representation directory.')
parser.add_argument('--model_save_path',
                    default='/home/fujiyama/PredictiveCoding/ridge_regression/models/',
                    help='Path to the directory to save trained model.')
parser.add_argument('--predicted_save_path',
                    default='/home/fujiyama/PredictiveCoding/'
                            'predicted_internal_representation/ridge/',
                    help='Path to the directory to save predicted internal representation.')
args = parser.parse_args()

alpha = args.alpha
depth = args.depth
# dim_table: (# dimensions of internal representaion of each layer in PredNet)
dim_table = {"R0":57600, "R1":230400, "R2":115200, "R3":57600}
dim_internal_representation = dim_table[depth]

brain_train_path = args.brain_train_path
brain_test_path = args.brain_test_path
internal_representation_path = args.internal_representation_path
model_save_path = args.model_save_path
predicted_save_path = args.predicted_save_path

print('# Alpha: {}'.format(alpha))
print('# Depth: {}'.format(depth))
print('# Brain for train: {}'.format(brain_train_path))
print('# Brain for test: {}'.format(brain_test_path))
print('# Internal representation: {}'.format(internal_representation_path))
print('')

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(predicted_save_path):
    os.makedirs(predicted_save_path)

if not os.path.exists(predicted_save_path + depth + '_alpha' + str(alpha)):
    os.makedirs(predicted_save_path + depth + '_alpha' + str(alpha))

dirs = ['vval001/', 'vval002/', 'vval003/', 'vval004/', 'vval005/']
for i in range(len(dirs)):
    if not os.path.exists(
        predicted_save_path + depth + '_alpha' + str(alpha) + '/' + dirs[i]):
        os.makedirs(predicted_save_path + depth + '_alpha' + str(alpha) + '/' + dirs[i])

print('Loading train data...')
begin = time.time()

x_data = pickle.load(open(args.brain_train_path, 'rb'))[:]
y_data = []
for i in range(0, 4497):
    frame_id = 345 + i * 60
    if frame_id <= 18105:
        dir = 'vtrn001/'
    elif frame_id <= 36105:
        dir = 'vtrn002/'
    elif frame_id <= 54105:
        dir = 'vtrn003/'
    elif frame_id <= 72105:
        dir = 'vtrn004/'
    elif frame_id <= 90105:
        dir = 'vtrn005/'
    elif frame_id <= 108105:
        dir = 'vtrn006/'
    elif frame_id <= 126105:
        dir = 'vtrn007/'
    elif frame_id <= 144105:
        dir = 'vtrn008/'
    elif frame_id <= 162105:
        dir = 'vtrn009/'
    elif frame_id <= 180105:
        dir = 'vtrn010/'
    elif frame_id <= 198105:
        dir = 'vtrn011/'
    elif frame_id <= 216105:
        dir = 'vtrn012/'
    elif frame_id <= 234105:
        dir = 'vtrn013/'
    elif frame_id <= 252105:
        dir = 'vtrn014/'
    else:
        dir = 'vtrn015/'
    internal_representation = np.load(
        internal_representation_path + dir + '%08d.npz' % (frame_id))
    y_data.append(internal_representation[depth].reshape((dim_internal_representation,)))
    internal_representation.close()

print('Completed loading train data.')
training_start = time.time()
print('Time needed to load train data : {:.2f} sec'.format(
        training_start - begin))
print('Start to train.')

# Train the ridge regression
reg = linear_model.Ridge(alpha=alpha)
reg.fit(x_data, y_data)
print(reg.coef_)
print(reg.intercept_)

print('Completed training regression.')

# Save the trained model
joblib.dump(reg, model_save_path + 'ridge' + depth + '_alpha' + str(alpha) + '.pickle')

training_end = time.time()
print('Time needed to train regression : {:.2f} sec'.format(
        training_end - training_start))

# Evaluate the trained model with train data
diff = y_data - reg.predict(x_data)
diff = diff.ravel() # This operation flattens diff.
loss = diff.dot(diff) / diff.size
print ('training loss : {}'.format(loss))

eval_train_data_end = time.time()
print('Time needed to evaluate the trained model with train data : {:.2f} sec'.format(
        eval_train_data_end - training_end))

## Load val data for eval
print('Loading validation data...')
x_val_data = pickle.load(open(brain_test_path, 'rb'))[:]
y_val_data = []
for i in range(0, 300):
    frame_id = 165 + i * 60
    if frame_id <= 3705:
        dir = 'vval001/'
    elif frame_id <= 7305:
        dir = 'vval002/'
    elif frame_id <= 10905:
        dir = 'vval003/'
    elif frame_id <= 14505:
        dir = 'vval004/'
    else:
        dir = 'vval005/'
    internal_representation = np.load(internal_representation_path + dir + '%08d.npz' % (frame_id))
    y_val_data.append(internal_representation[depth].reshape((dim_internal_representation,)))
    internal_representation.close()

load_val_data_end = time.time()
print('Time needed to load validation data : {:.2f} sec'.format(
        load_val_data_end - eval_train_data_end))

## Evaluate the trained model with val data
predicted_y = reg.predict(x_val_data)
diff = y_val_data - predicted_y
diff = diff.ravel() # This operation flattens diff.
loss = diff.dot(diff) / diff.size
print ('validation loss : {}'.format(loss))

eval_val_data_end = time.time()
print('Time needed to evaluate the trained model with val data : {:.2f} sec'.format(
        eval_val_data_end - load_val_data_end))

## Save predicted internal representation for validation data
for i in range(predicted_y.shape[0]):
    frame_id = 165 + i * 60
    if frame_id <= 3705:
        dir = 'vval001/'
    elif frame_id <= 7305:
        dir = 'vval002/'
    elif frame_id <= 10905:
        dir = 'vval003/'
    elif frame_id <= 14505:
        dir = 'vval004/'
    else:
        dir = 'vval005/'

    if depth == 'R0':
        np.savez_compressed(predicted_save_path + depth + '_alpha' + str(alpha) + \
                                '/' +  dir + '%08d.npz' % (frame_id),
                            R0=predicted_y[i,:].reshape((1,3,120,160)))
    elif depth == 'R2':
        np.savez_compressed(predicted_save_path + depth + '_alpha' + str(alpha) + \
                                '/' +  dir + '%08d.npz' % (frame_id),
                            R2=predicted_y[i,:].reshape((1,96,30,40)))
    elif depth == 'R3':
        np.savez_compressed(predicted_save_path + depth + '_alpha' + str(alpha) + \
                                '/' +  dir + '%08d.npz' % (frame_id),
                            R3=predicted_y[i,:].reshape((1,192,15,20)))

    print('saved : ' + predicted_save_path + depth + '_alpha' + str(alpha) + \
              '/' + dir + '%08d.npz' % (frame_id))

save_end = time.time() 
print('Time needed to save : {:.2f} sec'.format(
        save_end - eval_val_data_end))
print('total time from the very beginning : {:.2f} sec'.format(save_end - begin))
