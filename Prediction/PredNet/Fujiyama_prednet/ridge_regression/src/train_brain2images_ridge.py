#!/usr/bin/python
# coding: utf-8
"""For training a ridge regression which takes brain activity as input,
   and predicts images (stimuli / predicted images)."""
#
# Description:
#   Given brain activity (= input) and (stimulus / predicted) images (= target),
#   trains ridge regression with train data, and evaluate it with test data.
#   After training, predicts (stimulus / predicted) images for test brain
#   activity using trained ridge regression.
#
# Usage:
#   python ./src/train_brain2stimuli_ridge.py \
#   --alpha <parameter for L2 regularization> \
#   --brain_train_path <absolute path to brain train file> \
#   --brain_test_path <absolute path to brain test file> \
#   --images_path <absolute path to images directory> \
#   --target <input | predicted> \
#   --model_save_path <absolute path to directory to save trained model> \
#   --predicted_save_path <absolute path to directory to save predicted images>
#
from __future__ import print_function
import argparse
import cPickle as pickle
import os
import sys
import time

import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

import image_io

parser = argparse.ArgumentParser(description='Ridge regression')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='Parameter for L2 regularization.')
parser.add_argument('--brain_train_path',
                    default='/home/fujiyama/PredictiveCoding/brain_activity/cortex/'
                            'z_brain_train_preprocessed.pickle',
                    help='Path to the brain activity file for training.')
parser.add_argument('--brain_test_path', 
                    default='/home/fujiyama/PredictiveCoding/brain_activity/cortex/'
                            'z_brain_val_preprocessed.pickle',
                    help='Path to the brain activity file for test.')
parser.add_argument('--images_path',
                    default='/home/fujiyama/PredictiveCoding/stimuli/',
                    help='Path to the stimulus images directory.')
parser.add_argument('--target', type=str, help="input | predicted")
parser.add_argument('--model_save_path',
                    default='/home/fujiyama/PredictiveCoding/ridge_regression/models/',
                    help='Path to the directory to save trained model.')
parser.add_argument('--model_name', type=str,
                    help='Model name (recommended a name indentifying target and alpha).')
parser.add_argument('--predicted_save_path',
                    default='/home/fujiyama/PredictiveCoding/'
                            'predicted_images/ridge/',
                    help='Path to the directory to save predicted images.')
args = parser.parse_args()

alpha = args.alpha
image_size = 57600

brain_train_path = args.brain_train_path
brain_test_path = args.brain_test_path
images_path = args.images_path
model_save_path = args.model_save_path
model_name = args.model_name
predicted_save_path = args.predicted_save_path

print('# Alpha: {}'.format(alpha))
print('# Brain for train: {}'.format(brain_train_path))
print('# Brain for test: {}'.format(brain_test_path))
print('# Images: {}'.format(images_path))
print('')

sys.stdout.flush()

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(predicted_save_path):
    os.makedirs(predicted_save_path)

dirs = ['vtrn001/', 'vtrn002/', 'vtrn003/', 'vtrn004/', 'vtrn005/',
        'vtrn006/', 'vtrn007/', 'vtrn008/', 'vtrn009/', 'vtrn010/',
        'vtrn011/', 'vtrn012/', 'vtrn013/', 'vtrn014/', 'vtrn015/',
        'vval001/', 'vval002/', 'vval003/', 'vval004/', 'vval005/']

for i in range(len(dirs)):
    if not os.path.exists(os.path.join(predicted_save_path, dirs[i])):
        os.makedirs(os.path.join(predicted_save_path, dirs[i]))

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
    # img.shape => (120, 160, 3) 
    parent_dir = os.path.join(images_path, dir)
    child_dir = 'paired_ver/%s_images' % args.target
    filename = os.path.join(parent_dir, child_dir)
    img = image_io.read_image(os.path.join(
        filename,
        (str(frame_id) + '_' + args.target + '.jpg')))
    y_data.append(img.reshape((image_size,)).copy())

print('Completed loading train data.')
training_start = time.time()
print('Time needed to load train data : {:.2f} sec'.format(
        training_start - begin))
print('Start to train.')

sys.stdout.flush()

# Train the ridge regression
reg = linear_model.Ridge(alpha=alpha)
reg.fit(x_data, y_data)
print(reg.coef_)
print(reg.intercept_)

print('Completed training regression.')

sys.stdout.flush()

# Save the trained model
joblib.dump(reg, model_save_path + model_name + '.pickle')

training_end = time.time()
print('Time needed to train regression : {:.2f} sec'.format(
        training_end - training_start))

sys.stdout.flush()

# Evaluate the trained model with train data
diff = y_data - reg.predict(x_data)
diff = diff.ravel() # This operation flattens diff.
loss = diff.dot(diff) / diff.size
print ('training loss : {}'.format(loss))

eval_train_data_end = time.time()
print('Time needed to evaluate the trained model with train data : {:.2f} sec'.format(
        eval_train_data_end - training_end))

sys.stdout.flush()

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
    img = image_io.read_image(os.path.join(
            os.path.join(images_path, dir),
            (str(frame_id) + '_' + args.target + '.jpg')))
    y_val_data.append(img.reshape((image_size,)))

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

sys.stdout.flush()

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

    filename = (os.path.join(os.path.join(predicted_save_path, dir),
                             '%08d_%s.jpg' % (frame_id, args.target)))
    image_io.write_image(predicted_y[i,:].reshape((120, 160, 3)).copy(), filename)

save_end = time.time() 
print('Time needed to save : {:.2f} sec'.format(
        save_end - eval_val_data_end))
print('total time from the very beginning : {:.2f} sec'.format(save_end - begin))
sys.stdout.flush()
