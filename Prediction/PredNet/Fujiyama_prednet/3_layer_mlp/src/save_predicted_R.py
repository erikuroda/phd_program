"""For predicting internal representation given brain activity, and save the predictions."""
#  Usage:
#    python ./src/save_predicted_R.py \
#    --initmodel <absolute path to the best model file> \
#    --gpu <GPU ID> \
#    --depth <R0 | R1 | R2 | R3> \
#    --brain_test_path <absolute path to brain test file> \
#    --internal_representation_path <absolute path to a directory to save predicted results>
#
from __future__ import print_function
import argparse
import cPickle as pickle
import time

import os
import numpy as np
import six
import csv

import chainer
from chainer import computational_graph
from  chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import chainer.functions as F

import mlp_net

parser = argparse.ArgumentParser(description='3-layer perceptron')
parser.add_argument('--initmodel', '-m',
                    default='/home/fujiyama/PredictiveCoding/3_layer_mlp/'
                            'models/best.model',
                    help='Initialize the model from given file. '
                         'Please specify absolute path to your best model.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID(negative value indicates CPU).')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='Number of units.')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='Learning minibatch size.')
parser.add_argument('--depth', '-d', default='R0',
                    help='Depth of Representation module(y) in PredNet '
                         '(R0 | R1 | R2 | R3). R0 indicates the input layer.')
parser.add_argument('--brain_test_path',
                    default='/home/fujiyama/PredictiveCoding/brain_activity/cortex/'
                            'z_brain_val_preprocessed.pickle',
                    help='Path to the brain activity file for prediction input.'
                         'Brain activity is expected to be a pickle formatted file.')
parser.add_argument('--internal_representation_path',
                    default='/home/fujiyama/PredictiveCoding/'
                            'predicted_internal_representation/mlp/',
                    help='Path to the directory to save predicted internal representation.')
args = parser.parse_args()

batchsize = args.batchsize
n_units = args.unit
depth = args.depth
N_test = 300
dim_brain = 65665
# dim_table: (# dimensions of internal representaion of each layer in PredNet)
dim_table = {"R0":57600, "R1":230400, "R2":115200, "R3":57600}
dim_internal_representation = dim_table[depth]

brain_test_path = args.brain_test_path
internal_representation_path = args.internal_representation_path + depth + '/'

print('Predict {} internal representation...'.format(depth))
print('GPU : {}'.format(args.gpu))
print('# Unit: {}'.format(n_units))
print('# Minibatch-size: {}'.format(batchsize))
print('# depth: {}'.format(depth))
print('# Brain for prediction: {}'.format(brain_test_path))
print('# Internal representation: {}'.format(internal_representation_path))
print('')

if not os.path.exists(internal_representation_path):
    os.makedirs(internal_representation_path)

dirs = ['vval001/', 'vval002/', 'vval003/', 'vval004/', 'vval005/']
for i in range(len(dirs)):
    if not os.path.exists(internal_representation_path + dirs[i]):
        os.makedirs(internal_representation_path + dirs[i])

data_load_begin = time.time()

# Prepare dataset
print('Load brain data.')
x_test = pickle.load(open(brain_test_path, 'rb'))[:]
print('shape of x_test (expected : (300, 65665)) => ' + str(x_test.shape))
print('type of x_test (exprected : (numpy.ndarray)) => ' + str(type(x_test)))
print('type of x_test[0,0] (expected : (numpy.float32)) => ' + str(type(x_test[0,0])))

data_load_end = time.time()
print('Time needed to load data = {} sec'.format(
        data_load_end - data_load_begin))

# Setup model
model = L.Classifier(
    mlp_net.Regression(dim_brain, n_units, dim_internal_representation),
    lossfun=F.mean_squared_error)
model.compute_accuracy = False

# Device setting
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Init
if args.initmodel:
    print('Load model from ', args.initmodel)
    serializers.load_npz(args.initmodel, model)

start = time.time()

# Pass val data
# Test Loop with minibatch
for i in six.moves.range(0, N_test, batchsize):
    frame_id = 165 + i * 60
    x = chainer.Variable(xp.asarray(x_test[i:i+batchsize]),
                         volatile = 'on')
    predict = model.predictor(x)

    if frame_id <= 3705:
        dir = dirs[0]
    elif 3705 < frame_id <= 7305:
        dir = dirs[1]
    elif 7305 < frame_id <= 10905:
        dir = dirs[2]
    elif 10905 < frame_id <= 14505:
        dir = dirs[3]
    else:
        dir = dirs[4]

    if depth == 'R0':
        np.savez_compressed(internal_representation_path + dir + '%08d.npz' % (frame_id),
                            R0=predict.data.reshape((1,3,120,160)))
    elif depth == 'R2':
        np.savez_compressed(internal_representation_path + dir + '%08d.npz' % (frame_id),
                            R2=predict.data.reshape((1,96,30,40)))
    elif depth == 'R3':
        np.savez_compressed(internal_representation_path + dir + '%08d.npz' % (frame_id),
                            R3=predict.data.reshape((1,192,15,20)))

    print('saved : ' + internal_representation_path + dir + '%08d.npz' % (frame_id))

    '''
    print('type(predict.data) => ' + str(type(predict.data)))
    print('predict.label => ' + str(predict.label))
    print('Can reshape?' + str(predict.data.reshape((1,3,120,160)).shape))
    print(predict.data)
    '''

# Compute throughput
end = time.time()
elapsed_time = end - start
throughput = N_test / elapsed_time
# Printout
print('throughput = {} samples/sec'.format(throughput))
