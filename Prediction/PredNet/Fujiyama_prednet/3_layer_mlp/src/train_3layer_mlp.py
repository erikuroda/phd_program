"""For training 3-layer perceptron."""
# Usage:
#   python ./src/train_3layer_mlp.py \
#   --gpu <GPU ID> \
#   --depth <R0 | R1 | R2 | R3> \
#   --brain_train_path <absolute path to brain train file> \
#   --brain_test_path <absolute path to brain test file> \
#   --internal_representation_path <absolute path to ineternal representaion directory> \
#   --save_path <absolute path to directory to save trained models>
#
from __future__ import print_function
import argparse
import cPickle as pickle
import os
import time

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
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file.')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot.')
parser.add_argument('--optimizer', '-o',
                    choices=('SGD', 'MomentumSGD', 'NesterovAG', 'Adagrad',
                             'Adadelta','RMSprop','Adam'),
                    default='SGD', help='Optimization algorithm.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID(negative value indicates CPU).')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='Number of epochs to learn.')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='Number of units.')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Learning minibatch size.')
parser.add_argument('--depth', '-d', default='',
                    help='Depth of Representation module(y) in PredNet '
                         '(R0 | R1 | R2 | R3 ). R0 indicates the input layer.')
parser.add_argument('--brain_train_path',
                    default='/home/fujiyama/PredictiveCoding/brain_activity/cortex/'
                            'z_brain_train_preprocessed.pickle',
                    help='Path to the brain activity file for training.'
                         'Brain activity is expected to be a pickle formatted file.')
parser.add_argument('--brain_test_path',
                    default='/home/fujiyama/PredictiveCoding/brain_activity/cortex/'
                            'z_brain_val_preprocessed.pickle',
                    help='Path to the brain activity file for test.'
                         'Brain activity is expected to be a pickle formatted file.')
parser.add_argument('--internal_representation_path',
                    default='/home/fujiyama/PredictiveCoding/internal_representation/',
                    help='Path to the internal representation directory.')
parser.add_argument('--save_path', default='/home/fujiyama/PredictiveCoding/3_layer_mlp/models/',
                    help='Path to the directory to save trained model, optimizer, '
                         'and csv file to output test losses.')
args = parser.parse_args()

batchsize = args.batchsize
n_epochs = args.epoch
n_units = args.unit
depth = args.depth
N_train = 4497 # (# train examples)
N_test = 300 # (# test examples)
dim_brain = 65665
# dim_table: (# dimensions of internal representaion of each layer in PredNet)
dim_table = {"R0":57600, "R1":230400, "R2":115200, "R3":57600}
dim_internal_representation = dim_table[depth]

brain_train_path = args.brain_train_path
brain_test_path = args.brain_test_path
internal_representation_path = args.internal_representation_path
save_path = args.save_path

print('GPU : {}'.format(args.gpu))
print('# Unit: {}'.format(n_units))
print('# Minibatch-size: {}'.format(batchsize))
print('# Epoch: {}'.format(n_epochs))
print('# Depth: {}'.format(depth))
print('# Brain for train: {}'.format(brain_train_path))
print('# Brain for test: {}'.format(brain_test_path))
print('# Internal representation: {}'.format(internal_representation_path))
print('')

if not os.path.exists(save_path):
    os.makedirs(save_path)

data_load_begin = time.time()

# Prepare dataset
print('Load brain data.')
x_train = pickle.load(open(brain_train_path, 'rb'))[:]
x_test = pickle.load(open(brain_test_path, 'rb'))[:]
print('Shape of x_train (expected : (4497, 65665)) => ' + str(x_train.shape))
print('Type of x_train (exprected : (numpy.ndarray)) => ' + str(type(x_train)))
print('Type of x_train[0,0] (expected : (numpy.float32)) => ' + str(type(x_train[0,0])))
print('Shape of x_test (expected : (300, 65665)) => ' + str(x_test.shape))
print('Type of x_test (exprected : (numpy.ndarray)) => ' + str(type(x_test)))
print('Type of x_test[0,0] (expected : (numpy.float32)) => ' + str(type(x_test[0,0])))

# Prepare internal representation
print('Load internal representation data.')
y_train = np.ndarray((N_train, dim_internal_representation), dtype = np.float32)
for i in range(0, N_train):
    frame_id = 345 + i * 60
    if frame_id <= 18105:
        dir = 'vtrn001/'
    elif 18105 < frame_id <= 36105:
        dir = 'vtrn002/'
    elif 36105 < frame_id <= 54105:
        dir = 'vtrn003/'
    elif 54105 < frame_id <= 72105:
        dir = 'vtrn004/'
    elif 72105 < frame_id <= 90105:
        dir = 'vtrn005/'
    elif 90105 < frame_id <= 108105:
        dir = 'vtrn006/'
    elif 108105 < frame_id <= 126105:
        dir = 'vtrn007/'
    elif 126105 < frame_id <= 144105:
        dir = 'vtrn008/'
    elif 144105 < frame_id <= 162105:
        dir = 'vtrn009/'
    elif 162105 < frame_id <= 180105:
        dir = 'vtrn010/'
    elif 180105 < frame_id <= 198105:
        dir = 'vtrn011/'
    elif 198105 < frame_id <= 216105:
        dir = 'vtrn012/'
    elif 216105 < frame_id <= 234105:
        dir = 'vtrn013/'
    elif 234105 < frame_id <= 252105:
        dir = 'vtrn014/'
    else:
        dir = 'vtrn015/'
    internal_representation = np.load(
        internal_representation_path + dir + '%08d.npz' % (frame_id))
    y_train[i,:] = internal_representation[depth].reshape((dim_internal_representation,))
    internal_representation.close()

y_test = np.ndarray((N_test, dim_internal_representation), dtype = np.float32)
for i in range(0, N_test):
    frame_id = 165 + i * 60
    if frame_id <= 3705:
        dir = 'vval001/'
    elif 3705 < frame_id <= 7305:
        dir = 'vval002/'
    elif 7305 < frame_id <= 10905:
        dir = 'vval003/'
    elif 10905 < frame_id <= 14505:
        dir = 'vval004/'
    else:
        dir = 'vval005/'
    internal_representation = np.load(
        internal_representation_path + dir + '%08d.npz' % (frame_id))
    y_test[i,:] = internal_representation[depth].reshape((dim_internal_representation,))
    internal_representation.close()

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

# Setup optimizer
if args.optimizer == 'SGD':
    optimizer = optimizers.SGD()
elif args.optimizer == 'MomentumSGD':
    optimizer = optimizers.MomentumSGD
elif args.optimizer == 'NesterovAG':
    optimizer = optimizers.NesterovAG()
elif args.optimizer == 'Adagrad':
    optimizer = optimizers.AdaGrad()
elif args.optimizer == 'Adadelta':
    optimizer = optimizers.AdaDelta()
elif args.optimizer == 'RMSprop':
    optimizer = optimizers.RMSprop()
elif args.optimizer == 'Adam':
    optimizer = optimizers.Adam()
    
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from ', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from ', args.resume)
    serializers.load_npz(args.resume, optimizer)

# Training loop with epoch
test_loss = []
for epoch in six.moves.range(1, n_epochs + 1):
    print('Epoch: ', epoch)

    # Training Loop with minibatch
    perm = np.random.permutation(N_train)
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N_train, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i+batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i+batchsize]]))

        # Pass the loss function 
        optimizer.update(model, x, t)

        # Save network architecture
        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
                                  'style': 'filled'}
                function_style = {'shape': 'record', 'fillcolor': '#6495ED',
                                  'style': 'filled'}
                g = computational_graph.build_computational_graph(
                    (model.loss, ),
                    variable_style=variable_style,
                    function_style=function_style)
                o.write(g.dump())
            print('graph generated')

        # Compute loss
        current_loss = float(model.loss.data) * len(t.data)
        #print('current_loss = {}'.format(current_loss))
        sum_loss += current_loss

    # Compute throughput
    end = time.time()
    elapsed_time = end - start
    throughput = (N_train - 97) / elapsed_time
    train_mean_loss = sum_loss / (N_train - 97)

    # Printout
    print('Train mean loss = {} / sample, throughput = {} samples/sec'.format(
            train_mean_loss, throughput))

    # Test loss and accuracy
    sum_loss = 0

    # Test Loop with minibatch
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i+batchsize]),
                             volatile = 'on')
        t = chainer.Variable(xp.asarray(y_test[i:i+batchsize]),
                             volatile = 'on')

        loss = model(x, t)
        current_test_loss = float(loss.data) * len(t.data)
        #print('current_test_loss = {}'.format(current_test_loss))
        sum_loss += current_test_loss

    # Printout
    test_mean_loss = sum_loss / N_test
    print('test mean loss = {} / sample'.format(test_mean_loss))

    # Record test loss
    test_loss.append([test_mean_loss])

    # Save model, and optimizer
    print('Save the model')
    fname = save_path + depth + '_' + args.optimizer + '_' + str(epoch) + '.model'
    serializers.save_npz(fname, model)
    
    print('Save the optimizer')
    fname = save_path + depth + '_' + args.optimizer + '_' + str(epoch) + '.state'
    serializers.save_npz(fname, optimizer)

# Save the losses
print('Save the losses')
fname = save_path + depth + '_' + args.optimizer + '_loss.csv'
f = open(fname, 'w')
writer = csv.writer(f,lineterminator = '\n')
writer.writerow(['loss'])
writer.writerows(test_loss)
f.close()
