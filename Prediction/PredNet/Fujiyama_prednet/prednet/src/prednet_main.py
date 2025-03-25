'''Prednet model implementation in Chainer.'''
#
# Usage:
#   python ./src/prednet_main.py \
#   -seq <path to train-sequence-list> \
#   --gpu <GPU ID> \
#   --lr <alpha value for Adam optimizer> \
#   --image_save_dir <path to image dir> \
#   --model_save_dir <path to model dir> \
#   --epochs <# epochs to train> \
#   --val_seq <path to val-sequence-list>
#
import argparse
import os
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import net

### argumentparse はコマンドのときの引数を指定できる
parser = argparse.ArgumentParser(description='PredNet')
parser.add_argument('--images', '-i', default='', help='Path to image list file.')
parser.add_argument('--sequences', '-seq',
                    default='data/lists/train_seq_list.txt',
                    help='Path to sequence list file for training.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU).')
parser.add_argument('--root', '-r',
                    default='/home/fujiyama/PredictiveCoding/prednet/',
                    help='Root directory path for dataset and models.')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file.')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot.')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels).')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers.')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels).')
parser.add_argument('--ext', '-e', default=100, type=int,
                    help='Extended prediction on test (frames).')
parser.add_argument('--bprop', default=10, type=int,
                    help='Back propagation length (frames).')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate.')
parser.add_argument('--save', default=5000, type=int,
                    help='Period of saving model and state (frames). '
                         'This equals to the number of frames in an epoch.')
parser.add_argument('--image_save_dir',
                    default='images/',
                    help='Path to directory to save input, predicted and '
                         'reference images.')
parser.add_argument('--model_save_dir',
                    default='models/',
                    help='Path to directory to save trained models.')
parser.add_argument('--epochs', default=150, type=int,
                    help='Number of epochs to train.')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--val_seq',default='data/lists/val_seq_list.txt',
                    help='Path to sequence list file for validation.')
parser.set_defaults(test=False)
args = parser.parse_args()

if (not args.images) and (not args.sequences):
    print('Please specify images or sequences')
    exit()

### ここまでをTD-VAEにいれるとコマンドでGPU指定ができるようになる気がする

args.size = args.size.split(',')
for i in range(len(args.size)):
    args.size[i] = int(args.size[i])
args.channels = args.channels.split(',')
for i in range(len(args.channels)):
    args.channels[i] = int(args.channels[i])
args.offset = args.offset.split(',')
for i in range(len(args.offset)):
    args.offset[i] = int(args.offset[i])

# Device setting
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# Create Model
prednet = net.PredNet(args.size[0], args.size[1], args.channels)
model = L.Classifier(prednet, lossfun=mean_squared_error)
model.compute_accuracy = False
optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    print('Running on a GPU.')
else:
    print('Running on a CPU.')

# Init/Resume
if args.initmodel:
    print('Load model from ', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from ', args.resume)
    serializers.load_npz(args.resume, optimizer)

if not os.path.exists(args.root + args.image_save_dir):
    os.makedirs(args.root + args.image_save_dir)
if not os.path.exists(args.root + args.model_save_dir):
    os.makedirs(args.root + args.model_save_dir)

def load_list(path, root):
    tuples = []
    for line in open(root + path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples

def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    image = image[:, :, :].astype(np.float32)
    # This fraction of code is necessary if no preprocessings are done.
    '''
    top = args.offset[1] + (image.shape[1]  - args.size[1]) / 2
    left = args.offset[0] + (image.shape[2]  - args.size[0]) / 2
    bottom = args.size[1] + top
    right = args.size[0] + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    '''
    image /= 255
    return image

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)

if args.images:
    sequencelist = [args.images]
else:
    sequencelist = load_list(args.sequences, args.root)

if args.test == True:
    for seq in range(len(sequencelist)):
        imagelist = load_list(sequencelist[seq], args.root)
        prednet.reset_state()
        loss = 0
        batchSize = 1
        x_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]),
                             dtype=np.float32)
        y_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]),
                             dtype=np.float32)
        for i in range(0, len(imagelist)):
            # print('frameNo: ' + str(i))
            x_batch[0] = read_image(imagelist[i])
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            loss.unchain_backward()
            loss = 0
            if args.gpu >= 0:model.to_cpu()
            write_image(x_batch[0].copy(),
                        args.root + args.image_save_dir + 'test_' + str(i) + '_x.jpg')
            write_image(model.y.data[0].copy(),
                        args.root + args.image_save_dir + 'test_' + str(i) + '_y.jpg')
            if args.gpu >= 0:model.to_gpu()

        if args.gpu >= 0:model.to_cpu()
        x_batch[0] = model.y.data[0].copy()
        if args.gpu >= 0:model.to_gpu()
        for i in range(len(imagelist), len(imagelist) + args.ext):
            print('extended frameNo: ' + str(i))
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            loss.unchain_backward()
            loss = 0
            if args.gpu >= 0:model.to_cpu()
            write_image(model.y.data[0].copy(),
                        args.root + args.image_save_dir + 'test_' + str(i) + '_y.jpg')
            x_batch[0] = model.y.data[0].copy()
            if args.gpu >= 0:model.to_gpu()

else:
    val_seq_list = load_list(args.val_seq, args.root)
    nth_epoch = 0
    nth_train_seq = 0
    train_seq = 0
    val_seq = len(val_seq_list)
    count = 0

    while nth_epoch < args.epochs:
        print('epoch : ' + str(nth_epoch))
        imagelist = load_list(sequencelist[train_seq], args.root)
        prednet.reset_state()
        loss = 0
        batchSize = 1
        x_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]),
                             dtype=np.float32)
        y_batch = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]),
                             dtype=np.float32)
        x_batch[0] = read_image(imagelist[0]);
        for i in range(1, len(imagelist)):
            y_batch[0] = read_image(imagelist[i]); # correct prediction
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            #print('frameNo: ' + str(i))
            if i % args.bprop == 0:
                nth_train_seq += 1
                # print('backpropagation@' + str(nth_train_seq) + ' th sequence')
                model.zerograds()
                loss.backward()
                loss.unchain_backward()
                loss = 0
                optimizer.update()
                # If you want to save (input[x], model prediction[y], correct prediction[z])
                # triple at each backpropagation step, please uncomment this fraction of code.
                '''
                if args.gpu >= 0:model.to_cpu()
                write_image(x_batch[0].copy(),
                            args.root + args.image_save_dir + str(count) + 'th_image_x.jpg')
                write_image(model.y.data[0].copy(),
                            args.root + args.image_save_dir + str(count) + '_th_image_y.jpg')
                write_image(y_batch[0].copy(),
                            args.root + args.image_save_dir + str(count) + '_th_image_z.jpg')
                if args.gpu >= 0:model.to_gpu()
                '''
                print('training_loss: ' + str(float(model.loss.data)))

            # epoch-based learning
            if count % args.save == 0:
                print('Save the model, and the optimizer.')
                serializers.save_npz(
                    args.root + args.model_save_dir + str(nth_epoch) + '.model',
                    model)
                serializers.save_npz(
                    args.root + args.model_save_dir + str(nth_epoch) + '.state',
                    optimizer)

                # validation
                x_batch_val = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]),
                                         dtype=np.float32)
                y_batch_val = np.ndarray((batchSize, args.channels[0], args.size[1], args.size[0]),
                                         dtype=np.float32)
                evaluator = model.copy()
                evaluator.predictor.reset_state()
                loss_val = 0
                print('validation @ ' + str(nth_epoch) + ' th epoch')
                val_losses = []
                for nth_val_seq in range(0, val_seq):
                    val_imagelist = load_list(val_seq_list[nth_val_seq], args.root)
                    x_batch_val[0] = read_image(val_imagelist[0])
                    for j in range(1, len(val_imagelist)):
                        y_batch_val[0] = read_image(val_imagelist[j])
                        loss_val += evaluator(chainer.Variable(xp.asarray(x_batch_val), volatile='on'),
                                              chainer.Variable(xp.asarray(y_batch_val), volatile='on')).data
                        if (j % 10) == 0:
                            #print('loss summed up over 10 frames : ' + str(loss_val))
                            val_losses.append(loss_val)
                            #loss_val.unchain_backward()
                            loss_val = 0
                        x_batch_val[0] = y_batch_val[0]

                average = sum(val_losses) / len(val_losses)
                print('validation_loss @ ' +  str(nth_epoch) + ' th epoch: ' + str(average))

                nth_epoch += 1
                print('incremented nth_epoch. next epoch : ' + str(nth_epoch))

            x_batch[0] = y_batch[0]
            count += 1

        train_seq = (train_seq + 1)%len(sequencelist)
