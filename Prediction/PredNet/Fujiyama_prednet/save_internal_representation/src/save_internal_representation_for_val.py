"""For saving interal representations (validation) in PredNet."""
# NOTE: This code requires sequence list not image list. It is different
#       from that for trainning data. I apologize for inconsistent data
#       preparation.
# Usage:
#   python ./src/save_internal_representation_for_val.py \
#   -seq <path to stimulus-sequence-list for validation> \
#   --gpu <GPU ID> \
#   --initmodel <absolute path to your best model> \
#   --save_dir <absolute path to a directory for saving>
#
import argparse
import os
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import net_save_internal_representation_for_val

parser = argparse.ArgumentParser(description='PredNet')
parser.add_argument('--sequences', '-seq',
                    default='/stimulus/lists/val_seq_list.txt',
                    help='Path to stimulus sequence list file.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU).')
parser.add_argument('--root', '-r',
                    default='/home/fujiyama/PredictiveCoding/',
                    help='Root directory path of sequence and image files.')
parser.add_argument('--initmodel',
                    default='/home/fujiyama/PredictiveCoding/prednet/models/'
                            'best.model',
                    help='Initialize the model from given file. Please specify '
                         ' your best model in form of an absolute path.')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels).')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers.')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels).')
parser.add_argument('--save_dir',
                    default='/home/fujiyama/PredictiveCoding/'
                            'internal_representation/',
                    help='Absolute path to a directory to save internal '
                         'representations.')
args = parser.parse_args()

if not args.sequences:
    print('Please specify sequences.')
    exit()
    
if not args.save_dir:
    print('Specify a path to a directory for saving internal representations.')
    exit()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

args.size = args.size.split(',')
for i in range(len(args.size)):
    args.size[i] = int(args.size[i])
args.channels = args.channels.split(',')
for i in range(len(args.channels)):
    args.channels[i] = int(args.channels[i])
args.offset = args.offset.split(',')
for i in range(len(args.offset)):
    args.offset[i] = int(args.offset[i])

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# Create Model
prednet = net_save_internal_representation_for_val.PredNet(
    args.size[0], args.size[1], args.channels, args.save_dir)
model = L.Classifier(prednet, lossfun=mean_squared_error)
model.compute_accuracy = False

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    print('Running on a GPU.')
else:
    print('Running on a CPU.')

# Init
if args.initmodel:
    print('Load model from ', args.initmodel)
    serializers.load_npz(args.initmodel, model)

def load_list(path, root):
    tuples = []
    for line in open(root + path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples

def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    top = args.offset[1] + (image.shape[1]  - args.size[1]) / 2
    left = args.offset[0] + (image.shape[2]  - args.size[0]) / 2
    bottom = args.size[1] + top
    right = args.size[0] + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image = image[:, :, :].astype(np.float32)
    image /= 255
    return image

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)

sequencelist = load_list(args.sequences, args.root)

for seq in range(len(sequencelist)):
    imagelist = load_list(sequencelist[seq], args.root)
    prednet.reset_state() # Delete the current output and value of memory cell
    loss = 0
    batchSize = 1
    x_batch = np.ndarray((batchSize, args.channels[0],
                          args.size[1], args.size[0]), dtype=np.float32)
    y_batch = np.ndarray((batchSize, args.channels[0],
                          args.size[1], args.size[0]), dtype=np.float32)
    for i in range(0, len(imagelist)):
        x_batch[0] = read_image(imagelist[i])
        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))
        loss.unchain_backward()
        loss = 0

        # If you want to save input and predicted images, please uncomment
        # this fraction of code. Please confirm that directory exists.
        '''
        frame_id = 165 + i * 60 + seq * 3600
        print('frameNo:' + str(frame_id))
        if seq == 0:
            dir = 'vval001/'
        elif seq == 1:
            dir = 'vval002/'
        elif seq == 2:
            dir = 'vval003/'
        elif seq == 4:
            dir = 'vval004/'
        else:
            dir = 'vval005/'
        if args.gpu >= 0:model.to_cpu()
        write_image(x_batch[0].copy(),
                    'stimulus/' + dir + 'predicted/%08d_input.jpg' % (frame_id))
        write_image(model.y.data[0].copy(),
                    'stimulus/' + dir + 'predicted/%08d_predicted.jpg' % (frame_id))
        if args.gpu >= 0:model.to_gpu()
        '''
