"""For saving intenral representaions in PredNet with training stimulus images."""
#
# Usage:
#   python ./src/save_internal_representation.py \
#   --images <path to stimulus-image-list for train> \
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
import net_save_internal_representation

parser = argparse.ArgumentParser(description='PredNet')
parser.add_argument('--images', '-i',
                    default='stimulus/lists/train_list.txt',
                    help='Path to stimulus images list file.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU).')
# TODO: Update the default value.
parser.add_argument('--root', '-r',
                    default='/home/fujiyama/PredictiveCoding/',
                    help='Root directory path of sequence and image files.')
parser.add_argument('--initmodel',
                    default='/home/fujiyama/PredictiveCoding/prednet/models/'
                            'best.model',
                    help='Initialize the model from given file. Please specify '
                         'your best model in form of an absolute path.')
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

if not args.images:
    print('Please specify images.')
    exit()

if not args.save_dir:
    print('Specify the path to a directory for saving internal representations.')
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
prednet = net_save_internal_representation.PredNet(
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

sequencelist = [args.images]

seq = 0
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
    frame_id = 345 + i * 60
    print('frameNo:' + str(frame_id))
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
    if args.gpu >= 0:model.to_cpu()
    write_image(x_batch[0].copy(),
                'stimulus/' + dir + 'predicted/%08d_input.jpg' % (frame_id))
    write_image(model.y.data[0].copy(),
                'stimulus/' + dir + 'predicted/%08d_predicted.jpg' % (frame_id))
    if args.gpu >= 0:model.to_gpu()
    '''
