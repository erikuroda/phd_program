"""For decoding predictive images from estimated internal representations R3."""
#
# Usage:
#   python decode_internal_representation_R3.py \
#   --sequences <path to sequence list file> \ 
#   --gpu <GPU ID> \
#   --initmodel <absolute path to your best model> \
#   --internal_representation_path <absolute path to predicted internal representation> \
#   --save_dir <absolute path to a directory to save decoded images>
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
import net_decode_internal_representation_R3

parser = argparse.ArgumentParser(description='PredNet')
parser.add_argument('--sequences', '-seq',
                    default='stimulus/lists/val_seq_list.txt',
                    help='Path to stimulus sequence list file.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU).')
parser.add_argument('--root', '-r', default='/home/fujiyama/PredictiveCoding/',
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
parser.add_argument('--internal_representation_path',
                    default='/home/fujiyama/PredictiveCoding/'
                            'predicted_internal_representation/ridge/R3_alpha0.5/',
                    help='Absolute path to a directory which contains predicted '
                         'internal representation to insert into PredNet.')
parser.add_argument('--save_dir',
                    default='/home/fujiyama/PredictiveCoding/from_brain/R3/',
                    help='Absolute path to a directory to save decoded '
                         'predictive images from brain activity.')
args = parser.parse_args()

if not args.sequences:
    print('Please specify sequences.')
    exit()
    
if not args.internal_representation_path:
    print('Specify the path to internal representation to insert into PredNet.')
    exit()

if not args.save_dir:
    print('Specify the path to a directory for saving decoded images.')
    exit()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

dirs = ['vval001/', 'vval002/', 'vval003/', 'vval004/', 'vval005/']
for i in range(len(dirs)):
    if not os.path.exists(args.save_dir + dirs[i]):
        os.makedirs(args.save_dir + dirs[i])

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
prednet = net_decode_internal_representation_R3.PredNet(
    args.size[0], args.size[1], args.channels, args.internal_representation_path)
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
batchSize = 1

for seq in range(len(sequencelist)):
    imagelist = load_list(sequencelist[seq], args.root)
    prednet.reset_state() # Delete the current output and value of memory cell
    loss = 0
    x_batch = np.ndarray((batchSize, args.channels[0],
                          args.size[1], args.size[0]), dtype=np.float32)
    y_batch = np.ndarray((batchSize, args.channels[0],
                          args.size[1], args.size[0]), dtype=np.float32)
    for i in range(0, len(imagelist)):
        frame_id = 165 + i * 60 + seq * 3600
        # print('frameNo:' + str(frame_id))
        if seq == 0:
            dir = dirs[seq]
        elif seq == 1:
            dir = dirs[seq]
        elif seq == 2:
            dir = dirs[seq]
        elif seq == 3:
            dir = dirs[seq]
        else:
            dir = dirs[seq]
        x_batch[0] = read_image(imagelist[i])
        loss += model(chainer.Variable(xp.asarray(x_batch)),
                      chainer.Variable(xp.asarray(y_batch)))
        loss.unchain_backward()
        loss = 0
        if args.gpu >= 0:model.to_cpu()
        # If you want to save the input images as well, please uncomment.
        '''
        write_image(x_batch[0].copy(),
                    args.save_dir + dir + '%08d_stimulus.jpg' % (frame_id))
        '''
        write_image(model.y.data[0].copy(),
                    args.save_dir + dir + '%08d_decoded.jpg' % (frame_id))
        if args.gpu >= 0:model.to_gpu()

