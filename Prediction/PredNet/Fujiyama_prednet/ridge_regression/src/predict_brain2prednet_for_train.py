import argparse
import cPickle as pickle

import os
import sys
import time
import numpy as np

from sklearn import linear_model
from sklearn.externals import joblib

args = None
parser = argparse.ArgumentParser()
parser.add_argument('--brain_path', type=str,
                    default=('/home/fujiyama/PredNet/brain_activity/cortex/'
                             'z_brain_train_preprocessed.pickle'),
                    help='Absolute path to brain file for training.')
parser.add_argument('--depth', type=str, default='R0',
                    help='R0 | R1 | R2 | R3.')
parser.add_argument('--predicted_save_path', type=str,
                    help=('Absolute path to directory to save predicted '
                          'internal representation path.'))
parser.add_argument('--model_path', type=str,
                    help='Absolute path to trained model.')

args = parser.parse_args()

begin = time.time()

depth = args.depth
predicted_save_path = args.predicted_save_path

if not os.path.exists(predicted_save_path):
    os.makedirs(predicted_save_path)

dirs = ['vtrn001/', 'vtrn002/', 'vtrn003/', 'vtrn004/', 'vtrn005/',
        'vtrn006/', 'vtrn007/', 'vtrn008/', 'vtrn009/', 'vtrn010/',
        'vtrn011/', 'vtrn012/', 'vtrn013/', 'vtrn014/', 'vtrn015/']

for i in range(len(dirs)):
    if not os.path.exists(
        predicted_save_path + depth + '/' + dirs[i]):
        os.makedirs(predicted_save_path + depth + '/' + dirs[i])

print('Loading data and trained model...')
x_data = pickle.load(open(args.brain_path, 'rb'))[:]
reg = joblib.load(args.model_path)

print('Completed loading data and trained model.')
print('Start to predict...')
sys.stdout.flush()

predicted_y = reg.predict(x_data)

for i in range(0, 4497):
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

    np.savez_compressed(predicted_save_path + depth + '/' + dir + '%08d.npz' % (frame_id),
                        R0=predicted_y[i,:].reshape((1,3,120,160)).copy())

end = time.time()
print('Elapsed time: {:.2f} sec'.format(end - begin))
sys.stdout.flush()
