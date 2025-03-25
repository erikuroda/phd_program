#!usr/bin/python
#-*- coding: utf-8 -*-
"""Calculate Pearson's correlation coefficient between predicted and reference
   internal representations on train data."""
#
# Usage:
#   python calc_correlation.py \
#   --depth <R0| R1| R2| R3> \
#   --reference_path <absolute path to reference internal representation directory> \
#   --predicted_path <absolute path to predicted internal representation directory>
#
import argparse
import math
import numpy as np
import os
import time

parser = argparse.ArgumentParser(description='Correlation')
# TODO: Confirm arguments.
parser.add_argument('--depth', default='R0',
                    help='Depth of representation module in PredNet.')
parser.add_argument('--reference_path',
                    default=('/Users/fujiyamachihiro/PredictiveCoding/data/'
                             'internal_representation/paired/'),
                    help='Path to reference representation directory.')
parser.add_argument('--predicted_path',
                    default=('/Users/fujiyamachihiro/PredictiveCoding/data/'
                             'predicted_internal_representation/ridge/'),
                    help='Path to predicted representation directory.')
args = parser.parse_args()

depth = args.depth
dim_table = {"R0":57600, "R1":230400, "R2":115200, "R3":57600}
dim_internal_representation = dim_table[depth]

reference_path = args.reference_path
predicted_path = os.path.join(args.predicted_path, depth)

begin = time.time()

reference_internal_representations = np.ndarray((dim_internal_representation, 4497),
                                                dtype = np.float32)
predicted_internal_representations = np.ndarray((dim_internal_representation, 4497),
                                                dtype = np.float32)

dirs = ['vtrn001', 'vtrn002', 'vtrn003', 'vtrn004', 'vtrn005',
        'vtrn006', 'vtrn007', 'vtrn008', 'vtrn009', 'vtrn010',
        'vtrn011', 'vtrn012', 'vtrn013', 'vtrn014', 'vtrn015']

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

    ref_internal_representation = np.load(reference_path + dir + '%d.npz' % (frame_id))
    reference_internal_representations[:, i] = (
        ref_internal_representation[depth].reshape((dim_internal_representation,)))
    ref_internal_representation.close()
    predicted_internal_representation = np.load(predicted_path + '/' + dir + '%08d.npz' % (frame_id))
    predicted_internal_representations[:, i] = (
        predicted_internal_representation[depth].reshape((dim_internal_representation,)))
    predicted_internal_representation.close()

data_loading_end = time.time()
print('elapsed time needed to load data: {:.2f} sec'.format(data_loading_end - begin))

# Calculate correlation.
## Calculate average vectors.
### reference internal representation.
average_of_ref = np.zeros(reference_internal_representations.shape[0])
for i in range(reference_internal_representations.shape[1]):
    average_of_ref += reference_internal_representations[:,i]
average_of_ref /= float(reference_internal_representations.shape[1])
 
### predicted internal representation.
average_of_predicted = np.zeros(predicted_internal_representations.shape[0])
for i in range(predicted_internal_representations.shape[1]):
    average_of_predicted += predicted_internal_representations[:,i]
average_of_predicted /= float(predicted_internal_representations.shape[1])

## Calculate numerator.
numerator = 0.0
for i in range(predicted_internal_representations.shape[1]):
    numerator += np.dot(
        (reference_internal_representations[:,i] - average_of_ref), 
        (predicted_internal_representations[:,i] - average_of_predicted))
numerator /= float(predicted_internal_representations.shape[1])
print('numerator : ' + str(numerator))

## Calculate denominator.
### Calculate squeared averages.
squared_average_of_ref = 0.0
squared_average_of_predicted = 0.0
for i in range(predicted_internal_representations.shape[1]):
    squared_average_of_ref += np.dot(
        (reference_internal_representations[:,i] - average_of_ref),
        (reference_internal_representations[:,i] - average_of_ref))
    squared_average_of_predicted += np.dot(
        (predicted_internal_representations[:,i] - average_of_predicted),
        (predicted_internal_representations[:,i] - average_of_predicted))
squared_average_of_ref /= float(reference_internal_representations.shape[1])
squared_average_of_predicted /= float(predicted_internal_representations.shape[1])

denominator = math.sqrt(squared_average_of_ref *
                        squared_average_of_predicted)
print('denominator : ' + str(denominator))

correlation = float(numerator) / denominator

calculation_end = time.time()
print('elapsed time needed to calculate correlation : {:.2f} sec'.format(
        calculation_end - data_loading_end))
print('elapsed time : {:.2f} sec'.format(calculation_end - begin))
print('correlation : ' +str(correlation))   
