import argparse
import math
import numpy as np
import time

parser = argparse.ArgumentParser(description='Correlation')
parser.add_argument('--depth', default='R0',
                    help='Depth of representation module in PredNet.')
parser.add_argument('--reference_path',
                    default='/home/kuroda/prednet/internal_representation_2020/',
                    help='Path to reference representation directory.')
parser.add_argument('--predicted_path',
                    default='/home/kuroda/prednet/'
                            'predicted_internal_representation_2020/R0_alpha0.5/',
                    help='Path to predicted representation directory.')
args = parser.parse_args()

depth = args.depth
dim_table = {"R0":57600, "R1":230400, "R2":115200, "R3":57600}
dim_internal_representation = dim_table[depth]

reference_path = args.reference_path
predicted_path = args.predicted_path

begin = time.time()

reference_internal_representations = np.ndarray((dim_internal_representation, 300),
                                                dtype = np.float32)
predicted_internal_representations = np.ndarray((dim_internal_representation, 300),
                                                dtype = np.float32)
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
    ref_internal_representation = np.load(reference_path + dir + '%08d.npz' % (frame_id))
    reference_internal_representations[:, i] = \
        ref_internal_representation[depth].reshape((dim_internal_representation,))
    ref_internal_representation.close()
    predicted_internal_representation = np.load(predicted_path + dir + '%08d.npz' % (frame_id))
    predicted_internal_representations[:, i] = \
        predicted_internal_representation[depth].reshape((dim_internal_representation,))
    predicted_internal_representation.close()

data_loading_end = time.time()
print('elapsed time needed to load data: {:.2f} sec'.format(data_loading_end - begin))

average_of_ref = np.zeros(reference_internal_representations.shape[0])
for i in range(reference_internal_representations.shape[1]):
    average_of_ref += reference_internal_representations[:,i]
average_of_ref /= float(reference_internal_representations.shape[1])

average_of_predicted = np.zeros(predicted_internal_representations.shape[0])
for i in range(predicted_internal_representations.shape[1]):
    average_of_predicted += predicted_internal_representations[:,i]
average_of_predicted /= float(predicted_internal_representations.shape[1])

numerator = 0.0
for i in range(predicted_internal_representations.shape[1]):
    numerator += np.dot(
        (reference_internal_representations[:,i] - average_of_ref),
        (predicted_internal_representations[:,i] - average_of_predicted))
numerator /= float(predicted_internal_representations.shape[1])
print('numerator : ' + str(numerator))

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
