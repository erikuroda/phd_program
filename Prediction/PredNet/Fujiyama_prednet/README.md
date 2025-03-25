# PredictiveCoding

## Overview
This repository contains code for research on a correlation between internal representations in a predictive neural network ([PredNet](https://coxlab.github.io/prednet/)) and brain activity evoked by natural movies.

## Description
PredNet (Lotter et al., 2017) was introduced being inspired by the concept of predictive coding from the neuroscience literature. With the code here, we explored a correlation between internal representaions in PredNet and brain activity.

### Directories
#### prednet/
##### prednet/src/

For training a prednet model with natural images (= YouTube) dataset.

- prednet_main.py: Run this to train.
- net.py: PredNet architecture definition.

##### prednet/data/lists/

Sample sequence list and image lists.

- sample_seq_list.txt: A sample sequence list.
- sample_image0000000[1-3]_list.txt: Sample image lists.

     Here we expect that images (, %8d.jpg(s)) extracted from 0000000[1-3].avi exist in PredictiveCoding/prednet/data/images/image0000000[1-3]/. 
     Please see usage below.

#### save\_internal\_representation/src/

For saving internal representations with stimulus images.

- save_internal_representation.py: Run this to save internal representations for training stimulus images.
- net_save_internal_representation.py: Prednet archtecture definition for saving internal representations for training stimulus images.
- save_internal_representation_for_val.py: Run this to save internal representations for validation stimulus images.
- net_save_internal_representation_for_val.py: Prednet archtecture definition for saving internal representations for validation stimulus images.

#### ridge\_regression/src/

For training a ridge regression which predicts internal representaion given brain activity.

- train_brain2prednet_ridge.py: Run this to train, and make predictions for test data.

#### 3\_layer\_mlp/src/

- train_3layer_mlp.py: Run this to train.
- mlp_net.py: 3-layer NN architecture definition.
- save_predicted_R.py: Run the best 3-layer NN and save predicted internal representation.

#### brain\_decoder/
##### brain\_decoder/src/fromR0/

- decode_internal_representation_R0.py: Run this to decode predictive images from estimated R0. This is for validation data.
- net_decode_internal_representation_R0.py: Network architecture for decoding predictive images from estimated R0.

##### brain\_decoder/src/fromR2/

- decode_internal_representation_R2.py: Run this to decode predictive images using estimated R2. This is for validation data.
- net_decode_internal_representation_R2.py: Network architecture for decoding predictive images using estimated R2.

##### brain\_decoder/src/fromR3/

- decode_internal_representation_R3.py: Run this to decode predictive images using estimated R3. This is for validation data.
- net_decode_internal_representation_R3.py: Network architecture for decoding predictive images using estimated R3.

#### eval/
##### eval/quantitative\_eval/Pearon/

- calc_correlation.py: Run this to calculate Pearson's correlation coefficient between reference internal representation and predicted internal representation.

##### eval/quantitative\_eval/MSE/

- MSE.py: Run this to calculate the average of MSEs between 2 frames in a row.
- MSE_Lv1_seq.py: Run this to compare the averages of MSEs between (input, correct) pairs and (predict, correct) pairs in a specific sequence. If (input, correct) > (predict, correct), learned model can generate better predictive images rather than reconstructiong the last input frames.
- MSE_Lv2_seq.py: Run this to calculate the averages of MSEs between (input, predict) pairs and (predict, correct) pairs in a specific sequence. If (input, predict) > (predict, correct), learned model can generate predictive frames which are closer to the correct frames than they are to the last input frames.
- description.txt: Detailed description of programs in this directory.

 
## Usage
**1.Prepare dataset.**

1.1. Extract still images from videos with ffmpeg. For extracting images from a video, run (please see [ffmpeg pages](https://www.ffmpeg.org/ffmpeg.html) in detail.):

```
$ ffmpeg -i foo.avi -r <sampling rate(fps)> -s WxH -ss <starting point(sec)> -t <stop point(sec)> -f image2 foo-%08d.jpg
```

1.2. Prepare still image lists. Each still image list contains paths to extracted still images from a video in order of time. Sample image lists are available at PredictiveCoding/prednet/data/lists/sample_image0000000[1-3]_list.txt.

1.3. Prepare a sequence list which contains paths to still image lists. Sample sequence list is available at PredictiveCoding/prednet/data/lists/sample_seq_list.txt.
* Please confirm that each image exists at its indicated point.

**2.Train a prednet model with the dataset.**

2.1. Run the following command:

```
$ python ./src/prednet_main.py -seq <path to train-seq-list> --gpu <GPU ID> --lr <alpha value for Adam optimizer> --image_save_dir <path to image dir> --model_save_dir <path to model dir> --epochs <the number of epochs to train> --val_seq <path to val-seq-list>
```

**3.Prepare brain activity dataset.**

3.1. Prepare brain activity. Convert brain activity files in MAT format into pickle format. A converter code is in HDD. I can give you preprocessed brain activity files in pickle format personally.

3.2. Prepare stimulus images. Extract images from stimulus videos. 

3.3. Prepare an image list for training data, and a sequence list for validation data. (I apologize for inconsistent data preparation. Feel free to modify code for consistency.)

**4.Run the best prednet model with stimulus images and save the internal representations.**

4.1. For training data, run the following commad:

```
$ python ./src/save_internal_representation.py --images <path to stimulus-image-list> --gpu <GPU ID> --initmodel <absolute path to your best model> --save_dir <absolute path to a directory to save>
```

4.2. For validation data, run the following command:

```
$ python ./src/save_internal_representation_for_val.py -seq <path to sequence list> --gpu <GPU ID> --initmodel <absolute path to your best model> --save_dir <absolute path to a directory to save>
```

**5.Train a ridge regression (or a 3-layer NN) which predicts internal representation given brain activity, and make predictions using the best regression model.**

5.1. To train ridge regression, and save predictions, run the following command:

```
$ python ./src/train_brain2prednet_ridge.py --alpha <parameter for L2 regularization> --depth <R[0-3]> --brain_train_path <absolute path to brain train file> --brain_test_path <absolute path to brain test path> --internal_representation_path <absolute path to internal representation directory> --model_save_path <absolute path to directory to save trained model> --predicted_save_path <absolute path to directory to save predicted internal representation>
```

5.2. To train 3-layer NN, run the following command:

```
$ python ./src/train_3layer_mlp.py --gpu <GPU ID> --depth <R[0-3]> --brain_train_path <absolute path to brain train file> --brain_test_path <absolute path to brain test file> --internal_representation_path <absolute path to internal representation directory> --save_path <absolute path to directory to save trained models>
```

5.3. To save predicted internal representation with the best 3-layer NN, run the following command:

```
$ python ./src/save_predicted_R.py --initmodel <absolute path to the best model file> --gpu <GPU ID> --depth <R[0-3]> --brain_test_path <absolute path to brain test file> --internal_representation_path <absolute path to a directory to save predicted internal representation>
```

* The behaviour of train_brain2prednet_ridge.py is different from that of train_3layer_mlp.py. train_brain2prednet_ridge.py trains ridge regression with train data, evaluates the regression with test data, makes predictions from test brain activity, and saves the model and predictions. On the other hand, train_3layer_mlp.py trains 3-layer NN with test data while evaluating loss with test data and saving the models at each epoch. I apologize for the inconsistency.
* NOTE: These programs may cause memory shortage. The internal representation, R1 has so large dimensions that it was impossible to train brain-to-R1 in my experiment.

**6.Evaluate a correlation between predicted and correct internal representaions.**

6.1. Run the following command:

```
$ python calc_correlation.py --depth <R[0-3]> --reference_path <absolute path to reference internal representation directory> --predicted_path <absolute path to predicted internal representation directory>
```

**7.Decode predictive images from predicted internal representations.**

7.1 To decode predictive images from estimated R0, run the following command:

```
$ python decode_internal_representation_R0.py --sequences <path to stimulus sequence list file> --gpu <GPU ID> --initmodel <absolute path to your best model> --internal_representation_path <absolute path to predicted internal representation> --save_dir <absolute path to a directory to save decoded images>
```
7.2 To decode predictive images using estimated R2, run the following command:

```
$ python decode_internal_representation_R2.py --sequences <path to stimulus sequence list file> --gpu <GPU ID> --initmodel <absolute path to your best model> --internal_representation_path <absolute path to predicted internal representation> --save_dir <absolute path to a directory to save decoded images>
```
7.3 To decode predictive images using estimated R3, run the following command:

```
$ python decode_internal_representation_R3.py --sequences <path to stimulus sequence list file> --gpu <GPU ID> --initmodel <absolute path to your best model> --internal_representation_path <absolute path to predicted internal representation> --save_dir <absolute path to a directory to save decoded images>
```

**(8.Evaluate the performance of prednet.)**

8.1 To calculate the average of MSEs between 2 frames in a row, run the following command:

```
$ python MSE.py -data <image list> -begin_id <beginning frame ID(int)> -end_id <end frame ID(int)> --root <root directory> --channels <(# channels)> --size <size of images(width,height)>
```

8.2 To compare MSEs between (input, correct) pairs and (predict, correct) pairs, run the following command:

```
$ python MSE_Lv1_seq.py --data <sequence name> --seq_range <range of the sequence to evaluate(begin,end)> --path <image directory path> --channels <(# channels)> --size <size of images(width,height)>
```

8.3 To compare MSEs between (input, predict) pairs and (predict, correct) pairs, run the following command:

```
$ python MSE_Lv2_seq.py --data <sequence name> --seq_range <range of the sequence to evaluate(begin,end)> --path <image directory path> --channels <(# channels)> --size <size of images(width,height)>
```

* NOTE: The way to give data to MSE_Lv[1|2]_seq.py are different from that to MSE.py. I apologize for the inconsistency.

## References

- original paper : [Deep Predictive Coding Networks for Video Prediction and Unsupervised learning](https://arxiv.org/pdf/1605.08104.pdf)
- PredNet source code: [PredNet implementation in Chainer](https://github.com/quadjr/PredNet)

    A large part of PredNet-related code was developed with reference to codes available at the above link. I appreciate it.