import glob
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import scipy
from imageio import imread
from skimage.transform import resize, rescale

start = time.time()

files = glob.glob("/Users/eri_kuroda/sam/sim_0000*/*.png")
im_list = sorted(files)
Y = []
for i, im_file in enumerate(im_list):
    #print(i)
    im = imread(im_file, format='png') # 画像読み込み
    resize_32 = resize(im,(32,32),mode='reflect',anti_aliasing=True)
    Y.append(resize_32)
Y_array = np.array(Y)
np.save('/Users/eri_kuroda/Research/M2/CLEVRER+VTA_211021/sa_21', Y_array)
t = time.time() - start
print('finish!!!',t)
