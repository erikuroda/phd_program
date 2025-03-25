import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
#from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
import random
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec as word2vec
import numpy as np
import glob
import os
import scipy
from imageio import imread
from skimage.transform import resize, rescale
import time
import re
import itertools
import datetime
from json import load

dt_now = datetime.datetime.now()
print('start:::', dt_now)

data2 = pd.read_csv('minipos.txt',
                    names=["folder", "img_id", "shape", "x1", "y1", "x2", "y2"])

Y = []

k= 0
pos = {}
p_f = 1
pos_list = []
pos_all = []

fea = {}
dic_l = {}
dic_j = {}
fea_list = []

p_f_max = max(data2['img_id'])
p_f = 1
p_fo = 0

obj_n = 0
cv = 0

for p in data2.itertuples():

    if p.folder == p_fo:
        if p.img_id == p_f:
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)
            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            fea[k_s] = p.shape
            obj_n += 1
            k += 1

        else:
            obj_n = 0
            pos_list.append(pos)
            fea_list.append(fea)
            fea = {}
            pos = {}
            k = 0
            p_f += 1
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)
            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            fea[k_s] = p.shape
            obj_n += 1
            k += 1

    elif p.folder == p_fo+1:
        #print('22222else:::::')
        p_f = 1
        fea_list.append(fea)
        fea = {}
        pos_list.append(pos)
        pos = {}
        if p.img_id == p_f:
            obj_n = 0
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)

            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            fea[k_s] = p.shape
            obj_n += 1
            k += 1
            p_fo += 1

        else:
            # 今までのpos
            pos_list.append(pos)
            fea_list.append(fea)
            fea = {}
            pos = {}
            k = 0
            p_f += 1
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)
            fea[k_s] = p.shape
            obj_n += 1
            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            k += 1

    else:
        break

# 最後のpos
pos_list.append(pos)
fea_list.append(fea)

l2 = []

sa = []
sa2 = []
d = 0
a = 0
aa = 0
for i in fea_list:
    for h, hk in i.items():
        for h2, hk2 in i.items():
            if h >= h2:
                continue
            else:
                sa.append(int(h))
                sa.append(int(h2))
                sa2.append(sa)
                sa = []
    dic_l["edges"] = sa2
    sa2 = []

    dic_l["features"] = i

    dic_j[a] = dic_l
    a += 1
    dic_l = {}

b = 0
for i in pos_list:
    dic_j[b]["pos"] = i
    b += 1

c = 0
for s in dic_j.values():
    with open('./cus211229_yolo/sample_{:05}.json'.format(c), 'w') as f2:
        json.dump(s, f2)
    c += 1

print(len(dic_j))

dt_now2 = datetime.datetime.now()
print('finish:::', dt_now2)
