import os
import json
from json import load
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

dt_now = datetime.datetime.now()
print('start:::', dt_now)

#読み込み対象ファイル
#path = 'annotation_train/annotation_00000-01000/annotation_00000.json'
#ファイルモード, 読み込み用
mode = "r"

# ファイル一覧
files = glob.glob("/Storage/kuroda/M2/CLEVRER/executor/data/annotation_train/annotation_00000-01000/annotation_00*.json")
ann_list = sorted(files)
print('im_list ok::::::::::')

pos = {}
pos_dict = {}
pos_file = {}


for ann in ann_list:

    ann_f = ann.strip('/Storage/kuroda/M2/CLEVRER/executor/data/annotation_train/')
    ann_f = ann_f.strip('00000-01000/')
    ann_f = ann_f.strip('annotation_')
    ann_f = ann_f.strip('.json')

    with open(ann, mode) as f:
        data = load(f)
        #128物体数分、クラスはリスト

        video_file = data['video_filename']
        video_file = re.findall('video_(.*).mp4', video_file)
        motion = data['motion_trajectory']
        coll = data['collision']


        for m in motion:
            # dict型 128

            for o in m['objects']:
                # kye: 'object_id', 'location', 'orientation', 'velocity', 'angular_velocity', 'inside_camera_view'
                #3D座用でpos作成
                # o['location'][0]がy座標：縦
                # o['location'][1]がx座標：横
                # 右下が正と正
                # x -2~2 y -2.5~2.5
                pos[str(o['object_id'])] = (o['location'][1], o['location'][0])

                # idは通し番号
                id = str(m['frame_id']+128*int(ann_f))

            pos_dict[id] = pos
            pos = {}

print('pos ex ok::::::::::')

################
###move_vec#####
################
m_v = {}
m_vall = {}
o = np.array([0,0])

a = 0
c = -1
co = 0
coo = 0
mo_f = []
mo_f_al = []
mo_f_all = np.zeros((631040, 6, 32, 3))

for k6,p6 in pos_dict.items():
    #print('k:::::::::::::::::::::::::::::',k6)
    #c += 1

    for j in itertools.permutations(p6, 2):
        be = pos_dict[k6][j[0]]
        ne = pos_dict[k6][j[1]]

        be = np.array([be[0], be[1]])
        ne = np.array([ne[0], ne[1]])

        sabun = be - ne

        if int(j[0]) == co:
            c += 1
            if sabun[0]>=0 and sabun[1]>=0:
                mo_f_all[coo][c] = [5,5,5]

            elif sabun[0]>=0 and sabun[1]<0:
                mo_f_all[coo][c] = [1,1,1]

            elif sabun[0]<0 and sabun[1]>=0:
                mo_f_all[coo][c] = [-1,-1,-1]

            else:
                mo_f_all[coo][c] = [-5,-5,-5]

        elif int(j[0]) == co+1:
            c = 0
            co += 1
            coo += 1
            mo_f_al.append(mo_f)
            mo_f = []
            if sabun[0]>=0 and sabun[1]>=0:
                mo_f_all[coo][c] = [5,5,5]

            elif sabun[0]>=0 and sabun[1]<0:
                mo_f_all[coo][c] = [1,1,1]

            elif sabun[0]<0 and sabun[1]>=0:
                mo_f_all[coo][c] = [-1,-1,-1]

            else:
                mo_f_all[coo][c] = [-5,-5,-5]

        else:
            c = 0
            co = 0
            coo += 1
            mo_f = []
            if sabun[0]>=0 and sabun[1]>=0:
                mo_f_all[coo][c] = [5,5,5]

            elif sabun[0]>=0 and sabun[1]<0:
                mo_f_all[coo][c] = [1,1,1]

            elif sabun[0]<0 and sabun[1]>=0:
                mo_f_all[coo][c] = [-1,-1,-1]

            else:
                mo_f_all[coo][c] = [-5,-5,-5]

ve = np.array(mo_f_all)
print(ve.shape)

#########
#go_data#
#########
sada = np.load('/Storage/kuroda/M2/CLEVRER/executor/data/clvann_go_999_255_211115.npy')
print(sada.shape)

print('go_concate_start!!!:::')
con = np.concatenate((sada,ve), axis = 1)
print(con.shape)
np.save('/Storage/kuroda/M2/CLEVRER/executor/data/clvann_go_999_posflag_211220', con)

###########
#go_data###
###########
sada2 = np.load('/Storage/kuroda/M2/CLEVRER/executor/data/clvann_gi_999_255_211124.npy')
print(sada2.shape)

print('gi_concate_start!!!:::')
con2 = np.concatenate((sada2,ve), axis = 1)
print(con2.shape)
np.save('/Storage/kuroda/M2/CLEVRER/executor/data/clvann_gi_999_posflag_211220', con2)

########
#time###
########

dt_now = datetime.datetime.now()
print('start:::', dt_now)
