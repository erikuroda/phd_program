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

mode="r"

co = 0
pos = {}
pos_dict = {}
pos_file = {}

l = []
l2 = []
dic_l = {}
dic_c = {}

# ファイル一覧
files = glob.glob("../annotation_train/annotation_00000-01000/annotation_00000.json")
im_list = sorted(files)

for im in im_list:
    print(im)
    k = im.strip('../annotation_train/')
    k = k.strip('00000-01000/')
    k = k.strip('annotation_')
    k = k.strip('.json')

    with open(im, mode) as f:
        data = load(f)
        #128物体数分、クラスはリスト

        video_file = data['video_filename']
        video_file = re.findall('video_(.*).mp4', video_file)
        motion = data['motion_trajectory']
        coll = data['collision']
        obj = data['object_property']


        for m in motion:
            # dict型 128
            #print(m)
            fr = m['frame_id']

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

            for i in obj:
                l.append(i['object_id'])
                dic_c[str(i['object_id'])] = i['color']

            for j in itertools.combinations(l, 2):
                j = list(j)
                l2.append(j)

            dic_l["edges"] = l2
            dic_l["features"] = dic_c
            dic_l["pos"] = pos
            dic_j = json.dumps(dic_l)

            co += 1
            with open('./cus211228/sample_{:05}.json'.format(co), 'w') as f2:
                json.dump(dic_l, f2)

            l = []
            l2 = []
            dic_l = {}
            dic_c = {}
            pos = {}

dt_now2 = datetime.datetime.now()
print('finish:::', dt_now2)
