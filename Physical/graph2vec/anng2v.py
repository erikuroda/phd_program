import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
import random
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec as word2vec
import numpy as np
import glob
import scipy
from imageio import imread
from skimage.transform import resize, rescale
import time
import re
import itertools
import datetime
from json import load
import math

dt_now = datetime.datetime.now()
print('start:::', dt_now)

#読み込み対象ファイル
#path = 'annotation_train/annotation_00000-01000/annotation_00000.json'
#ファイルモード, 読み込み用
mode = "r"

# ファイル一覧
files = glob.glob("../annotation_train/annotation_00000-01000/annotation_0000*.json")
ann_list = sorted(files)
print('ann_list ok::::::::::')

pos = {}
pos_dict = {}
pos_file = {}

ori = {}
ori_dict = {}

vel = {}
vel_dict = {}

angv = {}
angv_dict = {}

for ann in ann_list:
    ann_f = ann.strip('../annotation_train/')
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
                ori[str(o['object_id'])] = (o['orientation'][1], o['orientation'][0])
                vel[str(o['object_id'])] = (o['velocity'][1], o['velocity'][0])
                angv[str(o['object_id'])] = (o['angular_velocity'][1], o['angular_velocity'][0])

                # idは通し番号
                id = str(m['frame_id']+128*int(ann_f))
            pos_dict[id] = pos
            ori_dict[id] = ori
            vel_dict[id] = vel
            angv_dict[id] = angv
            pos = {}
            ori = {}
            vel = {}
            angv = {}

print('pos ex ok::::::::::')

###########################################
## This program is annotation2graph_only.##
###########################################
input_path = 'cus211217/'
dimensions = 3072
workers = 4
epochs = 10
min_count = 5
wl_iterations = 2
learning_rate = 0.025
down_sampling = 0.0001

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
        #print('features:::::::::')
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v in features.items()}
    return graph, features, name

def feature_extractor(path, rounds):
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding(output_path, model, files, dimensions):
    out = []
    for f in files:
        identifier = path2name(f)
        out.append([identifier] + list(model.docvecs["g_"+identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out = out.drop("type", axis=1)
    print(out.shape)
    print(out)


graphs = glob.glob(os.path.join(input_path, "sample_0000*.json"))
print("\nFeature extraction started.\n")
document_collections = Parallel(n_jobs=workers)(delayed(feature_extractor)(g, wl_iterations) \
                                                     for g in tqdm(graphs))
print("\nOptimization started.\n")

model = Doc2Vec(document_collections,
                vector_size=dimensions,
                window=0,
                min_count=min_count,
                dm=0,
                sample=down_sampling,
                workers=workers,
                epochs=epochs,
                alpha=learning_rate)

out = []
files = graphs

for f in files:
    identifier = path2name(f)
    out.append([identifier] + list(model.dv["g_"+identifier]))

column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
out = pd.DataFrame(out, columns=column_names)
out = out.sort_values(["type"])
out = out.drop("type", axis=1)
a = out.to_numpy()
print(a.shape)
# -1にすると自動的に計算
b = np.reshape(a, (-1, 32, 32, 3))
print(b.shape)
#np.save(output_path2, b)
print('g2v ok:::::::::::')
####################################
#↑ここまででgraph2vecでのnpyを作成#
## graph_onlyはここまで#############
## (ディレクトリ数、32、32、3)の形##
####################################

###############
###velocity####
###############
v_a = []
v_in = []
l_v = []
v_all = []
for k_v, p_v in vel_dict.items():
    for k2_v, p2_v in p_v.items():
        v_a.append(p2_v[0])
        v_a.append(p2_v[1])
        v_a.append(0.0)
        for j in range(32):
            v_in.append(v_a)
        #print('v_in:::', v_in)
        l_v.append(v_in)
        v_all.append(l_v)
        l_v = []
        v_a = []
        v_in = []
    v_in = []
v_ar = np.array(v_all, dtype='float32')
print('v_ar ok::::::::::')

#######################
###angular_velocity####
#######################
angv_a = []
angv_in = []
l_angv = []
l_angv_all = []
for k_angv, p_angv in angv_dict.items():
    for k2_angv, p2_angv in p_angv.items():
        angv_a.append(p2_angv[0])
        angv_a.append(p2_angv[1])
        angv_a.append(0.0)
        for angj in range(32):
            angv_in.append(angv_a)
        #print('angv_in:::', angv_in)
        l_angv.append(angv_in)
        l_angv_all.append(l_angv)
        l_angv = []
        angv_a = []
        angv_in = []
    angv_in = []
angv_ar = np.array(l_angv_all, dtype='float32')
#print('angv_ar:::', angv_ar)
print('angv_ar ok::::::::::')

################
###collision####
################

p_an1 = 0
p_an2 = 0
c_l = []
c_l_f = []


for k1, p1 in pos_dict.items():
    # k1はframe k1 = frame+video_id*128
    for k2, p2 in p1.items():
        for k3, p3 in p1.items():
            if k2<k3:
                # x座標
                an1 = math.floor(abs(p2[0]- p3[0])*1000)/1000
                # y座標
                an2 = math.floor(abs(p2[1]- p3[1])*1000)/1000

                # ２点間距離
                x_a = np.array([p2[0], p2[1]])
                x_b = np.array([p3[0], p3[1]])
                u = x_b - x_a
                kyori = np.linalg.norm(u)

                sa_an1 = math.floor(abs(p_an1-an1)*1000)/1000
                sa_an2 = math.floor(abs(p_an2-an2)*1000)/1000

                if kyori < 0.425:
                    # videoframe
                    v_k1 = int(k1)//128
                    r_k1  = int(k1)-128*v_k1
                    p_an1 = an1
                    p_an2 = an2
                    c_l.append(k1)
                    c_l.append(k2)
                    c_l.append(k3)
                    c_l_f.append(c_l)
                else:
                    continue
                c_l = []
            else:
                continue

c = 0
f_id = 1
c_flag = []
c_flag_a = []
c_flag_all = []
all_n = np.zeros((10, 1, 32, 3))
for k1, p1 in pos_dict.items():
    # k1はframe k1 = frame+video_id*128
    for c_p in p1:
        # c_pはnode番号, k1*c_p=6400, 全node数
        c += 1
        for h in c_l_f:
            if h[0] == k1:
                if h[1] == c_p:
                    all_n[c] = [1,1,1]
                elif h[2] == c_p:
                    all_n[c] = [1,1,1]
                else:
                    continue
            else:
                continue

all_n = np.array(all_n, dtype='float32')
print('all_n ok::::::::::')

################
###move_rec#####
################
sa = {}
sam = {}
for k5,p5 in pos_dict.items():
    if int(k5)%128==127:
        continue
    else:
        for k4, p4 in pos_dict.items():
            if int(k4)==int(k5)+1:
                for nu in p4:
                    x_dif = math.floor((pos_dict[str(k4)][str(nu)][0]-pos_dict[str(k5)][str(nu)][0])*10000)/10000
                    y_dif = math.floor((pos_dict[str(k4)][str(nu)][1]-pos_dict[str(k5)][str(nu)][1])*10000)/10000
                    sa[str(nu)] = (x_dif, y_dif, 0.0)
                #sam.append(sa)
                sam[str('{}_{}').format(k5,k4)] = sa
                sa = {}
            elif int(k4)==int(k5) and int(k5)%128==0:
                for nu in p4:
                    sa[str(nu)] = (0.0, 0.0, 0.0)
                    sam[str('{}_{}').format(k5,k4)] = sa
                sa = {}

s_a = []
s_aa = []
s_aaa = []
s_all = []
a_in = []
a_in_all = []
for s1 , sp in sam.items():
    for s2, sp2 in sp.items():
        s_a.append(sp2[0])
        s_a.append(sp2[1])
        s_a.append(sp2[2])
        for g in range(32):
            s_aa.append(s_a)
        a_in.append(s_aa)
        a_in_all.append(a_in)
        a_in = []
        s_a = []
        s_aa = []

    s_all = np.array(a_in_all, dtype='float32')
print('s_all ok::::::::::')

################
###save_data#####
################
print('concate_start!!!:::')
con = np.concatenate((b, v_ar, angv_ar, all_n, s_all), axis = 1)
print('con shape:::', con.shape)
print('saving_start:::')
np.save('./features/211217_9', con)
print('saving_finish!!!!!!')

dt_now2 = datetime.datetime.now()
print('finish:::', dt_now2)
