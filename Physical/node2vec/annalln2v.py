from json import load
import itertools
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
#import cupy as cp
import time
import re
import math

###############
#annからpos抽出#
###############

#読み込み対象ファイル
#path = 'annotation_train/annotation_00000-01000/annotation_00000.json'
#ファイルモード, 読み込み用
mode = "r"

# ファイル一覧
files = glob.glob("annotation_train/annotation_00000-01000/annotation_000*.json")
ann_list = sorted(files)
print('im_list ok::::::::::')

# 物体検知後の画像読み込み
im_files = glob.glob("/Users/eri_kuroda/sam/sim_000*/output_*.png")
im_list = sorted(im_files)
im_list2 = list()
im_Y = []
print('img_list ok::::::::::')

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
    #print(j)

    ann_f = ann.strip('annotation_train/')
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

            for i2 in im_list:
                sim = re.findall('sim_(.*)/', i2)
                out = re.findall('output_(.*).png', i2)
                if int(video_file[0]) == int(sim[0]):
                    if int(m['frame_id']) == int(out[0])-1:

                        for ob in m['objects']:
                            # kye: 'object_id', 'location', 'orientation'
                            #'velocity', 'angular_velocity', 'inside_camera_view'
                            im_list2.append(i2)

                    else:
                        continue
                else:
                    continue

            for o in m['objects']:
                # kye: 'object_id', 'location', 'orientation', 'velocity', 'angular_velocity', 'inside_camera_view'
                #3D座用でpos作成
                # o['location'][0]がy座標：縦
                # o['location'][1]がx座標：横
                # 右下が正と正
                # x -2~2 y -2.5~2.5
                pos[str(o['object_id'])] = (o['location'][1], o['location'][0])
                #print('z::::', o['location'][2])

                ori[str(o['object_id'])] = (o['orientation'][1], o['orientation'][0])
                vel[str(o['object_id'])] = (o['velocity'][1], o['velocity'][0])
                angv[str(o['object_id'])] = (o['angular_velocity'][1], o['angular_velocity'][0])


                # idは通し番号
                id = str(m['frame_id']+128*int(ann_f))
            #print('id::::', id)
            #print('pos:::', pos)
            pos_dict[id] = pos
            ori_dict[id] = ori
            vel_dict[id] = vel
            angv_dict[id] = angv
            pos = {}
            ori = {}
            vel = {}
            angv = {}

print('pos ex ok::::::::::')

num_of_walk = 7
length_of_walk = 7

walks = list()
v_list = []
v_f_list = []
v_all = []
co = 1

for key, p in pos_dict.items():
    G = nx.Graph()
    for t in p:
        G.add_node(t)
        for i in range(int(t)+1, len(p)):
            G.add_edge(str(t), str(i))

    fig, ax = plt.subplots()
    ax.invert_yaxis()
    nx.draw_networkx(G, pos=p, node_color="c")
    plt.close()

    co += 1
    #print(s)
    node_list = []
    node_list = list(G.nodes())

    for node in node_list:
        now_node = node
        walk = list()
        walk.append(node)

        for j in range(length_of_walk):
            if len(list(G.neighbors(now_node))) > 1:
                next_node = random.choice(list(G.neighbors(now_node)))
                walk.append(str(next_node))
                now_node = node
            else:
                next_node = now_node
                walk.append(str(next_node))
                now_node = node

        walks.append(walk)

    # gensim の Word2Vecを使った学習部分
    model = word2vec(walks, vector_size=3072, min_count=0, window=7, workers=4)
    # walks: 対象
    # size: 中間層の数（分散表現の次元）
    # window: 周辺単語の数
    # min_count: カウントする単語の最小出現数
    # workers: 処理のスレッド数（とりあえず、4を引数とする）

    for node in G.nodes():
        vector = model.wv[str(node)]
        v_re = vector.reshape([32,32,3])
        v_re = v_re*255
        v_list.append(v_re)

    #print('v_array:::')
    v_array = np.array(v_list, dtype='float32')

    node_list = []
    walks = list()

print('v_array ok::::::::::')

###############
###img_vec####
###############
for i_li in im_list2:
    im = imread(i_li, format='png')
    resize_32 = resize(im,(32,32),mode='reflect',anti_aliasing=True)
    im_Y.append(resize_32)
img_array = np.array(im_Y, dtype='float32')
print('im_array ok::::::::::')

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
all_n = np.zeros((64000, 1, 32, 3))
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
print('concate1_start!!!:::')
con1 = np.concatenate((v_array, v_ar, angv_ar, all_n, s_all), axis = 1)
print('con1_shape:::', con1.shape)
print('saving_start:::')
np.save('ann_data_211024/clvann_graph_only_211029_0_99_255', con1)

print('concate2_start!!!:::')
con2 = np.concatenate((v_array,img_array, v_ar, angv_ar, all_n, s_all), axis = 1)
print('saving_start:::')
np.save('ann_data_211024/clvann_graph_img_211028_0_99', con2)
print('saving_finish!!!!!!')
