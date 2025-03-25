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
import datetime

data2 = pd.read_csv('./make_graph_dataset_211013/minipos.txt',
                    names=["folder", "img_id", "shape", "x1", "y1", "x2", "y2"]#, delimiter=r",\s*"
                   )


# 物体検知後の画像読み込み
im_files = glob.glob("/Users/eri_kuroda/sam/sim_0000*/output_*.png")
im_list = sorted(im_files)
im_list2 = list()
im_Y = []
print('img_list ok::::::::::')


Y = []

k= 1
pos = {}
p_f = 1
pos_list = []
pos_all = []

p_f_max = max(data2['img_id'])
p_f = 1
p_fo = 0

for p in data2.itertuples():

    for i in im_list:
        re_i_folder = re.sub('/Users/eri_kuroda/sam/sim_','', i)
        re_i_folder = re.sub('/output.*', '', re_i_folder)
        re_i_img = re.sub('/Users/eri_kuroda/sam/sim_.*/output_','',i)
        re_i_img = re.sub('.png', '', re_i_img)

        if p.folder == int(re_i_folder):
            if p.img_id == int(re_i_img):
                im_list2.append(i)

    if p.folder == p_fo:
        if p.img_id == p_f:
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)
            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            k += 1

        else:
            # 今までのpos
            pos_list.append(pos)
            pos = {}
            k = 1
            p_f += 1
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)
            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            k += 1

    elif p.folder == p_fo+1:
        #print('00001:::')
        p_f = 1
        if p.img_id == p_f:
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)

            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            k += 1
            p_fo += 1

        else:
            # 今までのpos
            pos_list.append(pos)
            pos = {}
            k = 1
            p_f += 1
            g_x = float((p.x1+p.x2)/2)
            g_y = float((p.y1+p.y2)/2)

            k_s = str(k)
            pos[k_s] = (g_x, g_y)
            k += 1
        #print(pos)

    else:
        break

# 最後のpos
pos_list.append(pos)

num_of_walk = 7
length_of_walk = 7

walks = list()
v_list = []
v_f_list = []
v_all = []
co = 1

print('pos ok!!!')

for s in pos_list:
    G = nx.Graph()
    for t in range(1, len(s)+1):
        G.add_node(str(t))
        for i in range(t+1, len(s)+1):
            G.add_edge(str(t), str(i))
            b = 0

    ax = plt.subplot()
    ax.invert_yaxis()
    nx.draw_networkx(G, pos=s, node_color="c")
    plt.close()
    co += 1
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

    v_array = np.array(v_list, dtype='float32')

    node_list = []
    walks = list()

print(v_array.shape)
###############
###img_vec####
###############
for i_li in im_list2:
    im = imread(i_li, format='png')
    resize_32 = resize(im,(32,32),mode='reflect',anti_aliasing=True)
    im_Y.append(resize_32)
img_array = np.array(im_Y, dtype='float32')
print('im_array ok::::::::::')
print(img_array.shape)

print('go_save:::')
np.save('/Users/eri_kuroda/Research/M2/CLEVRER+VTA_211021/yolo_sam_go_26', v_array)
print('concate!!!')
c = np.concatenate((v_array,img_array), axis = 1)
print('saving start!!!')
print(c.shape)
np.save('/Users/eri_kuroda/Research/M2/CLEVRER+VTA_211021/yolo_sam_gi_26', c)
print('saving finish!!!')
dt_now = datetime.datetime.now()
print('now:::',dt_now)
