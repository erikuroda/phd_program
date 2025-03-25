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

input_path = 'cus211229_yolo/'
#output_path = 'features/sample.csv'
output_path2 = 'features/sa211228'
dimensions = 3072
workers = 4
epochs = 10
min_count = 5
wl_iterations = 2
learning_rate = 0.025
down_sampling = 0.0001

pos_d = {}

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
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
    graph = nx.Graph()
    graph.add_edges_from(data["edges"])
    pos = data["pos"]
    for key, p in pos.items():
        pos_d[int(key)] = (p[0], p[1])
    sizes = 1200
    nx.draw_networkx(graph, pos=pos_d, with_labels=True, node_shape='.', node_size=sizes, node_color='cyan')
    plt.axis("off")
    #plt.show()
    plt.close()

    if "features" in data.keys():
        features = data["features"]
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

graphs = glob.glob(os.path.join(input_path, "*.json"))
graphs = sorted(graphs)
print("\nFeature extraction started.\n")
document_collections = Parallel(n_jobs=workers)(delayed(feature_extractor)(g, wl_iterations) for g in tqdm(graphs))
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

#filesはjsonファイルのこと
for f in files:
    identifier = path2name(f)
    out.append([identifier] + list(model.docvecs["g_"+identifier]))

column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
out = pd.DataFrame(out, columns=column_names)
out = out.sort_values(["type"])
out = out.drop("type", axis=1)
a = out.to_numpy()
print(a.shape)
b = np.reshape(a, (-1, 32, 32, 3))
print(b.shape)
#np.save(output_path2, b)
####################################
#↑ここまででgraph2vecでのnpyを作成#
## graph_onlyはここまで#############
## (ディレクトリ数、32、32、3)の形##
####################################
print('ok_g2v:::::::')

# 物体検知後の画像読み込み
im_files = glob.glob("/Storage/kuroda/M2/CLEVRER/executor/data/after_kenchi_00999/sim_00*/output_*.png")
#im_files = glob.glob("/Users/eri_kuroda/sam/sim_0000*/output_*.png")
im_list = sorted(im_files)
im_list2 = list()
im_Y = []
print('img_list ok::::::::::')

###############
###img_vec####
###############
for i_li in im_list:
    im = imread(i_li, format='png')
    resize_32 = resize(im,(32,32),mode='reflect',anti_aliasing=True)
    im_Y.append(resize_32)
img_array = np.array(im_Y, dtype='float32')
print(img_array.shape)
print('im_array ok::::::::::')

################
###save_data#####
################
print('saving_start:::')
np.save('/Storage/kuroda/M2/CLEVRER/executor/data/g2v_go_yolo_999_211230', b)
print('concate_start!!!:::')
con = np.concatenate((b,img_array), axis = 1)
print('saving_start:::')
np.save('/Storage/kuroda/M2/CLEVRER/executor/data/g2v_gi_yolo_999_211230', con)
print('saving_finish!!!!!!')

dt_now2 = datetime.datetime.now()
print('finish:::', dt_now2)
