import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from model import *
from prep_data import *
import random

checkpoint = torch.load("../output/model/new_model_epoch_3799.pt")
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

with open("../data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)
tdvae.eval()
tdvae = tdvae.cuda()

data = MNIST_Dataset(MNIST['train_image'], binary = False)
batch_size = 6
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)
idx, images = next(enumerate(data_loader))

images = images.cuda()

## calculate belief
tdvae.forward(images)

t1, t2 = 11, 15
rollout_images = tdvae.rollout(images, t1, t2)

fig = plt.figure(0, figsize = (12,4))

fig.clf()
gs = gridspec.GridSpec(batch_size,t2+20)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(batch_size):
    for j in range(t2):
        axes = plt.subplot(gs[i,j])
        axes.imshow(1-images.cpu().data.numpy()[i,j].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')

    for j in range(t1,t2+1):
        axes = plt.subplot(gs[i,j+10])
        axes.imshow(1-rollout_images.cpu().data.numpy()[i,j-t1].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')

fig.savefig("../output/images/rollout0205_1.eps")
plt.savefig("../output/images/rollout0205_1.png")

fig.clf()
gs = gridspec.GridSpec(batch_size,t2+20)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(batch_size):
    for j in range(t2): # t1
        axes = plt.subplot(gs[i,j]) # i,j
        axes.imshow(1-images.cpu().data.numpy()[i,j].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')
#### 上のfor文は0からt1までを順番に出力

    for j in range(t1,t2+1):
        axes = plt.subplot(gs[i,j+10]) #[i,j+1]
        axes.imshow(1-rollout_images.cpu().data.numpy()[i,j-t1-3].reshape(28,28),## j-t1
                    cmap = 'binary') # 1- は色の反転
        axes.axis('off')
fig.savefig("../output/images/rollout0205_3.eps")
plt.savefig("../output/images/rollout0205_3.png")

fig.clf()
gs = gridspec.GridSpec(batch_size,t2+20)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(batch_size):
    for j in range(t2): # t1
        axes = plt.subplot(gs[i,j]) # i,j
        axes.imshow(1-images.cpu().data.numpy()[i,j].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')
#### 上のfor文は0からt1までを順番に出力

    for j in range(t1,t2+1):
        random_num = random.randint(1,10)
        axes = plt.subplot(gs[i,j+10]) #[i,j+1]
        axes.imshow(1-rollout_images.cpu().data.numpy()[i,j-t1-random_num].reshape(28,28),## j-t1
                    cmap = 'binary') # 1- は色の反転
        axes.axis('off')
fig.savefig("../output/images/rollout0205_random.eps")
plt.savefig("../output/images/rollput0205_random.png")
##plt.show()
sys.exit()
