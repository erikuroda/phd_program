import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *
from prep_data import *
import sys

#### preparing dataset
with open("../data/MNIST.pkl", 'rb') as file_handle:
    MNIST = pickle.load(file_handle)

data = MNIST_Dataset(MNIST['train_image'])
batch_size = 512 #### データセットを512ずつにわける
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)

#### build a TD-VAE model
input_size = 784 #### inputの大きさ
processed_x_size = 784 #### 入力xの大きさ
belief_state_size = 50 #### 信念状態の大きさ
state_size = 8 #### 状態は全部で8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
tdvae = tdvae.cuda() #### GPU使用

#### training
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005) #### 学習率の設定
num_epoch = 4000 #### 4000回学習する
log_file_handle = open("../log/loginfo_new.txt", 'w') #### log
for epoch in range(num_epoch):
    for idx, images in enumerate(data_loader): #### enumerateでindex番号も取ってこれる
        images = images.cuda()
        tdvae.forward(images)
        t_1 = np.random.choice(16)   ## t1ランダムで決めて、
        t_2 = t_1 + np.random.choice([1,2,3,4]) ## そこから1~4の間で飛ばして学習させてる
        loss = tdvae.calculate_loss(t_1, t_2) ##model.pyのcalculate_lossモジュール呼び出し
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
              file = log_file_handle, flush = True)

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()))

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "../output/model/new_model_epoch_{}.pt".format(epoch))
