import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

class PreProcess(nn.Module):
    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t

class Decoder(nn.Module):
    ## デコーダのモジュール
    ## 最後に画像に戻す
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


class TD_VAE(nn.Module):
    ## beliefの層を多層にする
    ## TD-VAEのNNモジュール
    ## lossをbeliefのLSTMのところに回帰

    def __init__(self, x_size, processed_x_size, b_size, z_size):
        super(TD_VAE, self).__init__()
        self.x_size = x_size ## 観測
        self.processed_x_size = processed_x_size
        self.b_size = b_size ## 信念 -> LSTM
        self.z_size = z_size ## 推測

        ## input pre-process layer
        self.process_x = PreProcess(self.x_size, self.processed_x_size)

        self.lstm = nn.LSTM(input_size = self.processed_x_size,
                            hidden_size = self.b_size,
                            batch_first = True)

        ## ここのLSTMしているbeliefのところに、KLを回帰
        ## サンプリングは高い層で行われている
        ## bからz生成
        ## beliefからzの予測
        ## 図でいうと、P_Bのところ　zの前の確率のところ
        self.l2_b_to_z = DBlock(b_size, 50, z_size) # layer 2
        self.l1_b_to_z = DBlock(b_size + z_size, 50, z_size) # layer 1

        ## 2層目のhat_zを定義
        self.l2_b_to_hat_z = DBlock(b_size, 50, z_size) # layer 2
        self.l1_b_to_hat_z = DBlock(b_size + z_size, 50, z_size) # layer 1

        ## zを推論する t2からt1のzを生成するところ hat_z_t1_l
        self.l2_infer_z = DBlock(b_size + 2*z_size, 50, z_size) # layer 2
        self.l1_infer_z = DBlock(b_size + 2*z_size + z_size, 50, z_size) # layer 1

        ## bからz_t1_lを推論→calculate_loss

        ## t1が与えられたとき、t2に行くときの遷移 P_T_t2
        self.l2_transition_z = DBlock(2*z_size, 50, z_size)
        self.l1_transition_z = DBlock(2*z_size + z_size, 50, z_size)

        ## 観測 P_D
        self.z_to_x = Decoder(2*z_size, 200, x_size)

    def forward(self, images):
        self.batch_size = images.size()[0]
        self.x = images
        ## 歳差 回転軸が円を描く減少
        self.processed_x = self.process_x(self.x)

        ## 信念状態の集合
        ## LSTM
        self.b, (h_n, c_n) = self.lstm(self.processed_x)


    def calculate_loss(self, t1, t2):
        ## ロスの計算
        ## bからzを出すP(確率)を用いてる P -> z という流れ
        t2_l2_z_mu, t2_l2_z_logsigma = self.l2_b_to_z(self.b[:, t2, :])
        t2_l2_z_epsilon = torch.randn_like(t2_l2_z_mu)
        t2_l2_z = t2_l2_z_mu + torch.exp(t2_l2_z_logsigma)*t2_l2_z_epsilon

        ## z in layer 1
        t2_l1_z_mu, t2_l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t2,:], t2_l2_z),dim = -1))
        t2_l1_z_epsilon = torch.randn_like(t2_l1_z_mu)
        t2_l1_z = t2_l1_z_mu + torch.exp(t2_l1_z_logsigma)*t2_l1_z_epsilon

        ## 層1と層2のzを結合 t2のｚ
        t2_z = torch.cat((t2_l1_z, t2_l2_z), dim = -1)
        ## t2からt1を推測
        ## 未来から過去を予測

        ## layer2
        t1_l2_qs_z_mu, t1_l2_qs_z_logsigma = self.l2_infer_z(
            torch.cat((self.b[:,t1,:], t2_z), dim = -1))
        t1_l2_qs_z_epsilon = torch.randn_like(t1_l2_qs_z_mu)
        t1_l2_qs_z = t1_l2_qs_z_mu + torch.exp(t1_l2_qs_z_logsigma)*t1_l2_qs_z_epsilon

        ## layer1
        t1_l1_qs_z_mu, t1_l1_qs_z_logsigma = self.l1_infer_z(torch.cat((self.b[:,t1,:], t2_z, t1_l2_qs_z), dim = -1))
        t1_l1_qs_z_epsilon = torch.randn_like(t1_l1_qs_z_mu)
        t1_l1_qs_z = t1_l1_qs_z_mu + torch.exp(t1_l1_qs_z_logsigma)*t1_l1_qs_z_epsilon

        ## 層1と層2を合体　q
        t1_qs_z = torch.cat((t1_l1_qs_z, t1_l2_qs_z), dim = -1)

        ## belief_t1からP_t1予測
        ## 信念状態belief
        t1_l2_pb_z_mu, t1_l2_pb_z_logsigma = self.l2_b_to_z(self.b[:, t1, :])
        t1_l1_pb_z_mu, t1_l1_pb_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t1,:], t1_l2_qs_z),dim = -1))

        ## 二層目のbelief
        t1_l2_pb_z_mu_2, t1_l2_pb_z_logsigma_2 = self.l2_b_to_hat_z(self.b[:, t1, :])
        t1_l1_pb_z_mu_2, t1_l1_pb_z_logsigma_2 = self.l1_b_to_hat_z(
            torch.cat((self.b[:,t1,:], t1_l2_qs_z),dim = -1))

        ## z_t1からP_t2予測
        t2_l2_t_z_mu, t2_l2_t_z_logsigma = self.l2_transition_z(t1_qs_z)
        t2_l1_t_z_mu, t2_l1_t_z_logsigma = self.l1_transition_z(
            torch.cat((t1_qs_z, t2_l2_z), dim = -1))

        ## z_t2からx_t2予測
        t2_x_prob = self.z_to_x(t2_z)

        ## layer1とlayer2に関してKLを回帰
        ## KLの計算をlayer1とlayer2に足す
        ## t1_l2_pb_z = beliefからの予測　t1からt1
        ## t1_l2_qs_z = 未来からの予測　t2からt1
        ## layer2のKL
        loss = 0.5*torch.sum(((t1_l2_pb_z_mu - t1_l2_qs_z)/torch.exp(t1_l2_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l2_pb_z_logsigma, -1) - torch.sum(t1_l2_qs_z_logsigma, -1)

        ## t1_l1_pb_z = beliefからの予測
        ## t1_l1_qs_z = 未来からの予測
        ## layer1のKL
        ## layer1で出したKLに足す
        loss += 0.5*torch.sum(((t1_l1_pb_z_mu - t1_l1_qs_z)/torch.exp(t1_l1_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l1_pb_z_logsigma, -1) - torch.sum(t1_l1_qs_z_logsigma, -1)

        ## z_t1_l_1を定義
        z_t1_l_1 = 0
        z_t1_l_1 += loss

        ## t2におけるlog 信念
        loss += torch.sum(-0.5*t2_l2_z_epsilon**2 - 0.5*t2_l2_z_epsilon.new_tensor(2*np.pi) - t2_l2_z_logsigma, dim = -1)
        loss += torch.sum(-0.5*t2_l1_z_epsilon**2 - 0.5*t2_l1_z_epsilon.new_tensor(2*np.pi) - t2_l1_z_logsigma, dim = -1)

        ## t2におけるlog 遷移
        loss += torch.sum(0.5*((t2_l2_z - t2_l2_t_z_mu)/torch.exp(t2_l2_t_z_logsigma))**2 + 0.5*t2_l2_z.new_tensor(2*np.pi) + t2_l2_t_z_logsigma, -1)
        loss += torch.sum(0.5*((t2_l1_z - t2_l1_t_z_mu)/torch.exp(t2_l1_t_z_logsigma))**2 + 0.5*t2_l1_z.new_tensor(2*np.pi) + t2_l1_t_z_logsigma, -1)

        ## torch.mean()：テンソル要素の算術平均
        ## lossの平均を最後に計算
        loss += -torch.sum(self.x[:,t2,:]*torch.log(t2_x_prob) + (1-self.x[:,t2,:])*torch.log(1-t2_x_prob), -1)

        ## belief二層目のerror再計算の層を入れる
        ## z_t1_l_1 のところには下位層でのKLのみが入ってくる
        ## 上位層でのKLの計算は、エラーの誤差を計算して小さくしていく

        loss = torch.mean(loss)

        ## t1_l2_pb_z_mu t1のbelief
        ## t1_l1_pb_z_mu t2のbelief
        t1_l1_pb_z_mu += loss
        t1_l2_pb_z_mu += loss

        t1_l2_hat_z_mu, t1_l2_hat_z_logsigma = self.l2_b_to_hat_z(self.b[:, t2, :])
        t1_l2_hat_z_epsilon = torch.randn_like(t1_l2_hat_z_mu)
        t1_l2_hat_z = t1_l2_hat_z_mu + torch.exp(t1_l2_hat_z_logsigma)*t1_l2_hat_z_epsilon
        t1_l1_hat_z_mu, t1_l1_hat_z_logsigma = self.l1_b_to_hat_z(
            torch.cat((self.b[:,t2,:], t1_l2_hat_z),dim = -1))
        t1_l1_hat_z_epsilon = torch.randn_like(t1_l1_hat_z_mu)
        t1_l1_hat_z = t1_l1_hat_z_mu + torch.exp(t1_l1_hat_z_logsigma)*t1_l1_hat_z_epsilon
        t1_hat_z = torch.cat((t1_l1_hat_z, t1_l2_hat_z), dim = -1)

        ## hat_z_t1_l_1とz_t1_l_1のKL
        loss_l = 0.5*torch.sum(((t1_l2_hat_z_mu - t1_l2_qs_z)/torch.exp(t1_l2_hat_z_logsigma))**2,-1)
        loss_l += 0.5*torch.sum(((t1_l1_hat_z_mu - t1_l1_qs_z)/torch.exp(t1_l1_hat_z_logsigma))**2,-1)

        return loss

    def rollout(self, images, t1, t2):
        self.forward(images)

        l2_z_mu, l2_z_logsigma = self.l2_b_to_z(self.b[:,t1-1,:])
        l2_z_epsilon = torch.randn_like(l2_z_mu)
        l2_z = l2_z_mu + torch.exp(l2_z_logsigma)*l2_z_epsilon

        l1_z_mu, l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t1-1,:], l2_z), dim = -1))
        l1_z_epsilon = torch.randn_like(l1_z_mu)
        l1_z = l1_z_mu + torch.exp(l1_z_logsigma)*l1_z_epsilon
        current_z = torch.cat((l1_z, l2_z), dim = -1)

        rollout_x = []

        for k in range(t2 - t1 + 1):
            next_l2_z_mu, next_l2_z_logsigma = self.l2_transition_z(current_z)
            next_l2_z_epsilon = torch.randn_like(next_l2_z_mu)
            next_l2_z = next_l2_z_mu + torch.exp(next_l2_z_logsigma)*next_l2_z_epsilon

            next_l1_z_mu, next_l1_z_logsigma  = self.l1_transition_z(
                torch.cat((current_z, next_l2_z), dim = -1))
            next_l1_z_epsilon = torch.randn_like(next_l1_z_mu)
            next_l1_z = next_l1_z_mu + torch.exp(next_l1_z_logsigma)*next_l1_z_epsilon

            next_z = torch.cat((next_l1_z, next_l2_z), dim = -1)

            next_x = self.z_to_x(next_z)
            rollout_x.append(next_x)

            current_z = next_z

        rollout_x = torch.stack(rollout_x, dim = 1)

        return rollout_x
