import math

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import time, joblib

from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time, joblib
from datetime import datetime
from tqdm import tqdm


# import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
# from torch.utils.Data import Dataset, DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib


import wandb
from config import sweep_config,parameters_dict

import argparse

# TODO 搬运 SimCLRDatasetWrapper的dataset实现

class DKTDataset(Dataset):
    def __init__(self, group, min_samples=3, max_seq=100, max_skill_id = 20000):
        super(DKTDataset, self).__init__()
        self.max_seq = max_seq
        self.samples = {}
        # TODO key:skill_真ID  --- value:skill_index_id
        # TODO key:user_真ID --- value:user_index_id

        self.user_ids = []
        for user_id in group.index:
            s, q, qa = group[user_id]
            if len(q) < min_samples:
                continue

            # Main Contribution
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    # self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial])
                    self.samples[f"{user_id}_0"] = (s[:initial], q[:initial], qa[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq + 1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    self.samples[f"{user_id}_{seq + 1}"] = (s[start:end], q[start:end], qa[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (s, q, qa)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        s_, q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        # NOTE : 加一条 因为只有一条数据的时候 ，q_ qa_ 不为二维，就强行变成二维
        if len(q_.shape) == 1:
            q_ = q_[None]

        # q = np.zeros((self.max_seq, q_.shape[1]))
        q = np.full((self.max_seq, q_.shape[1]), 0.0) # torch.full((self.max_seq, q_.shape[1]), -10000)   # NOTE 很多情况下需要使用0 但是后面筛选掉0 导致BUG,  -10000使
        qa = np.zeros(self.max_seq, dtype=int)
        target_mask = np.full(self.max_seq, False)       # 标记非空
        # qa = np.full(self.max_seq, 0.001)

        if seq_len == self.max_seq:     # TODO  传入后后得到了整数？？
            q[:] = q_
            qa[:] = qa_
            target_mask[:] = True
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
            target_mask[-seq_len:] = True

        # NOTE 数据增强


        x_emb = self.onehot1(q[:-1], qa[:-1])  # NOTE 投入的特征是no1-49
        q_next = q[1:]  # NOTE 用于预测的是no2-50
        labels = qa[1:]
        target_mask_next = target_mask[1:]

        # NOTE dataloder不支持ndaary
        x_emb_torch = torch.Tensor(x_emb)
        # x_emb_torch = x_emb
        q_next_torch = torch.Tensor(q_next)
        labels_torch = torch.Tensor(labels)
        target_mask_next_torch = torch.Tensor(target_mask_next)
        user_id = torch.Tensor(user_id)

        return x_emb_torch, q_next_torch, labels_torch, target_mask_next_torch, user_id

    def onehot1(self, questions, answers):
        emb_num = questions.shape[-1]
        result = np.zeros(shape=[self.max_seq - 1, 2 * emb_num])

        for i in range(self.max_seq - 1):
            if answers[i] > 0:
                result[i][:emb_num] = questions[i]
            elif answers[i] == 0:
                result[i][emb_num:] = questions[i]
        return result


class STU_CL_cluster(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(STU_CL_cluster, self).__init__()
        self.device = device
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN NOTE 内置了一个RNN 所以在init里就没有声明隐藏变量/层
        # self.rnn = nn.RNN(input_dim * 2, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        # self.lstm = nn.LSTM(input_dim * 2, hidden_dim, layer_dim, batch_first=True, bidirectional=True)    # NOTE 被onehot了 所以input_dim *2
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)    # NOTE 没onehot了 所以input_dim 不变
        self.gru = nn.GRU(input_dim * 2, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        # self.fc = nn.Linear(hidden_dim * 2 + input_dim, output_dim)     #NOTE 因为 返回层是两层
        self.fc = nn.Linear(hidden_dim + input_dim, output_dim)     # NOTE BiLSTM *2
        self.sig = nn.Sigmoid()

        # NOTE 对比学习

        # NOTE 对比聚类

    def forward(self, x, q_next):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)      # NOTE BiLSTM *2
        # One time step
        _out_next, (hn, cn) = self.gru(x, h0)
        out_next = _out_next

        out = self.fc(out_next)     # NOTE  SIG激活函数使得loss变化 检查全连接层
        return out

        # TODO 对比损失
        #

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        cl_loss = torch.mean(out_dict["cl_loss"])  # torch.mean() for multi-gpu FIXME
        mask = true > -1

        loss = self.loss_fn(pred[mask], true[mask]) + self.reg_cl * cl_loss

        return loss, len(pred[mask]), true[mask].sum().item()

    def get_interaction_embed(self, skills, responses): # NOTE 其实就他妈ONE-HOT
        masked_responses = responses * (responses > -1).long()
        interactions = skills + self.num_skills * masked_responses
        return self.interaction_embed(interactions)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def main():

    train_dataset = dataset(train_df, seq_len, num_skills, num_questions)

    train_loader =
        DataLoader(
            SimCLRDatasetWrapper(
                train_dataset,
                seq_len,
                mask_prob,
                crop_prob,
                permute_prob,
                replace_prob,
                negative_prob,
                eval_mode=False,
            ),
            batch_size=batch_size,
        )
