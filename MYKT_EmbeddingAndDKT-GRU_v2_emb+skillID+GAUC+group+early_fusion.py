import random

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
import config
import MYKT_StuGroup_and_diff_dataprocess

import wandb
from config import sweep_config, parameters_dict

import argparse

# NOTE 全局参数
args = config.get_args()
K = args.K
dataset_path = args.dataset_path
# dataset_name = args.dataset_name
all_stuGroup_file_name = args.all_stuGroup_file_name
KC_num = args.KC_num
infering_method = args.infering_method
cluster_method = args.cluster_method
channels = args.channels  #
EMB_method = args.EMB_method


#
# # note 定义JOBname
# # jobname_all_stuGroup_inDiffKC ="{}cluster_method={}/all_stuGroup_inDiff_KC_K={}_NAN=0.5_.pkl.zip".format(
# #     dataset_path, cluster_method, K) # NOTE : 学生的分类情况dataframe
# config.jobname_all_c2c_bystuGroup = "{}cluster_method={}/infering_methods={}/all_c2c_bystuGroup_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path,cluster_method, infering_method, K, KC_num) # NOTE 通过学生分类情况得到C2C三维数据
# config.jobname_all_skill_diff = "{}cluster_method={}/all_skill_diff_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, KC_num) # NOTE 只和聚类方法、K、KCnum有关
# config.jobname_all_skill_prob = "{}cluster_method={}/all_skill_prob_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, KC_num)
# config.jobname_skill_df = "{}skill_prob.pkl.zip".format(dataset_path)
# config.jobname_all_c2c_emb_bystuGroup = \
#     "{}cluster_method={}/infering_methods={}/all_c2c_emb_bystuGroup_EMB_method={}_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path,cluster_method, infering_method, EMB_method, K, KC_num)
#
# # note 本例专属
# config.jobname_filtered_skill_prob = "{}filtered_skill_prob_.pkl.zip".format(
#     dataset_path)
# config.jobname_train_group_c = \
#     "{}cluster_method={}/infering_methods={}/train_group_c_EMB_method={}_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path,cluster_method, infering_method, EMB_method, K, KC_num)
# config.jobname_test_group_c = \
#     "{}cluster_method={}/infering_methods={}/test_group_c_EMB_method={}_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path,cluster_method, infering_method, EMB_method, K, KC_num)
ONE_KEYS = ["fold", "uid"]


class DKTDataset(Dataset):
    def __init__(self, group, min_samples=3, max_seq=100, max_skill_id=20000, is_fusion_mean=False):
        '''
        魔改数据集
        :param group:
        :param min_samples:
        :param max_seq:
        :param max_skill_id:
        :param is_fusion_mean: note True: early_mean (early_fusion：(concepts)取该题对应的所有知识点的hidden state求平均，再输入到prediction模块)
        '''
        super(DKTDataset, self).__init__()
        self.max_seq = max_seq
        self.samples = {}
        # self.skill_embdding = nn.Embedding(max_skill_id+1, 3)  #TODO 先不emb 直插试试
        self.answer_embdding = nn.Embedding(2, 3)  # TODO 先不emb 直插试试
        self.is_fusion_mean = is_fusion_mean

        self.user_ids = []
        for stu_group, user_id in group.index:
            cur_group = group.loc[:, user_id].values[0]
            s = cur_group[0]    # s 是skill_id
            q = cur_group[1]    # c_emb
            qa = cur_group[2]   # responses
            is_repeat = cur_group[3]
            # s, q, qa = group.loc[:, user_id] # s 是skill_id

            if is_fusion_mean == False:
                if len(s) < min_samples:
                    continue
                # Main Contribution
                if len(s) > self.max_seq:
                    total_questions = len(s)
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
            elif is_fusion_mean == True:
                dcur = {}
                dcur['skill_id'] = s
                dcur['c_emb'] = q
                dcur['responses'] = qa
                dcur['is_repeat'] = is_repeat
                # asd
                dexpand, _ = self.expand_question(dcur, -1)
                for i in range(len(dexpand["skill_id"])):
                    cur_s = dexpand['skill_id'][i]
                    cur_q = dexpand['c_emb'][i]
                    cur_qa = dexpand['responses'][i]
                    cur_masks = np.array(dexpand["selectmasks"][i])
                    if len(cur_s) < min_samples:
                        continue
                    # Main Contribution
                    if len(cur_s) > self.max_seq:
                        total_questions = len(cur_s)
                        initial = total_questions % self.max_seq
                        if initial >= min_samples:
                            self.user_ids.append(f"{user_id}_{i}_0")
                            # self.samples[f"{user_id}_0"] = (cur_q[:initial], cur_qa[:initial])
                            self.samples[f"{user_id}_{i}_0"] = (cur_s[:initial], cur_q[:initial], cur_qa[:initial],cur_masks[:initial])
                        for seq in range(total_questions // self.max_seq):
                            self.user_ids.append(f"{user_id}_{i}_{seq + 1}")
                            start = initial + seq * self.max_seq
                            end = start + self.max_seq
                            self.samples[f"{user_id}_{i}_{seq + 1}"] = (cur_s[start:end], cur_q[start:end], cur_qa[start:end],cur_masks[start:end])
                    else:
                        self.user_ids.append(f"{str(user_id)}_{i}")
                        self.samples[f"{str(user_id)}_{i}"] = (cur_s, cur_q, cur_qa, cur_masks)


    def expand_question(self, dcur, global_qidx, pad_val=0):
        dextend, dlast = dict(), dict()
        repeats = dcur["is_repeat"]
        last = -1
        # dcur["qidxs"], dcur["rest"], global_qidx = add_qidx(dcur, global_qidx)
        for i in range(len(repeats)):
            if str(repeats[i]) == "0":
                for key in dcur.keys():
                    if key in ONE_KEYS:
                        continue
                    dlast[key] = dcur[key][0: i]
            if i == 0:      # 第0个初始化dextend
                for key in dcur.keys():
                    if key in ONE_KEYS:
                        continue
                    dextend.setdefault(key, [])
                    dextend[key].append(np.array([dcur[key][0]]))
                dextend.setdefault("selectmasks", [])
                dextend["selectmasks"].append([pad_val])
            else:
                # print(f"i: {i}, dlast: {dlast.keys()}")
                for key in dcur.keys():
                    if key in ONE_KEYS:
                        continue
                    if key in ['c_emb']:
                        dextend.setdefault(key, [])
                        if last == "0" and str(repeats[i]) == "0":
                            # dextend[key][-1] += [dcur[key][i]]
                            dextend[key][-1] = np.append(dextend[key][-1], [dcur[key][i]], axis=0)
                        else:
                            # dextend[key].append(dlast[key] + [dcur[key][i]])
                            dextend[key].append(np.append(dlast[key], [dcur[key][i]], axis=0))

                    else:
                        dextend.setdefault(key, [])
                        if last == "0" and str(repeats[i]) == "0":
                            # dextend[key][-1] += [dcur[key][i]]
                            dextend[key][-1] = np.append(dextend[key][-1], [dcur[key][i]])
                        else:
                            # dextend[key].append(dlast[key] + [dcur[key][i]])
                            dextend[key].append(np.append(dlast[key], [dcur[key][i]]))
                dextend.setdefault("selectmasks", [])
                if last == "0" and str(repeats[i]) == "0":
                    dextend["selectmasks"][-1] += [1]
                elif len(dlast["responses"]) == 0:  # the first question
                    dextend["selectmasks"].append([pad_val])
                else:
                    dextend["selectmasks"].append(len(dlast["responses"]) * [pad_val] + [1])

            last = str(repeats[i])

        return dextend, global_qidx

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        cur_sample = self.samples[user_id]
        s_, q_, qa_ = cur_sample[0],cur_sample[1],cur_sample[2]
        # NOTE mean
        if self.is_fusion_mean==True:
            selemask_ = cur_sample[3]
        seq_len = len(q_)

        # NOTE : 加一条 因为只有一条数据的时候 ，q_ qa_ 不为二维，就强行变成二维
        if len(q_.shape) == 1:
            q_ = q_[None]
        # if len(qa_.shape) == 1:
        #     qa_ = [qa_]

        # q = np.zeros((self.max_seq, q_.shape[1]))
        q = np.full((self.max_seq, q_.shape[1]), 0.0)  # torch.full((self.max_seq, q_.shape[1]), -10000)   # NOTE 很多情况下需要使用0 但是后面筛选掉0 导致BUG,  -10000使
        qa = np.zeros(self.max_seq, dtype=int)
        target_mask = np.full(self.max_seq, False)  # 标记非空
        early_mean_mask = np.full(self.max_seq, False)
        # qa = np.full(self.max_seq, 0.001)

        if seq_len == self.max_seq:  # TODO  传入后后得到了整数？？
            q[:] = q_
            qa[:] = qa_
            target_mask[:] = True
            if self.is_fusion_mean==True:
                early_mean_mask[:] = selemask_
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
            target_mask[-seq_len:] = True
            if self.is_fusion_mean == True:
                early_mean_mask[-seq_len:] = selemask_
        # X_emb TODO 插入学生特征
        # NOTE _x_emb需要额外处理
        # WHAT 对于双向LSTM来说 可能把标签也学习进去了 所以尝试去掉回答标签 只输入题目信息，或者有其他方法？
        x_emb = self.onehot1(q[:-1], qa[:-1])  # NOTE 投入的特征是no1-49
        # x_emb = torch.cat([torch.Tensor(q[:-1]), torch.Tensor(qa[:-1]).long()], dim=1)
        # x_emb = q[:-1]  # NOTE 修改model参数
        # x_emb = torch.cat([torch.Tensor(q[:-1]), self.answer_embdding(torch.Tensor(qa[:-1]).long())], dim=1) # WHAT 错误XX NOTE 修改model参数, 增加qa       self.answer_embdding(torch.Tensor(qa[:-1]).long())

        q_next = q[1:]  # NOTE 用于预测的是no2-50
        labels = qa[1:]
        target_mask_next = target_mask[1:]

        # NOTE dataloder不支持ndaary
        x_emb_torch = torch.Tensor(x_emb)
        # x_emb_torch = x_emb
        q_next_torch = torch.Tensor(q_next)
        labels_torch = torch.Tensor(labels)
        target_mask_next_torch = torch.Tensor(target_mask_next)

        # NOTE early mean mask:
        if self.is_fusion_mean == True:
            target_early_mean_mask_next = early_mean_mask[1:]
        else:
            target_early_mean_mask_next = np.full(self.max_seq-1, False)
            target_early_mean_mask_next[-1] = True
        # target_early_mean_mask_next[-1] = True
        target_early_mean_mask_next = torch.Tensor(target_early_mean_mask_next)

        return [x_emb_torch, q_next_torch, labels_torch, target_mask_next_torch, target_early_mean_mask_next]

    def onehot1(self, questions, answers):
        emb_num = questions.shape[-1]
        result = np.zeros(shape=[self.max_seq - 1, 2 * emb_num])

        for i in range(self.max_seq - 1):
            if answers[i] > 0:
                result[i][:emb_num] = questions[i]
            elif answers[i] == 0:
                result[i][emb_num:] = questions[i]
        return result

    def onehot2(self, questions, answers):
        emb_num = questions.shape[-1]
        result = np.zeros(shape=[self.max_seq - 1, emb_num + 1])

        for i in range(self.max_seq - 1):
            result[i] = np.append(questions[i], answers[i])
        return result

    def onehot3(self, questions, answers):
        emb_num = questions.shape[-1]
        result = np.zeros(shape=[self.max_seq - 1, emb_num + 1])

        for i in range(self.max_seq - 1):
            result[i] = np.append(questions[i], answers[i])
        return result


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(DKT, self).__init__()
        self.device = device
        # NOTE Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # NOTE Number of hidden layers
        self.layer_dim = layer_dim

        # NOTE Readin layer
        self.readin_linear = nn.Linear(input_dim*2, input_dim*2)  #

        # NOTE drop out
        self.dropout_layer = nn.Dropout(0.2)

        # RNN NOTE 内置了一个RNN 所以在init里就没有声明隐藏变量/层
        # self.rnn = nn.RNN(input_dim * 2, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        # self.lstm = nn.LSTM(input_dim * 2, hidden_dim, layer_dim, batch_first=True, bidirectional=True)    # NOTE 被onehot了 所以input_dim *2
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)    # NOTE 没onehot了 所以input_dim 不变
        # self.lstm = nn.LSTM(input_dim * 2, hidden_dim, layer_dim, batch_first=True)
        self.gru = nn.GRU(input_dim * 2, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        # self.fc = nn.Linear(hidden_dim * 2 + input_dim, output_dim)     #NOTE 因为 返回层是两层
        # self.fc = nn.Linear(hidden_dim + input_dim, output_dim)  # NOTE BiLSTM *2
        self.fc = nn.Linear(hidden_dim, output_dim)  # NOTE BiLSTM *2
        self.sig = nn.Sigmoid()

    def forward(self, x, q_next):
        # x = self.readin_linear(x)
        # Initialize hidden state with zeros
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)  # NOTE BiLSTM *2
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        # One time step
        # _out_next, (hn, cn) = self.lstm(x, (h0, c0))
        _out_next, hn = self.gru(x, h0)
        out_next = _out_next
        out_next = self.dropout_layer(out_next)

        # out_next = torch.cat((out_next, q_next), axis=2)  # TODO 貌似能防止过拟合？？  TODO 没用！
        #         out_next = torch.cat((out.reshape(-1,hidden_dim), q_next.reshape(-1,20)), axis=1)
        # out = self.sig(self.fc(out_next))
        out = self.fc(out_next)  # NOTE  SIG激活函数使得loss变化 检查全连接层
        return out


def train_fn(model, dataloader, optimizer, criterion, scheduler=None, device="cpu"):
    model.train()
    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []
    out_mean = []
    labels_mean = []

    for data in dataloader:

        x = data[0].to(device).float()
        q_next = data[1].to(device).float()
        y = data[2].to(device).float()
        target_mask = data[3].to(device).bool()
        target_mask_mean = data[4].to(device).bool()

        out = model(x, q_next).squeeze()  # [:, :-1]

        # loss = criterion(out, y)
        # loss.backward()
        # optimizer.step()  # NOTE adam
        # # scheduler.step()      # NOTE
        # train_loss.append(loss.item())

        # target_mask = (q_next != -10000).unique(dim=2).squeeze()        # NOTE [batch size, max_seq -1] type
        #         target_mask = (y!=-1)

        filtered_out = torch.masked_select(out, target_mask)
        filtered_label = torch.masked_select(y, target_mask)

        loss = criterion(filtered_out, filtered_label)
        loss.backward()
        optimizer.step()  # NOTE adam
        # scheduler.step()      # NOTE
        train_loss.append(loss.item())

        # filtered_pred = (torch.sigmoid(filtered_out) >= 0.5).long()
        filtered_pred = filtered_out  # NOTE 在forward中已经sigmoid了过了  但是是否应当继续 >=0.5 ?

        # num_corrects += (filtered_pred == filtered_label).sum().item()
        num_corrects += ((filtered_out >= 0.5).long() == filtered_label).sum().item()

        num_total += len(filtered_label)

        labels.extend(filtered_label.view(-1).data.cpu().numpy())
        outs.extend(filtered_pred.view(-1).data.cpu().numpy())

        # # NOTE early_mean
        filtered_out_mean = torch.masked_select(out, target_mask_mean)
        filtered_label_mean = torch.masked_select(y, target_mask_mean)
        out_mean.extend(filtered_out_mean.view(-1).data.cpu().numpy())
        labels_mean.extend(filtered_label_mean.view(-1).data.cpu().numpy())


    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc


def valid_fn(model, dataloader, criterion, device="cpu"):
    model.eval()

    valid_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    labels_mean = []
    outs = []
    outs2 = []
    out_mean = []

    for data in dataloader:

        x = data[0].to(device).float()
        q_next = data[1].to(device).float()
        y = data[2].to(device).float()
        target_mask = data[3].to(device).bool()
        target_mask_mean = data[4].to(device).bool()


        out = model(x, q_next).squeeze()  # [:, :-1]

        loss = criterion(out, y)
        valid_loss.append(loss.item())

        # target_mask = (q_next != -10000).unique(dim=2).squeeze()
        #         target_mask = (y!=-1)

        filtered_out = torch.masked_select(out, target_mask)
        filtered_label = torch.masked_select(y, target_mask)
        filtered_pred = filtered_out  # NOTE 在forward中已经sigmoid了过了  但是是否应当继续 >=0.5 ?
        filtered_pred2 = (filtered_out >= 0.5).long()

        # num_corrects += (filtered_pred == filtered_label).sum().item()
        num_corrects += ((filtered_out >= 0.5).long() == filtered_label).sum().item()
        num_total += len(filtered_label)

        # NOTE mean
        filtered_out_mean = torch.masked_select(out, target_mask_mean)
        filtered_label_mean = torch.masked_select(y, target_mask_mean)
        out_mean.extend(filtered_out_mean.view(-1).data.cpu().numpy())
        labels_mean.extend(filtered_label_mean.view(-1).data.cpu().numpy())

        labels.extend(filtered_label.view(-1).data.cpu().numpy())
        outs.extend(filtered_pred.view(-1).data.cpu().numpy())
        outs2.extend(filtered_pred2.view(-1).data.cpu().numpy())


    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    auc2 = roc_auc_score(labels, outs2)
    auc_mean = roc_auc_score(labels_mean, out_mean)
    # pre = precision_score(labels, outs)
    # f1 = f1_score(labels, outs)
    # rec = recall_score(labels, outs)
    loss = np.mean(valid_loss)

    # return loss, acc, pre, rec, f1, auc
    return loss, acc, auc, auc2, auc_mean


def train_DKT():
    '''

    :param _input_dim:
    :return:
    '''
    # NOTE wandb
    wandb.init(project=f"test-project-MYKT-V7", entity="kumi")
    # wandb.config = {
    #     "learning_rate": 0.0001,
    #     "epochs": 500,
    #     "batch_size": 64
    # }

    # Model settings
    input_dim = _input_dim  # input dimension #NOTE 输入大小应当是我潜入大小？  原来是input_dim = 20
    hidden_dim = 128  # hidden layer dimension
    layer_dim = 1  # number of hidden layers # NOTE 隐藏层数
    output_dim = 1  # output dimension
    MAX_LEARNING_RATE = 1e-3  # NOTE 最大学习率
    LEARNING_RATE = 1e-5  # NOTE 学习率
    EPOCHS = 200  # NOTE 论述      # NOTE 轮数会影响到scheduler 工作
    BATCH_SIZE = 64
    MAX_SEQ = 150

    print("model param:")
    print("input_dim", input_dim)
    print("hidden_dim", hidden_dim)
    print("layer_dim", layer_dim)
    print("output_dim", output_dim)
    print("MAX_LEARNING_RATE", MAX_LEARNING_RATE)
    print("LEARNING_RATE", LEARNING_RATE)
    print("EPOCHS", EPOCHS)
    print("BATCH_SIZE", BATCH_SIZE)
    print("MAX_SEQ", MAX_SEQ)

    train_group_c = joblib.load(config.jobname_train_group_c)
    test_group_c = joblib.load(config.jobname_test_group_c)
    print("load file:", config.jobname_train_group_c)
    print("load file:", config.jobname_test_group_c)

    train_dataset = DKTDataset(train_group_c, max_seq=MAX_SEQ)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_fusion_mean_dataset = DKTDataset(test_group_c, max_seq=MAX_SEQ, is_fusion_mean=True)
    test_fusion_mean_dataloader = DataLoader(test_fusion_mean_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # TODO 改造GAUC
    valid_group_dataloder_list = []
    valid_group_early_mean_dataloder_list = []
    for k in range(K):  # test_group_c[[i for i in test_group_c.index if i[0]==0]]
    # for k in range(2):  # test_group_c[[i for i in test_group_c.index if i[0]==0]]
    #     print("K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2")
    #     print("K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2")
    #     print("K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2")
    #     print("K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2K=2")
        cur_test_group_c = test_group_c[[i for i in test_group_c.index if i[0] == k]]   # 虽然看起来很复杂 但是有用就行啦
        valid_dataset = DKTDataset(cur_test_group_c, max_seq=MAX_SEQ)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_group_dataloder_list.append(valid_dataloader)
        valid_dataset = DKTDataset(cur_test_group_c, max_seq=MAX_SEQ, is_fusion_mean=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_group_early_mean_dataloder_list.append(valid_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ("cuda" if torch.cuda.is_available() else "cpu")

    model = DKT(input_dim, hidden_dim, layer_dim, output_dim, device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  #

    criterion = nn.BCEWithLogitsLoss()  # BCELoss
    # criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
    )

    model.to(device)
    criterion.to(device)

    max_train_auc = 0
    max_valid_auc = 0
    max_train_auc_epoch = 0
    max_valid_auc_epoch = 0
    max_test_early_mean_auc = 0.0
    max_avg_early_mean_auc = 0.0
    max_avg_early_mean_acc = 0.0
    max_avg_auc = 0
    test_mean_auc = 0
    test_mean_auc_2 = 0
    avg_early_mean_auc = 0
    train_loss_list = []
    train_auc_list = []
    test_loss_list = []
    test_auc_list = []

    for epoch in (range(EPOCHS)):
        # NOTE tarin_fn & valid_fn
        train_loss, acc, train_auc = train_fn(model, train_dataloader, optimizer, criterion, device=device)
        print(" ")
        print("epoch - {}/{} train: - {:.3f} acc - {:.3f} auc - {:.3f}".format(epoch + 1, EPOCHS, train_loss, acc, train_auc))
        test_auc_list_bygroup = []
        for valid_dataloader in valid_group_dataloder_list:
            test_loss, acc, test_auc, test_auc2, test_auc_mean = valid_fn(model, valid_dataloader, criterion, device)
            test_auc_list_bygroup.append(test_auc)
        avg_auc = sum(test_auc_list_bygroup) / len(test_auc_list_bygroup)
        print("epoch - {}/{} valid: - {:.3f} acc - {:.3f} avg_auc - {:.3f} auc2 - {:.3f} auc_mean - {:.3f}".format(epoch + 1, EPOCHS, test_loss, acc, avg_auc, test_auc2, test_auc_mean))
        print("test_auc_list_bygroup:", test_auc_list_bygroup)

        # NOTE early_mean
        test_early_mean_auc_list_bygroup = []
        test_early_mean_acc_list_bygroup = []

        if epoch % 10 == 0:
            print("-" * 50)
            test_mean_loss, test_mean_acc, test_mean_auc, test_mean_auc_05, test_early_mean_auc = valid_fn(model, test_fusion_mean_dataloader, criterion, device)
            print("all_early_mean:")
            print("test_mean_auc:{}, test_early_mean_auc:{}".format(test_mean_auc, test_early_mean_auc))
            max_test_early_mean_auc = max(max_test_early_mean_auc, test_early_mean_auc)
            print("group_early_mean:")
            for group_mean_dataloader in valid_group_early_mean_dataloder_list:
                group_early_mean_test_loss, group_early_mean_acc, group_early_mean_test_auc, group_early_mean_test_auc2, group_early_mean_test_auc_mean = valid_fn(model, group_mean_dataloader, criterion, device)
                test_early_mean_auc_list_bygroup.append(group_early_mean_test_auc_mean)
                test_early_mean_acc_list_bygroup.append(group_early_mean_acc)
            avg_early_mean_auc = sum(test_early_mean_auc_list_bygroup) / len(test_early_mean_auc_list_bygroup)
            max_avg_early_mean_auc = max(max_avg_early_mean_auc, avg_early_mean_auc)
            avg_early_mean_acc = sum(test_early_mean_acc_list_bygroup) / len(test_early_mean_acc_list_bygroup)
            max_avg_early_mean_acc = max(max_avg_early_mean_acc, avg_early_mean_acc)
            print("avg_test_early_mean_auc_list_bygroup:", avg_early_mean_auc)
            print("test_early_mean_auc_list_bygroup:", test_early_mean_auc_list_bygroup)
            print("avg_test_early_mean_auc_list_bygroup:", avg_early_mean_acc)
            print("test_early_mean_auc_list_bygroup:", test_early_mean_acc_list_bygroup)
            print("-" * 50)

        # NOTE save max
        if max_train_auc <= train_auc:
            max_train_auc = train_auc
            max_train_auc_epoch = epoch
        if max_valid_auc + 1e-3 < test_auc:  # auc > max_auc+1e-3
            max_valid_auc = test_auc
            max_valid_auc_epoch = epoch
        if avg_auc > max_avg_auc + 1.5e-3:
            max_avg_auc = avg_auc
            max_avg_auc_epoch = epoch
            if epoch > 50:
                print("-" * 50)
                test_mean_loss, test_mean_acc, test_mean_auc, test_mean_auc_05, test_early_mean_auc = valid_fn(model, test_fusion_mean_dataloader, criterion, device)
                print("all_early_mean:")
                print("test_mean_auc:{}, test_early_mean_auc:{}".format(test_mean_auc, test_early_mean_auc))
                max_test_early_mean_auc = max(max_test_early_mean_auc, test_early_mean_auc)
                print("group_early_mean:")
                for group_mean_dataloader in valid_group_early_mean_dataloder_list:
                    group_early_mean_test_loss, group_early_mean_acc, group_early_mean_test_auc, group_early_mean_test_auc2, group_early_mean_test_auc_mean = valid_fn(model, group_mean_dataloader, criterion, device)
                    test_early_mean_auc_list_bygroup.append(group_early_mean_test_auc_mean)
                avg_early_mean_auc = sum(test_early_mean_auc_list_bygroup) / len(test_early_mean_auc_list_bygroup)
                max_avg_early_mean_auc = max(max_avg_early_mean_auc, avg_early_mean_auc)
                print("avg_test_early_mean_auc_list_bygroup:", avg_early_mean_auc)
                print("test_early_mean_auc_list_bygroup:", test_early_mean_auc_list_bygroup)
                print("-" * 50)

        train_loss_list.append(train_loss)
        train_auc_list.append(train_auc)
        test_loss_list.append(test_loss)
        test_auc_list.append(test_auc)

        # NOTE wandb out
        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "train_auc": train_auc,
                   "test_loss": test_loss,
                   "avg_auc": avg_auc,
                   "avg_early_mean_auc": avg_early_mean_auc,
                   "avg_early_mean_auc": avg_early_mean_acc,
                   "test_auc_list_bygroup": test_auc_list_bygroup,
                   "test_early_mean_auc_list_bygroup": test_early_mean_auc_list_bygroup,
                   "test_mean_auc": test_mean_auc,
                   "test_early_mean_auc": test_early_mean_auc
                   })

        # Optional
        # wandb.watch(model)
        # if epoch - max_valid_auc_epoch >= 10 and epoch > 150:
        #     break

    wandb.log({
        "max_avg_auc": max_avg_auc,
        "max_avg_auc_epoch": max_avg_auc_epoch,
        "max_avg_early_mean_auc": max_avg_early_mean_auc
    })
    print("max_train_auc_epoch: {} , max_train_auc: {}".format(max_train_auc_epoch, max_train_auc))
    print("max_valid_auc_epoch: {} , max_valid_auc: {}".format(max_valid_auc_epoch, max_valid_auc))
    # show_me_see_see(len(train_loss),train_loss,train_auc,test_loss,test_auc)

def wandb_train_DKT():
    '''

    :param _input_dim:
    :return:
    '''
    # NOTE wandb
    # wandb.init(project="test-project", entity="kumi_team")
    wandb.init()

    # NOTE 重新roll点轮
    # df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_assist2009()
    # MYKT_StuGroup_and_diff_dataprocess.stu_cluster_assist09(df, NAN_FULL)
    # MYKT_StuGroup_and_diff_dataprocess.extract_c2c_byStuGroup_undirected(df)   # NOTE 计算无向图
    get_train_group_assist(AllK_group_skill_emb_list)


    # Model settings
    input_dim = _input_dim  # input dimension #NOTE 输入大小应当是我潜入大小？  原来是input_dim = 20
    hidden_dim = wandb.config.hidden_dim  # hidden layer dimension
    layer_dim = wandb.config.layer_dim  # number of hidden layers # NOTE 隐藏层数
    output_dim = 1  # output dimension
    MAX_LEARNING_RATE = 1e-3  # NOTE 最大学习率
    LEARNING_RATE = wandb.config.learning_rate  # 1e-4        # NOTE 学习率
    EPOCHS = 200  # NOTE 论述      # NOTE 轮数会影响到scheduler 工作
    BATCH_SIZE = wandb.config.batch_size
    MAX_SEQ = wandb.config.max_seq
    # DROPOUT = wandb.config.dropout
    print(wandb.config)
    print(wandb.config.batch_size)
    print("model param:")
    print("input_dim", input_dim)
    print("hidden_dim", hidden_dim)
    print("layer_dim", layer_dim)
    print("output_dim", output_dim)
    print("MAX_LEARNING_RATE", MAX_LEARNING_RATE)
    print("LEARNING_RATE", LEARNING_RATE)
    print("EPOCHS", EPOCHS)
    print("BATCH_SIZE", BATCH_SIZE)
    print("MAX_SEQ", MAX_SEQ)

    train_group_c = joblib.load(config.jobname_train_group_c)
    test_group_c = joblib.load(config.jobname_test_group_c)
    print("load file:", config.jobname_train_group_c)
    print("loda file:", config.jobname_test_group_c)

    train_dataset = DKTDataset(train_group_c, max_seq=MAX_SEQ)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_fusion_mean_dataset = DKTDataset(test_group_c, max_seq=MAX_SEQ, is_fusion_mean=True)
    test_fusion_mean_dataloader = DataLoader(test_fusion_mean_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # TODO 改造GAUC
    valid_group_dataloder_list = []
    valid_group_early_mean_dataloder_list = []
    for k in range(K):  # test_group_c[[i for i in test_group_c.index if i[0]==0]]
        cur_test_group_c = test_group_c[[i for i in test_group_c.index if i[0] == k]]   # 虽然看起来很复杂 但是有用就行啦
        valid_dataset = DKTDataset(cur_test_group_c, max_seq=MAX_SEQ)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_group_dataloder_list.append(valid_dataloader)
        valid_dataset = DKTDataset(cur_test_group_c, max_seq=MAX_SEQ, is_fusion_mean=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_group_early_mean_dataloder_list.append(valid_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ("cuda" if torch.cuda.is_available() else "cpu")

    model = DKT( input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim, device=device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  #

    criterion = nn.BCEWithLogitsLoss()  # BCELoss
    # criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
    )

    model.to(device)
    criterion.to(device)

    max_train_auc = 0
    max_valid_auc = 0
    max_train_auc_epoch = 0
    max_valid_auc_epoch = 0
    max_test_early_mean_auc = 0.0
    max_avg_early_mean_auc = 0.0
    max_avg_early_mean_acc = 0.0
    max_avg_auc = 0
    test_mean_auc = 0
    test_mean_auc_2 = 0
    avg_early_mean_auc = 0
    train_loss_list = []
    train_auc_list = []
    test_loss_list = []
    test_auc_list = []

    for epoch in (range(EPOCHS)):
        # NOTE tarin_fn & valid_fn
        train_loss, acc, train_auc = train_fn(model, train_dataloader, optimizer, criterion, device=device)
        print(" ")
        print("epoch - {}/{} train: - {:.3f} acc - {:.3f} auc - {:.3f}".format(epoch + 1, EPOCHS, train_loss, acc, train_auc))
        test_auc_list_bygroup = []
        for valid_dataloader in valid_group_dataloder_list:
            test_loss, acc, test_auc, test_auc2, test_auc_mean = valid_fn(model, valid_dataloader, criterion, device)
            test_auc_list_bygroup.append(test_auc)
        avg_auc = sum(test_auc_list_bygroup)/len(test_auc_list_bygroup)
        print("epoch - {}/{} valid: - {:.3f} acc - {:.3f} avg_auc - {:.3f} auc2 - {:.3f} auc_mean - {:.3f}".format(epoch + 1, EPOCHS, test_loss, acc, avg_auc, test_auc2, test_auc_mean))
        print("test_auc_list_bygroup:", test_auc_list_bygroup)

        # NOTE early_mean
        test_early_mean_auc_list_bygroup = []
        test_early_mean_acc_list_bygroup = []

        if epoch % 10 == 0:
            print("-" * 50)
            test_mean_loss, test_mean_acc, test_mean_auc, test_mean_auc_05, test_early_mean_auc = valid_fn(model, test_fusion_mean_dataloader, criterion, device)
            print("all_early_mean:")
            print("test_mean_auc:{}, test_early_mean_auc:{}".format(test_mean_auc, test_early_mean_auc))
            max_test_early_mean_auc = max(max_test_early_mean_auc, test_early_mean_auc)
            print("group_early_mean:")
            for group_mean_dataloader in valid_group_early_mean_dataloder_list:
                group_early_mean_test_loss, group_early_mean_acc, group_early_mean_test_auc, group_early_mean_test_auc2, group_early_mean_test_auc_mean = valid_fn(model, group_mean_dataloader, criterion, device)
                test_early_mean_auc_list_bygroup.append(group_early_mean_test_auc_mean)
                test_early_mean_acc_list_bygroup.append(group_early_mean_acc)
            avg_early_mean_auc = sum(test_early_mean_auc_list_bygroup) / len(test_early_mean_auc_list_bygroup)
            max_avg_early_mean_auc = max(max_avg_early_mean_auc, avg_early_mean_auc)
            avg_early_mean_acc = sum(test_early_mean_acc_list_bygroup) / len(test_early_mean_acc_list_bygroup)
            max_avg_early_mean_acc = max(max_avg_early_mean_acc, avg_early_mean_acc)
            print("avg_test_early_mean_auc_list_bygroup:", avg_early_mean_auc)
            print("test_early_mean_auc_list_bygroup:", test_early_mean_auc_list_bygroup)
            print("avg_test_early_mean_acc_list_bygroup:", avg_early_mean_acc)
            print("test_early_mean_acc_list_bygroup:", test_early_mean_acc_list_bygroup)
            print("-" * 50)

        # NOTE save max
        if max_train_auc <= train_auc:
            max_train_auc = train_auc
            max_train_auc_epoch = epoch
        if max_valid_auc+1e-3 < test_auc:   # auc > max_auc+1e-3
            max_valid_auc = test_auc
            max_valid_auc_epoch = epoch
        if avg_auc > max_avg_auc:
            max_avg_auc = avg_auc
            max_avg_auc_epoch = epoch

        train_loss_list.append(train_loss)
        train_auc_list.append(train_auc)
        test_loss_list.append(test_loss)
        test_auc_list.append(test_auc)

        # NOTE wandb out
        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "train_auc": train_auc,
                   "test_loss": test_loss,
                   "avg_auc": avg_auc,
                   "avg_early_mean_auc": avg_early_mean_auc,
                   "avg_early_mean_acc": avg_early_mean_acc,
                   "test_auc_list_bygroup": test_auc_list_bygroup,
                   "test_early_mean_auc_list_bygroup": test_early_mean_auc_list_bygroup,
                   "test_mean_auc": test_mean_auc,
                   "test_early_mean_auc": test_early_mean_auc,
                   "max_avg_early_mean_auc": max_avg_early_mean_auc,
                    "max_avg_early_mean_acc": max_avg_early_mean_acc

                   })

        # Optional
        # wandb.watch(model)
        # if max_valid_auc <= 0.74 and epoch > 120:
        #     break

    wandb.log({
        "max_avg_auc": max_avg_auc,
        "max_avg_auc_epoch": max_avg_auc_epoch,
        "max_avg_early_mean_auc": max_avg_early_mean_auc,
        "max_avg_early_mean_acc": max_avg_early_mean_acc
    })
    print("max_train_auc_epoch: {} , max_train_auc: {}".format(max_train_auc_epoch, max_train_auc))
    print("max_valid_auc_epoch: {} , max_valid_auc: {}".format(max_valid_auc_epoch, max_valid_auc))
    # show_me_see_see(len(train_loss),train_loss,train_auc,test_loss,test_auc)


def get_train_group_assist(AllK_group_skill_emb_list):  # NOTE 和skill_emb高度挂钩
    # GET DATASET
    print("-------LOAD ", dataset_path)
    if args.dataset_path in ["Data/assist2009/", "Data/assist2009_test/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_assist2009()
        print("lodad")
    elif args.dataset_path in ["Data/assist2017/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_assist2017()
    elif args.dataset_path in ["Data/assist2012/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_assist2012()
    elif args.dataset_path in ["Data/ednet/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_ednet()
    else:
        print("big error!!!!!!!!!!!!!!!!!!!!!!, not the dataset")
    print("skill_id_num:", len(set(df["skill_id"])))

    # NOTE 把学习得到的嵌入加入到数据集中
    df = df[df['skill_id'].isin(filtered_skill_prob.keys())]
    df['skill_cat'] = df['skill_id'].astype('category').cat.codes
    # df = df[df['problem_id'].isin(prob_emb.keys())]
    # df['e_emb'] = df['problem_id'].apply(lambda r: prob_emb[r])


    # NOTE 插入group
    if args.SGC == 1 or args.SGC == 2:  # TODO: 记得改
        all_stuGroup_inDiffKC = joblib.load(config.jobname_all_stuGroup_inDiffKC)
        print("loda file:", config.jobname_all_stuGroup_inDiffKC)
        print("KC_num:", KC_num)
        all_stuGroup = all_stuGroup_inDiffKC.iloc[:, KC_num]
        df['group'] = df['user_id'].apply(lambda r: all_stuGroup[r])
    elif args.SGC == 3:
        # TODO 等待测试
        print("未经过测试")
        all_stuGroup = joblib.load(config.jobname_all_stuGroup_bySGC)
        df['group'] = df['user_id'].apply(lambda r: all_stuGroup[r])
    elif args.SGC == 5:
        schools_map = joblib.load(config.jobname_schoolsID_index)
        print("file load:", config.jobname_schoolsID_index)
        df['group'] = df['school_id'].apply(lambda r: schools_map[r])  # NOTE 根据school_id添加group
    elif args.SGC == 11:
        all_stuGroup = joblib.load(config.jobname_all_stuGroup_bySGC)
        df['group'] = df['user_id'].apply(lambda r: all_stuGroup.loc[r])
    else:
        print("BIG ERROR NO SGC!")
        print("BIG ERROR NO SGC!")
        print("BIG ERROR NO SGC!")
        print("BIG ERROR NO SGC!")

    print("start insert c_emb")
    df["c_emb"] = df.apply(lambda row: AllK_group_skill_emb_list[int(row.group)][int(row.skill_id)], axis=1)

    # df['skill_index'] = df['skill_id'].apply()


    group_c = df[['user_id', 'skill_id', 'c_emb', 'correct', 'group', 'is_repeat']].groupby(['group', 'user_id']).apply(
        lambda r: (r['skill_id'].values, np.array(r['c_emb'].tolist()).squeeze(), r['correct'].values, r['is_repeat'].values))
    # train_group_c = group_c.sample(frac=0.8, random_state=2022)

    # note 控制对比实验变量
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int(len(uid) * 0.8)
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]
    #
    # train_group_c = train_df[['user_id', 'skill_id', 'c_emb', 'correct', 'group']].groupby(['group', 'user_id']).apply(
    #     lambda r: (r['skill_id'].values, np.array(r['c_emb'].tolist()).squeeze(), r['correct'].values))
    # test_group_c = test_df[['user_id', 'skill_id', 'c_emb', 'correct', 'group']].groupby(['group', 'user_id']).apply(
    #     lambda r: (r['skill_id'].values, np.array(r['c_emb'].tolist()).squeeze(), r['correct'].values))

    # 通过时间获得一个随机数
    # random.seed(time.time())
    # rand_num = random.random()
    # print("随机数为：", int(rand_num*10000))
    # _train_group_c = group_c.sample(frac=0.8, random_state=int(rand_num*10000))
    _train_group_c = group_c.sample(frac=0.8, random_state=2023)
    test_group_c = group_c[~group_c.index.isin(_train_group_c.index)]
    # train_group_c = _train_group_c.sample(frac=0.8, random_state=2023)
    # valid_group_c = _train_group_c[~_train_group_c.index.isin(train_group_c.index)]

    joblib.dump(_train_group_c, config.jobname_train_group_c)
    joblib.dump(test_group_c, config.jobname_test_group_c)
    # joblib.dump(valid_group_c,config.jobname_valid_group_c)
    print("file save:",config.jobname_train_group_c)
    print("file save:",config.jobname_test_group_c)
    # print("file save:",config.jobname_valid_group_c)


def normalization(c2c_emb):
    # normalization
    scaler = StandardScaler()
    all_c_v = []
    for k, v in c2c_emb.items():
        all_c_v.extend(list(v.numpy()))
    all_c_v = scaler.fit_transform(np.array(all_c_v).reshape(-1, 1))

    all_c_v1 = {}
    length_c2cEmb = c2c_emb.get(next(iter(c2c_emb))).shape[0]  # NOTE
    print("got from GAE, the c2c_Emb length:", length_c2cEmb)
    for i, (k, v) in enumerate(c2c_emb.items()):
        all_c_v1[k] = all_c_v[i * length_c2cEmb:(i + 1) * length_c2cEmb].reshape(-1, )  # TODO 这里就不是乘以10了

    """       
    all_e_v = {}
    for skill, qu_embs in e2e_emb.items():
        q_num = qu_embs.shape[0]
        temp_all_v = qu_embs.numpy().reshape(-1, )
        temp_all_v = scaler.fit_transform(np.array(temp_all_v).reshape(-1, 1))
        all_e_v[skill] = temp_all_v.reshape(-1, 10)
    """

    # TODO 训练skill嵌入和prob的嵌入
    skill_emb = {}
    # for skill in tqdm(filtered_skill_prob.keys()):
    #     try:
    #         temp_c = (np.array(all_c_v1[skill]))
    #         # temp_e = np.array(np.mean(all_e_v[skill], axis=0))
    #         # skill_emb[skill] = np.append(temp_c, temp_e)
    #         skill_emb[skill] = temp_c
    #     except:
    #         continue
    # TODO 训练skill嵌入和prob的嵌入
    for skill in tqdm(c2c_emb.keys()):
        try:
            temp_c = (np.array(all_c_v1[skill]))
            # temp_e = np.array(np.mean(all_e_v[skill], axis=0))
            # skill_emb[skill] = np.append(temp_c, temp_e)
            skill_emb[skill] = temp_c
        except:
            continue

    '''
    prob_emb = {}
    for skill in tqdm(filtered_skill_prob.keys()):
        for i, prob in enumerate(filtered_skill_prob[skill]):
            temp_c = (np.array(all_c_v1[skill]))
            temp_e = (np.array(all_e_v[skill][i]))
            new_emb = np.append(temp_c, temp_e)
            if prob in prob_emb.keys():
                prob_emb[prob] = np.row_stack((prob_emb[prob], new_emb)).squeeze()
            #             print(prob_emb[prob].shape)
            else:
                prob_emb[prob] = new_emb

    for prob in tqdm(prob_emb.keys()):
        if len(prob_emb[prob].shape) > 1:
            prob_emb[prob] = np.mean(prob_emb[prob], axis=0)
    '''
    return skill_emb


def get_all_group_skill_emb_concatALL():
    c2c_emb = joblib.load(config.jobname_all_c2c_emb_bystuGroup)
    skill_prob = joblib.load(config.jobname_skill_df)
    print("load file:", config.jobname_all_c2c_emb_bystuGroup)
    all_skill_diff = joblib.load(config.jobname_all_skill_diff)
    print("loda file:", config.jobname_all_skill_diff)
    all_skill_group_first_reaction = joblib.load(config.jobname_all_skill_group_first_reaction)
    print("loda file:", config.jobname_all_skill_group_first_reaction)
    all_skill_group_time = joblib.load(config.jobname_all_skill_group_time)
    print("loda file:", config.jobname_all_skill_group_time)
    print("c2c_emb_length:", c2c_emb[0].get(next(iter(c2c_emb[0]))).shape[0])

    # NOTE 定义一些初始化变量
    global K
    K = len(c2c_emb)
    max_time = 0
    for temp in all_skill_group_time:
        max_time = max(temp.max(), max_time)
    max_time = int(max_time)
    skill_time_emb_length = 1
    print("nn.Embedding skill_diff_emb_length:", skill_time_emb_length)
    skill_time_embedding = nn.Embedding(max_time+1, skill_time_emb_length)


    AllK_group_skill_emb = []
    for k in range(0, K):
        # note c2c
        cur_c2c_emb = c2c_emb[k]
        cur_c2c_emb_nomaled = normalization(cur_c2c_emb)

        # note skill_diff
        cur_skill_diff = all_skill_diff[k]

        # note skill_group_first_reaction
        cur_group_first_reaction = {}
        for index, row in all_skill_group_first_reaction[k].iterrows():
            cur_group_first_reaction[index] = torch.Tensor(row.values)
        cur_skill_group_first_reaction_nomaled = normalization(cur_group_first_reaction)    # TODO 这样做好像不对，三个特征都是有区别的，不可以同尺度正则化吧

        # note skill_group_time
        cur_skill_group_time_embed = {}
        for key, value in all_skill_group_time[k].items():
            cur_skill_group_time_embed[key] = skill_time_embedding(torch.Tensor(np.array(value)).long()).detach().numpy()

        cur_skill_emb = {}
        for key,value in cur_c2c_emb_nomaled.items():
            try:
                cur_c2c_emb_nomaled[key] = np.append(cur_c2c_emb_nomaled[key], np.array(cur_skill_diff[key]))
            except:
                cur_c2c_emb_nomaled[key] = np.append(cur_c2c_emb_nomaled[key], 0.0)

            try:
                cur_c2c_emb_nomaled[key] = np.append(cur_c2c_emb_nomaled[key], cur_skill_group_first_reaction_nomaled[key])
            except:
                cur_c2c_emb_nomaled[key] = np.append(cur_c2c_emb_nomaled[key], np.array([0.0,0.0,0.0]))

            try:
                cur_c2c_emb_nomaled[key] = np.append(cur_c2c_emb_nomaled[key], cur_skill_group_time_embed[key])
            except:
                cur_c2c_emb_nomaled[key] = np.append(cur_c2c_emb_nomaled[key], 0.0)


        AllK_group_skill_emb.append(cur_c2c_emb_nomaled)


    return AllK_group_skill_emb


if __name__ == '__main__':
    # NOTE 写的有点乱 总之 能run就行
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', torch.cuda.is_available())
    print('torch version:', torch.__version__)  # 查看torch当前版本号'

    # e2e_emb = joblib.load('Data/assist2009/e2e_emb.pkl.zip')
    # GET C2C_emb_concated
    skill_prob = joblib.load(config.jobname_skill_df)
    print("load file:", config.jobname_skill_df)

    # NOTE 过滤
    max_skill_index = skill_prob.size
    filtered_skill_prob = {}
    channel = 10
    # NOTE : 过滤 只有题目数量大于channel的才会被留下
    for i, skill_id in enumerate(skill_prob.index):
        if (len(skill_prob[skill_id]) >= channel) or (skill_prob[skill_id] == ['NA']):
            filtered_skill_prob[skill_id] = skill_prob[skill_id]
    joblib.dump(filtered_skill_prob, config.jobname_filtered_skill_prob)


    AllK_group_skill_emb_list = get_all_group_skill_emb_concatALL()   # skill_emb 是一个list


    # NOTE 分训练集
    get_train_group_assist(AllK_group_skill_emb_list)

    # NOTE: DKT Models
    _input_dim = AllK_group_skill_emb_list[0].get(next(iter(AllK_group_skill_emb_list[0]))).shape[0]  # NOTE 参数设置而已
    train_DKT()            # NOTE 一般训练法



    # # TODO 参数搜索========================================================
    # # 第一步：定义sweeps 配置，也就是超参数搜索的方法和范围
    # # 超参数搜索方法，可以选择：grid random bayes NOTE 在config.py里
    #
    # sweep_config['parameters'] = parameters_dict
    # # from pprint import pprint
    # # pprint(sweep_config)
    # # 第二步：初始化sweep
    # # 一旦定义好超参数调优策略和搜索范围，需要一个sweep controller管理调优过程
    # sweep_id = wandb.sweep(
    #     sweep_config,
    #     project="MYKT_v7_2_assist12_随机数",
    #     entity="kumi"
    # )
    #
    # wandb.agent(sweep_id, wandb_train_DKT)
    # # TODO 参数搜索========================================================
