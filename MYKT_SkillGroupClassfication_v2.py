import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import time, joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from sklearn.cluster import SpectralClustering, KMeans

import config
import MYKT_StuGroup_and_diff_dataprocess
import argparse

args = config.get_args()

args = config.get_args()
K = args.K
NAN_FULL = args.NAN_FULL
dataset_path = args.dataset_path
# dataset_name = args.dataset_name
all_stuGroup_file_name = args.all_stuGroup_file_name
KC_num = args.KC_num
infering_method = args.infering_method
cluster_method = args.cluster_method
channels = args.channels  #
EMB_method = args.EMB_method
skill_group_threshold = args.skill_group_threshold
skill_nclusters = args.skill_nclusters
SGC = args.SGC

# Training matrix in Autoencoder
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K, v=2.0):
        super(Encoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = pyg_nn.GCNConv(2 * out_channels, out_channels, cached=True)

        self.cluster_layer = Parameter(torch.Tensor(K, out_channels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

    def forward(self, x, edge_index):
        """

        :param x: 结点特征矩阵
        :param edge_index: 领结矩阵
        :return:
        """
        x = (self.conv1(x, edge_index)).relu()  # note 原文室友relu的？
        self.feature = self.conv2(x, edge_index)
        z = self.feature
        q = self.get_Q(z)
        return self.feature, q  # 就是下面返回的Z？

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def gae_train(epoc_trainh, x, model, optimizer, train_pos_edge_index, Q):
    model.train()
    optimizer.zero_grad()

    if epoc_trainh % 3 == 0: # NOTE [1,3,5]
        # update_interval
        _, Q = model.encode(x, train_pos_edge_index)
        # q = Q.detach().data.cpu().numpy().argmax(1)  # Q

    z, q = model.encode(x, train_pos_edge_index)  # NOTE 在这里调用encoder
    p = target_distribution(Q.detach())  # NOTE 好神奇 自监督？  自己定时更新大Q 然后用来影响loss, 那么loss会影响聚类效果吗？
    kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

    re_loss = model.recon_loss(z, train_pos_edge_index)
    loss = 10 * kl_loss + re_loss

    loss.backward(retain_graph=True)
    optimizer.step()
    # writer.add_scalar("loss", loss.item(), epoch)
    return q.detach().data.cpu().numpy().argmax(1), Q   # NOTE 这里q/Q其实就是最后的聚类类别了


def gae_test(model, x, pos_edge_index, neg_edge_index, train_pos_edge_index):
    model.eval()
    with torch.no_grad():
        z,_ = model.encode(x, train_pos_edge_index)
    # loss = model.recon_loss(z, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def cluster(skill_emb, model_skill_emb):
    # NOTE SpectralClustering
    #  TODO 有问题我解决不了RuntimeWarning: invalid value encountered in sqrt w = np.where(isolated_node_mask, 1, np.sqrt(w))

    # Cluster = SpectralClustering(n_clusters=skill_nclusters, affinity='precomputed', random_state=0)
    # # db, acc, nmi, adjscore = clustering(Cluster, hidden_emb, true_labels)
    # # f_adj = np.matmul(skill_emb, np.transpose(skill_emb))   # NOTE 虽然我也不知道为什么要这样做，但是AGE都这样做了
    # f_adj = np.matmul(model_skill_emb, np.transpose(model_skill_emb))   # NOTE 虽然我也不知道为什么要这样做，但是AGE都这样做了
    # predict_labels = Cluster.fit_predict(f_adj)

    Cluster = KMeans(n_clusters=skill_nclusters)  # n_init=1?
    predict_labels = Cluster.fit_predict(model_skill_emb)

    return predict_labels


def train_skill_emb(model, data, skill_df):
    # NOTE train gae
    labels = data.y  # what: 也没有说这个是要干嘛的
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data, val_ratio=0, test_ratio=0.1)
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)  # NOTE 选择你的训练导师 CPU/GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    Q = 0.0
    for epoch in tqdm(range(0, 1000)):   # 1001 轮
        try:
            skill_group_byQ, Q = gae_train(epoch, x, model, optimizer, train_pos_edge_index, Q)  # # NOTE 在这里不需要传入self
            # NOTE 正样本 就是存在的边，负样本 就是不存在的边
            auc, ap = gae_test(model, x, data.test_pos_edge_index, data.test_neg_edge_index, train_pos_edge_index)  # NOTE 在这里不需要传入self
            if epoch % 100 == 0:
                print('Epoch: {:03d} , AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
        except ValueError:
            if epoch % 1000 == 0:
                print("{}ValueError:auc ap error".format(epoch))
            pass

        skill_emb = {}  # NOTE 这是字典啊
        for i, emb in enumerate(model.encoder.feature.cpu()):
            skill_emb[skill_df.index[i]] = emb

    return skill_emb, model.encoder.feature.cpu(), skill_group_byQ  # NOTE 返回学习的嵌入的字典啊


def get_skill_group_logic(df, skill_df, skill_group_threshold, train_skill_group_logic=False):
    # NOTE 调用预处理MYKT_StuGroup_and_diff_dataprocess的函数即可
    # NOTE 这个训练要花费很多时间,考虑保存
    if train_skill_group_logic:
        skill_group_logic = MYKT_StuGroup_and_diff_dataprocess.extract_c2c_matrix_diredctional(df, skill_df)
        # TODO 保存
        joblib.dump(skill_group_logic, config.jobname_skill_group_logic)
        print("file save:", config.jobname_skill_group_logic)
    else:
        skill_group_logic = joblib.load(config.jobname_skill_group_logic)
        print("file load:", config.jobname_skill_group_logic)

    # NOTE 逻辑设定阈值
    # temp = skill_group_logic._values()
    # temp[np.where(temp < skill_group_threshold)] = 0    # NOTE 使得阈值小于指定值的都为0，后续的torch.nonzero就不会获得他
    skill_group_logic_dense_byshreshold = (skill_group_logic.to_dense() + (1 - skill_group_threshold)).long()  # NOTE (1 - skill_group_threshold)

    # cur_c2c_bystuGroup = all_c2c_bystuGroup[0]  # NOTE tyep:tensor   TODO 这个应该是全部C2C的结果
    # x = torch.tensor(cur_c2c_bystuGroup.toarray().tolist(), dtype=torch.float)
    edge_index = torch.nonzero(skill_group_logic_dense_byshreshold).T  # TODO 稀疏矩阵的邻接矩阵, 取的是里面非0的，非0才是存在边的

    return edge_index


def get_skill_group_feature(df, diff_coef=100):
    # NOTE 定义 一个EMB
    fea_emb1 = nn.Embedding(diff_coef + 1, embedding_dim=5)
    fea_emb2 = nn.Embedding(diff_coef * 2 + 1, embedding_dim=5)

    # NOTE get 正确率; get 错误率,反向 ; 正确率-错误率
    skill_right = df[['user_id', 'skill_id', 'correct']].groupby(['skill_id'])['correct'].agg('mean')
    skill_wrong = 1 - skill_right
    skill_cognition = skill_right - skill_wrong

    # NOTE 将系数化为[0,diff_coef]
    skill_right_torch = torch.Tensor(skill_right.values * diff_coef).long()
    skill_cognition_torch = torch.Tensor(skill_cognition.values * diff_coef).long() + diff_coef  # NOTE 从[-100,100]->[0,200]

    # NOTE embdding
    skill_right_emb = fea_emb1(skill_right_torch)
    skill_cognition_torch = fea_emb2(skill_cognition_torch)

    skill_group_feature_emb = torch.cat([skill_right_emb, skill_cognition_torch], dim=1)  # NOTE 横向合并
    # # TODO 复制一个skill_right的df 然后将fea_emb插入 NOTE 好像没必要，只是需要一个x而已
    # skill_group_feature_emb_df = skill_right.iloc[:, :0].copy(deep=True)  # 创建空DF
    # skill_group_feature_emb_df.insert(loc=0, column="skill_group_feature_emb", value=skill_group_feature_emb)

    return skill_group_feature_emb  # NOTE 这玩意就是一个x


if __name__ == '__main__':

    if SGC!=3:
        print("SGC3 才包含知识概念群划分！！！！！！！！！")
        print("SGC3 才包含知识概念群划分！！！！！！！！！")
        print("SGC3 才包含知识概念群划分！！！！！！！！！")
        print("SGC3 才包含知识概念群划分！！！！！！！！！")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', torch.cuda.is_available(), torch.version.cuda)
    print('torch version:', torch.__version__)  # 查看torch当前版本号'

    print("channels:", channels)
    # all_c2c_bystuGroup = joblib.load(config.jobname_all_c2c_bystuGroup)
    skill_df = joblib.load(config.jobname_skill_df)
    # c2c = joblib.load('Data/assist2009/c2c.pkl.zip')
    print("load file:", config.jobname_all_c2c_bystuGroup)
    print("load file:", config.jobname_skill_df)
    # print("load file:", jobname_all_skill_diff)

    # load data
    # df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_assist2009()
    print("-------LOAD ", dataset_path)
    if args.dataset_path in ["Data/assist2009/", "Data/assist2009_test/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_assist2009()
        print("lodad")
    elif args.dataset_path in ["Data/assist2017/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_assist2017()
    elif args.dataset_path in ["Data/algebra2005/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_algebra2005()
    elif args.dataset_path in ["Data/ednet/"]:
        df = MYKT_StuGroup_and_diff_dataprocess.data_preprocess_ednet()
    else:
        print("big error!!!!!!!!!!!!!!!!!!!!!!, not the dataset")


    # NOTE 训练知识概念群c2c
    edge_index = get_skill_group_logic(df, skill_df, skill_group_threshold, train_skill_group_logic=True)
    x = get_skill_group_feature(df)  # NOTE x是特征矩阵，不应当是原本的C2C

    data = Data(edge_index=edge_index, x=x)  # 设置数据集

    # NOTE 训练GS的emb
    channels_dim = 10 # args.K
    print("GAEmodel channels-dim for skill_group:", channels_dim)
    GAEmodel = pyg_nn.GAE(Encoder(x.shape[1], channels_dim, K)).to(dev)  # 设为x的维度
    skill_emb, model_skill_emb, skill_group_byQ = train_skill_emb(model=GAEmodel, data=data, skill_df=skill_df)  # NOTE 在这里不需要传入self
    # skill_group_byCluster = cluster(skill_emb, model_skill_emb)  # NOTE skill_emb是个字典， model.encoder.feature.cpu()就不是

    # TODO
    #  得到了原本  <知识点> 与 <知识概念群>的映射关系，并用joblib保存
    #  {dataset_path}SGL_infering_method={}/skill_group_threshold={}_fea={v1}_skill_group.pkl.zip
    #  复制skill_df 然后插入
    # skill_group = pd.Series(skill_group_byCluster, index=skill_emb.keys())
    skill_group = pd.Series(skill_group_byQ, index=skill_emb.keys())
    joblib.dump(skill_group, config.jobname_skill_group)
    print("file save:", config.jobname_skill_group)
