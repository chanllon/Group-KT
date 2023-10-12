import torch
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import time, joblib
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit

import argparse
import config

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

# # note 定义JOBname
# config.jobname_all_stuGroup_inDiffKC ="{}cluster_method={}/all_stuGroup_inDiff_KC_K={}_NAN={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, NAN_FULL) # NOTE : 学生的分类情况dataframe
# config.jobname_all_c2c_bystuGroup = "{}cluster_method={}/infering_methods={}/all_c2c_bystuGroup_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path,cluster_method, infering_method, K, KC_num) # NOTE 通过学生分类情况得到C2C三维数据
# config.jobname_all_skill_diff = "{}cluster_method={}/all_skill_diff_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, KC_num) # NOTE 只和聚类方法、K、KCnum有关
# config.jobname_all_skill_prob = "{}cluster_method={}/all_skill_prob_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, KC_num)    # TODO 貌似不需要在这里
# config.jobname_skill_df = "{}skill_prob.pkl.zip".format(dataset_path)
# config.jobname_all_c2c_emb_bystuGroup = \
#     "{}cluster_method={}/infering_methods={}/all_c2c_emb_bystuGroup_EMB_method={}_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path ,cluster_method, infering_method, EMB_method, K, KC_num)



# Training matrix in Autoencoder
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = pyg_nn.GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        """

        :param x: 结点特征矩阵
        :param edge_index: 领结矩阵
        :return:
        """
        x = (self.conv1(x, edge_index)).relu()      # note 原文是relu的？
        # x = (self.conv1(x, edge_index))   # note 原文是relu的？
        self.feature = self.conv2(x, edge_index)
        return self.feature  # 就是下面返回的Z？


# 考虑在外面再封装一层 变成多群体的知识点编码器
class GroupStateEncoder_NonShared():
    def __init__(self, K, all_c2c_bystuGroup, skill_df):
        """
        群体知识点状态编码网络
        :param K:
        :param C2C_feature_dimension: 设定编码出来的emb维度是多少
        :param all_c2c_bystuGroup:
        :param all_skill_diff:
        """
        # self.C2C_feature_dimension = C2C_feature_dimension  # TODO 这里的维度应当加上skill_diff,每个群体对题目难度的理解也是不一样的
        self.Encoder = []
        self.z = []
        self.K = K
        # self.all_skill_diff = all_skill_diff
        # self.all_skill_prob = all_skill_prob # TODO 暂时不需要这个变量
        self.skill_df = skill_df
        self.all_x = []
        self.all_edge_index = []
        self.all_edge_weight = []  # 但是没看在哪里有用到
        self.all_data = []
        self.all_c2c_emb_bystuGroup = []
        for k in range(self.K):
            cur_c2c_bystuGroup = all_c2c_bystuGroup[k]
            # x = torch.tensor(cur_c2c_bystuGroup.toarray().tolist(), dtype=torch.float)
            # NOTE x是结点特征矩阵[知识点数量，特征]
            # x = torch.tensor(c2c_add.toarray().tolist(), dtype=torch.float)

            # #TODO:待删除！！ 临时改命
            # temp = cur_c2c_bystuGroup._values()
            # temp[np.where(temp < 0)] = 0



            # 将scipy稀疏矩阵转换为边索引和边属性 edge_index：c2c的row和col edge_weight：c2c的data
            # edge_index, edge_weight = pyg_utils.convert.from_scipy_sparse_matrix(cur_c2c_bystuGroup) # NOTE cur_c2c_bystuGroup已经是sparse_coo格式了
            edge_index, edge_weight = cur_c2c_bystuGroup._indices(),cur_c2c_bystuGroup._values()
            edge_index = torch.nonzero(cur_c2c_bystuGroup.to_dense()).T
            # x = cur_c2c_bystuGroup.to_dense()[edge_index[0],edge_index[1]]
            x = cur_c2c_bystuGroup.to_dense()
            data = Data(edge_index=edge_index, x=x)  # 设置数据集 TODO 需不需要重写一个？

            self.all_x.append(x)  # TODO X应当还要和skill_df进行concat NOTE:暂时不考虑在这里concat
            self.all_edge_index.append(edge_index)
            self.all_edge_weight.append(edge_weight)
            self.all_data.append(data)
            self.Encoder.append(pyg_nn.GAE(Encoder(cur_c2c_bystuGroup.shape[1], channels)).to(dev))

    def _test(self, model, x, pos_edge_index, neg_edge_index, train_pos_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        # loss = model.recon_loss(z, train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

    def _train(self, epoc_trainh, x, model, optimizer, train_pos_edge_index):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)  # NOTE 在这里调用encoder
        loss = model.recon_loss(z, train_pos_edge_index)
        loss.backward()
        optimizer.step()
        # writer.add_scalar("loss", loss.item(), epoch)

    def _train_one_group(self, model, data, skill_df):
        labels = data.y  # what: 也没有说这个是要干嘛的
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data, val_ratio=0, test_ratio=0.1)
        x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)  # NOTE 选择你的训练导师 CPU/GPU

        # TODO 使用RandomLinkSplit提高精度？
        transform = RandomLinkSplit(is_undirected=False, num_val=0, add_negative_train_samples=True)
        # train_data, val_data, test_data = transform(data)
        # x = train_data.x.to(dev)  # NOTE 选择你的训练导师 CPU/GPU
        # train_pos_edge_index = train_data.edge_index.to(dev)    # NOTE 训练集中的边全都是正样本

        # TODO 划分测试集正负边


        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        for epoch in tqdm(range(1, 1001)):
            try:
                self._train(epoch, x, model, optimizer, train_pos_edge_index) # # NOTE 在这里不需要传入self
                # todo:test_pos_edge_index  pos边和neg边是啥玩意啊 哪来的正负训练啊 没有搞明白
                auc, ap = self._test(model, x, data.test_pos_edge_index, data.test_neg_edge_index, train_pos_edge_index) # NOTE 在这里不需要传入self
                if epoch % 100 == 0:
                    print('Epoch: {:03d} , AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
            except ValueError:
                if epoch % 1000 == 0:
                    print("{}ValueError:auc ap error".format(epoch))
                pass

            c2c_emb = {}    # NOTE 这是字典啊
            for i, emb in enumerate(model.encoder.feature.cpu()):
                c2c_emb[skill_df.index[i]] = emb
        return c2c_emb

    def do_encoder(self, epoch=2000):
        '''
        调用该函数完成基于GAE群体知识点状态编码工作 调用
        :param all_x:
        :param edge_index:
        :return:
        '''

        for k in range(K):
            cur_moder = self.Encoder[k]
            cur_data = self.all_data[k]
            # NOTE 临时改命，将-1 变成0测试待删除

            # cur_skill_prob = self.all_skill_prob[k] # todo 暂时不需要这个了
            cur_c2c_emb = self._train_one_group(model=cur_moder, data=cur_data, skill_df=self.skill_df) # NOTE 在这里不需要传入self
            self.all_c2c_emb_bystuGroup.append(cur_c2c_emb)
            # NOTE 非共享型参数的 表征横向concat
        print("stop")
        joblib.dump(self.all_c2c_emb_bystuGroup, config.jobname_all_c2c_emb_bystuGroup)
        print("file had save:", config.jobname_all_c2c_emb_bystuGroup)


if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA availability:', torch.cuda.is_available())
    print('torch version:', torch.__version__)  # 查看torch当前版本号'
    print("channels:",channels)

    skill_df = joblib.load(config.jobname_skill_df)
    print("load file:", config.jobname_skill_df)

    all_c2c_bystuGroup = joblib.load(config.jobname_all_c2c_bystuGroup)
    print(f"config.jobname_all_c2c_bystuGroup={config.jobname_all_c2c_bystuGroup}")

    # if args.SGC==1 or args.SGC==2:
    #     all_c2c_bystuGroup = joblib.load(config.jobname_all_c2c_bystuGroup)
    #     print("load file:", config.jobname_all_c2c_bystuGroup)
    # else:
    #     all_c2c_bystuGroup = joblib.load(config.jobname_all_stuGroup_bySGC)
    #     print("load file:", config.jobname_all_stuGroup_bySGC)
    K = len(all_c2c_bystuGroup)
    print(f"k:{K}")

    GSE = GroupStateEncoder_NonShared(K, all_c2c_bystuGroup, skill_df)
    GSE.do_encoder()
