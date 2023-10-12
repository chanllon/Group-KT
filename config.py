import math

sweep_config = {
    'metric': {'name': 'avg_early_mean_auc', 'goal': 'maximize'},
    'method': 'bayes'
}
# 参数范围      'method': 'bayes'
parameters_dict = {
    # 'learning_rate': {"values": [ 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
    'learning_rate': {"values": [0.001, 0.0001, 0.00001]},
    "dropout": {"values": [0.05, 0.2, 0.5]},
    'batch_size': {"values": [32, 64, 128, 256, 512]},
    "hidden_dim": {"values": [128, 256, 512]},
    "layer_dim": {"values": [1, 2]},
    "max_seq": {"values": [50,  100,  150, 200]}

}

# parameters_dict = {
#     # 'learning_rate': {"values": [ 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
#     'learning_rate': {"values": [ 0.00001]},
#     "dropout": {"values": [0.2]},
#     'batch_size': {"values": [64]},
#     "hidden_dim": {"values": [128]},
#     "layer_dim": {"values": [1]},
#     "max_seq": {"values": [150]}
#
# }

# parameters_dict_akt = {
#     # 'learning_rate': {"values": [ 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
#     'learning_rate': {"values": [0.001, 0.0001, 0.00001]},
#     "dropout": {"values": [0.05, 0.2, 0.5]},
#     "n_head": {"values": [1, 2, 4, 8]},
#     'batch_size': {"values": [32, 64, 128, 256, 512]},
#     "final_fc_dim": {"values": [128, 256, 512]},
#     "max_seq": {"values": [50,  100,  150, 200]}
#
# }

# NOTE arg参数
import argparse


def get_args():
    # NOTE 全局参数
    parser = argparse.ArgumentParser()
    # NOTE 可选方法：
    #  routine
    #  routineUndirected
    #  routineSGC[i] i = 1 2 3      routineSGC3    routineSGC5
    #  下面3个一定要统一
    parser.add_argument('--infering-method', type=str, default="routineSGC11", help='学生群体C2C学习的方法(Enum)')
    parser.add_argument('--SGC', type=int, default=11, help='对应MYKT_StuGroupClassfication.py中')
    # note 不再考虑KC_num 效果好就好 不好就算鸟
    parser.add_argument('--KC-num', type=int, default=1, help='学生分组使用的特征数')

    # TODO 后面的K设置为动态获取，为了不限制SGC5
    parser.add_argument('--K', type=int, default=3, help='学生分类的数量')
    # assist2009    assist2017
    # parser.add_argument('--dataset-path', type=str, default="Data/assist2009_test/", help='数据集/路径')
    # parser.add_argument('--dataset-path', type=str, default="Data/assist2009/", help='数据集/路径')
    # parser.add_argument('--dataset-path', type=str, default="Data/assist2012/", help='数据集/路径')
    # parser.add_argument('--dataset-path', type=str, default="Data/assist2017/", help='数据集/路径')
    parser.add_argument('--dataset-path', type=str, default="Data/algebra2005/", help='数据集/路径')
    # parser.add_argument('--dataset-path', type=str, default="Data/statics2011/", help='数据集/路径')
    # parser.add_argument('--dataset-path', type=str, default="Data/ednet/", help='数据集/路径')
    # parser.add_argument('--KC-num', type=int, default=1, help='学生分组使用的特征数')     # NOTE if dataset==statics2011 就用KC-NUM=1


    # parser.add_argument('--dataset-name', type=str, default="skill_builder_data", help='数据集名字')

    parser.add_argument('--all-stuGroup-file-name', type=str,
                        default="all_stuGroup_inDiffKC_K=4_NAN=0.5_methods=kmeanXmedian.pkl.zip", help='学生分组信息')


    parser.add_argument('--cluster-method', type=str, default="kmeanXmedian", help='学生分组使用的聚类方法')

    parser.add_argument('--NAN-FULL', type=float, default=0, help='空值设定')
    parser.add_argument('--skill-nclusters', type=int, default=3, help='知识概念群数')
    parser.add_argument('--skill-feature', type=str, default="v1", help='知识点特征长度')

    parser.add_argument('--EMB-method', type=str, default="GAE", help='使用的编码器是啥')
    # 11 59 123
    parser.add_argument('--channels', type=int, default=10, help='训练得到的特征维度')
    parser.add_argument('--skill-group-threshold', type=float, default=0.5, help='训练得到的特征维度')

    # NOTE SGC5 用是否过滤学
    parser.add_argument('--delet_school_min_inter_num', type=int, default=500, help='当SGC5时候')



    return parser.parse_args()


args = get_args()
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
skill_feature = args.skill_feature
SGC = args.SGC

# NOTE 定义JOBname ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# note
jobname_all_stuGroup_inDiffKC = "{}cluster_method={}/all_stuGroup_inDiff_KC_K={}_NAN={}_.pkl.zip".format(
    dataset_path, cluster_method, K, NAN_FULL)  # NOTE : 学生的分类情况dataframe
jobname_all_stuGroup_bySGC = "{}cluster_method={}/all_stuGroup_bySGC{}_K={}_NAN={}_.pkl.zip".format(
    dataset_path, cluster_method, SGC, K, NAN_FULL)  # NOTE : SGC1、SGC2 之外的
jobname_all_skill_prob = "{}cluster_method={}/all_skill_prob_K={}_.pkl.zip".format(
    dataset_path, cluster_method, K)  # TODO 貌似不需要在这里
jobname_skill_df = "{}skill_prob.pkl.zip".format(dataset_path)
jobname_schoolsID_index = "{}schoolsID_index.pkl.zip".format(dataset_path)  # NOTE schoolsID_与index的映射 其实就是个list list的索引就是group_id
jobname_all_c2c_emb_bystuGroup = \
    "{}cluster_method={}/infering_methods={}/all_c2c_emb_bystuGroup_EMB_method={}_.pkl.zip".format(
        dataset_path, cluster_method, infering_method, EMB_method)

# TODO 下面吧SGL的推断射到文件夹里了，保不准以后还有什么东西要设置
jobname_skill_group_logic = "{}cluster_method={}/infering_methods={}/skill_group_logic.pkl.zip".format(dataset_path, cluster_method, infering_method)
jobname_skill_group = "{}cluster_method={}/infering_methods={}/skill_group_threshold={}_sfea={}_skill_group.pkl.zip".format(dataset_path, cluster_method, infering_method, skill_group_threshold, skill_feature)

# NOTE 差异量化指标的存储
# 指标1 认知难度
jobname_all_skill_diff = "{}cluster_method={}/infering_methods={}/all_skill_diff_.pkl.zip".format(
    dataset_path, cluster_method, infering_method, K)  # NOTE 只和聚类方法、K、KCnum有关  ；虽然和KCnum有关，但是随着c2c生成的 无所谓了
# 指标2 认知C2C
jobname_all_c2c_bystuGroup = "{}cluster_method={}/infering_methods={}/all_c2c_bystuGroup_K={}_KCnum={}_.pkl.zip".format(
    dataset_path, cluster_method, infering_method, K, KC_num)  # NOTE 通过学生分类情况得到C2C三维数据
jobname_all_stuGroup_bySGC = "{}cluster_method={}/infering_methods={}/all_stuGroup_bySGC{}_K={}_NAN={}_.pkl.zip".format(
    dataset_path, cluster_method, infering_method, SGC, K, NAN_FULL)  # NOTE : SGC1、SGC2 之外的
# 指标3 反应时间
jobname_all_skill_group_time = "{}cluster_method={}/infering_methods={}/all_skill_group_time_.pkl.zip".format(
    dataset_path, cluster_method, infering_method, K)  #
# 指标4 首次反应状态 未emb
jobname_all_skill_group_first_reaction = "{}cluster_method={}/infering_methods={}/all_skill_group_first_reaction_.pkl.zip".format(
    dataset_path, cluster_method, infering_method, K)  #

# NOTE 训练使用
jobname_filtered_skill_prob = "{}filtered_skill_prob_.pkl.zip".format(dataset_path)
jobname_train_group_c = \
    "{}cluster_method={}/infering_methods={}/train_group_c_EMB_method={}_K={}_KCnum={}_.pkl.zip".format(
        dataset_path, cluster_method, infering_method, EMB_method, K, KC_num)
jobname_test_group_c = \
    "{}cluster_method={}/infering_methods={}/test_group_c_EMB_method={}_K={}_KCnum={}_.pkl.zip".format(
        dataset_path, cluster_method, infering_method, EMB_method, K, KC_num)
jobname_valid_group_c = \
    "{}cluster_method={}/infering_methods={}/valid_group_c_EMB_method={}_K={}_KCnum={}_.pkl.zip".format(
        dataset_path, cluster_method, infering_method, EMB_method, K, KC_num)

# NOTE 群体评估应用
jobname_GS = f"{dataset_path}cluster_method={cluster_method}/infering_methods={infering_method}/GS.pkl.zip"
jobname_GS05 = f"{dataset_path}cluster_method={cluster_method}/infering_methods={infering_method}/GS05.pkl.zip"
jobname_GS_CSV = f"{dataset_path}cluster_method={cluster_method}/infering_methods={infering_method}/GS.csv"
jobname_GS05_CSV = f"{dataset_path}cluster_method={cluster_method}/infering_methods={infering_method}/GS05.csv"