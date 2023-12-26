# TODO : 0 设置全局参数列表目前 ①K ②数据集名

# TODO: 1 得到学生分类后的多种图、知识点难度 并且保存   NOTE √
# TODO: 1.1 可做改进：，学生知识点特征太多了，可以适当使用啥编码成更短的。或者不编码了，两个实验都可以做

# TODO：2 计算学生的群体知识点状态图,使用有向计算的。 并且保存
# TODO：2.1 可做改进：


# TODO： 没了

import numpy as np
import pandas as pd
import torch
import os
import joblib
from tqdm import notebook, tqdm
from scipy.sparse import coo_matrix
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import argparse
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import QuantificationOfGroupDiff

# from coo import co_acc_probs, co_acc_skills

# NOTE 全局参数

import config
args = config.get_args()
NAN_FULL = args.NAN_FULL
K = args.K
dataset_path = args.dataset_path
# dataset_name = args.dataset_name
all_stuGroup_file_name = args.all_stuGroup_file_name
KC_num = args.KC_num
infering_method = args.infering_method
cluster_method = args.cluster_method

# # note 定义JOBname
# # jobname_all_stuGroup_inDiffKC ="{}all_stuGroup_inDiffKC_K={}_NAN=0.5_cluster_method={}.pkl.zip".format(
# #     dataset_path, K, cluster_method) # NOTE : 学生的分类情况dataframe
# jobname_all_stuGroup_inDiffKC ="{}cluster_method={}/all_stuGroup_inDiff_KC_K={}_NAN={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, NAN_FULL) # NOTE : 学生的分类情况dataframe
# jobname_all_c2c_bystuGroup = "{}cluster_method={}/infering_methods={}/all_c2c_bystuGroup_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path,cluster_method, infering_method, K, KC_num) # NOTE 通过学生分类情况得到C2C三维数据
# jobname_all_skill_diff = "{}cluster_method={}/all_skill_diff_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, KC_num) # NOTE 只和聚类方法、K、KCnum有关
# jobname_all_skill_prob = "{}cluster_method={}/all_skill_prob_K={}_KCnum={}_.pkl.zip".format(
#     dataset_path, cluster_method, K, KC_num)
# jobname_skill_df = "{}skill_prob.pkl.zip".format(dataset_path)


def data_preprocess_assist2009(drop_dup=False):
    """
    assist2009 的数据预处理
    :return:
    """
    ## preprocess Data csv
    # preprocess assist2009
    read_col = ['order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id', 'correct',
                'sequence_id', 'base_sequence_id', 'skill_id', 'skill_name', 'original', 'school_id',
                'ms_first_response', 'attempt_count']
    target = 'correct'
    # read in the Data
    data_path = 'Data/assist2009_test/skill_builder_data.csv'
    print("data_path:", data_path)
    df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")[read_col]
    # delete empty skill_id
    df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct", "order_id"])
    df = df.dropna(subset=['skill_id'])
    df = df[~df['skill_id'].isin(['noskill'])]
    # NOTE 不存在school_id为空的
    # todo 由于数据太多了 目前为了学习 仅取1%玩一下
    # df = df.sample(frac=0.01, replace=False, axis=0, random_state=123)

    # NOTE 去掉大量重复的
    if drop_dup:
        df = df.drop_duplicates(subset=["user_id", "skill_id", "correct"])

    df['is_repeat'] = ['0']*len(df)

    # TODO 两个方法，第二个是删除多skill的，第一个是将他划分开来
    # TODO 方法1
    column_skill_id = df['skill_id'].str.split('_', expand=True).stack()
    column_skill_id_1 = column_skill_id.reset_index(level=1, drop=True, name='skill_id')
    column_skill_id_1.name = "skill_id"
    df = df.drop(['skill_id'], axis=1).join(column_skill_id_1)
    print("skill nums:", len(set(df.skill_id)))



    # 它将替换所有与条件不匹配的值
    # df["is_repeat"].where(df.duplicated(subset=["order_id"]) == False, "1", inplace=True)
    # df['is_repeat'] = ['0']*len(df)


    df.skill_id = df.skill_id.astype('int64')
    print('After removing empty skill_id,school_id, records number %d' % len(df))

    # TODO not delete scaffolding problems
    # df = df[df['original'].isin([1])]
    print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 3
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    df = df[['order_id','user_id', 'problem_id', 'skill_id', 'correct', 'school_id', 'ms_first_response','attempt_count','is_repeat','skill_name']]
    print('After deleting some users, records number %d' % len(df))

    if args.SGC == 5:
        delet_school_min_inter_num = args.delet_school_min_inter_num
        schools = df.groupby(['school_id'], as_index=True)
        delete_schools = []
        for ss in schools:
            if len(ss[1]) < delet_school_min_inter_num:
                delete_schools.append(ss[0])
        print('deleted delete_schools number based min-inters %d' % len(delete_schools))
        df = df[~df['school_id'].isin(delete_schools)]
        print('After deleting some school, records number %d' % len(df))

    # NOTE 做实验的时候用
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int( len(uid) * 0.8 )
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]



    return df


def data_preprocess_assist2017():
    """
        assist2017 的数据预处理
        :return:
        """
    ## preprocess Data csv
    # preprocess assist2009
    read_col = [['user_id','skill', 'problem_id', 'skill_id', 'correct', 'school_id', 'ms_first_response', 'startTime', 'attempt_count']]
    # read_col = ['order_id', 'assignmentId', 'studentId', 'assistment_id', 'problemId', 'correct',
    #             'skill', 'skill_name', 'original', 'schoolID',
    #             'startTime','endTime', 'attemptCount']
    target = 'correct'
    # read in the Data
    data_path = 'Data/assist2017/anonymized_full_release_competition_dataset_my_skillid.csv'
    print("data_path:", data_path)
    df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
    df = df.dropna(subset=["user_id", "problem_id", "correct", "skill", "startTime"])
    df["index"] = range(len(df))
    # df = df.sort_values(by=['startTime', 'index'])
    df.ms_first_response = df.ms_first_response.astype('int64')
    df = df[~df['skill'].isin(['noskill'])]


    df['is_repeat'] = ['0']*len(df)

    # 建立映射
    # delete empty skill_id
    df = df.dropna(subset=['skill_id'])
    # df = df[~df['skill_id'].isin(['noskill'])]

    # NOTE 不存在school_id为空的
    # todo 由于数据太多了 目前为了学习 仅取1%玩一下
    # df = df.sample(frac=0.01, replace=False, axis=0, random_state=123)

    # TODO 两个方法，第二个是删除多skill的，第一个是将他划分开来
    # TODO 方法1
    # column_skill_id = df['skill_id'].str.split('_', expand=True).stack()
    # column_skill_id_1 = column_skill_id.reset_index(level=1, drop=True, name='skill_id')
    # column_skill_id_1.name = "skill_id"
    # df = df.drop(['skill_id'], axis=1).join(column_skill_id_1)
    df.skill_id = df.skill_id.astype('int64')

    print('After removing empty skill_id,school_id, records number %d' % len(df))

    # NOTE delete scaffolding problems
    # df = df[df['original'].isin([1])]
    # print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 100
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'school_id', 'ms_first_response', 'attempt_count','is_repeat']]
    print('After deleting some users, records number %d' % len(df))

    if args.SGC == 5:
        delet_school_min_inter_num = args.delet_school_min_inter_num
        schools = df.groupby(['school_id'], as_index=True)
        delete_schools = []
        for ss in schools:
            if len(ss[1]) < delet_school_min_inter_num:
                delete_schools.append(ss[0])
        print('deleted delete_schools number based min-inters %d' % len(delete_schools))
        df = df[~df['school_id'].isin(delete_schools)]
        print('After deleting some school, records number %d' % len(df))

    # # NOTE 做实验的时候用
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int(len(uid) * 0.8)
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]

    return df


def data_preprocess_assist2012():
    """
        assist2012 的数据预处理
        :return:
        """
    ## preprocess Data csv
    # preprocess assist2012
    read_col = [["user_id", "skill_id", "start_time", "problem_id", "correct","ms_first_response", "attempt_count"]]
    # read_col = ['order_id', 'assignmentId', 'studentId', 'assistment_id', 'problemId', 'correct',
    #             'skill', 'skill_name', 'original', 'schoolID',
    #             'startTime','endTime', 'attemptCount']
    target = 'correct'
    # read in the Data
    data_path = 'Data/assist2012/assist12_my.csv'
    print("data_path:", data_path)
    df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
    df = df.dropna(subset=["user_id", "skill_id", "start_time", "problem_id", "correct","ms_first_response"])
    df["index"] = range(len(df))
    # df = df.sort_values(by=['startTime', 'index'])
    df.ms_first_response = df.ms_first_response.astype('int64')
    df = df.sort_values(['user_id', 'start_time'])


    df['is_repeat'] = ['0']*len(df)

    # 建立映射
    # delete empty skill_id
    df = df.dropna(subset=['skill_id'])
    df = df[~df['skill_id'].isin(['noskill'])]
    # NOTE 不存在school_id为空的
    # todo 由于数据太多了 目前为了学习 仅取1%玩一下
    # df = df.sample(frac=0.01, replace=False, axis=0, random_state=123)

    # TODO 两个方法，第二个是删除多skill的，第一个是将他划分开来
    # TODO 方法1
    # column_skill_id = df['skill_id'].str.split('_', expand=True).stack()
    # column_skill_id_1 = column_skill_id.reset_index(level=1, drop=True, name='skill_id')
    # column_skill_id_1.name = "skill_id"
    # df = df.drop(['skill_id'], axis=1).join(column_skill_id_1)
    df.skill_id = df.skill_id.astype('int64')

    print('After removing empty skill_id,school_id, records number %d' % len(df))

    # NOTE delete scaffolding problems
    # df = df[df['original'].isin([1])]
    # print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 10
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'school_id', 'ms_first_response', 'attempt_count','is_repeat']]
    print('After deleting some users, records number %d' % len(df))

    if args.SGC == 5:
        delet_school_min_inter_num = args.delet_school_min_inter_num
        schools = df.groupby(['school_id'], as_index=True)
        delete_schools = []
        for ss in schools:
            if len(ss[1]) < delet_school_min_inter_num:
                delete_schools.append(ss[0])
        print('deleted delete_schools number based min-inters %d' % len(delete_schools))
        df = df[~df['school_id'].isin(delete_schools)]
        print('After deleting some school, records number %d' % len(df))

    # # NOTE 做实验的时候用
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int(len(uid) * 0.8)
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]

    return df


def data_preprocess_algebra2005():
    """
        algebra2005 的数据预处理
        :return:
        """
    ## preprocess Data csv
    # preprocess algebra2005
    read_col = [["user_id","skill", "skill_id", "start_time", "problem_id", "correct","ms_first_response", "attempt_count", "is_repeat"]]
    # read in the Data
    data_path = 'Data/algebra2005/my_algebra2005_train_final.csv'
    print("data_path:", data_path)
    df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
    df = df.dropna(subset=["user_id", "skill_id", "start_time", "problem_id", "correct","ms_first_response"])
    df["index"] = range(len(df))
    # df = df.sort_values(by=['startTime', 'index'])
    df.ms_first_response = df.ms_first_response.astype('int64')
    df["skill_name"] = df["skill"]


    df['is_repeat'] = ['0']*len(df)

    # 建立映射
    # delete empty skill_id
    df = df.dropna(subset=['skill_id'])
    # df = df[~df['skill_id'].isin(['noskill'])]
    # NOTE 不存在school_id为空的
    # todo 由于数据太多了 目前为了学习 仅取1%玩一下
    # df = df.sample(frac=0.05, replace=False, axis=0, random_state=123)

    # TODO 两个方法，第二个是删除多skill的，第一个是将他划分开来
    # TODO 方法1
    # column_skill_id = df['skill_id'].str.split('_', expand=True).stack()
    # column_skill_id_1 = column_skill_id.reset_index(level=1, drop=True, name='skill_id')
    # column_skill_id_1.name = "skill_id"
    # df = df.drop(['skill_id'], axis=1).join(column_skill_id_1)
    df.skill_id = df.skill_id.astype('int64')

    print('After removing empty skill_id,school_id, records number %d' % len(df))

    # NOTE delete scaffolding problems
    # df = df[df['original'].isin([1])]
    # print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 3
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'ms_first_response', 'attempt_count','is_repeat','skill','skill_name']]
    print('After deleting some users, records number %d' % len(df))

    if args.SGC == 5:
        delet_school_min_inter_num = args.delet_school_min_inter_num
        schools = df.groupby(['school_id'], as_index=True)
        delete_schools = []
        for ss in schools:
            if len(ss[1]) < delet_school_min_inter_num:
                delete_schools.append(ss[0])
        print('deleted delete_schools number based min-inters %d' % len(delete_schools))
        df = df[~df['school_id'].isin(delete_schools)]
        print('After deleting some school, records number %d' % len(df))

    # # NOTE 做实验的时候用
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int(len(uid) * 0.8)
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]

    return df


def data_preprocess_algebra2006():
    """
        algebra2006 的数据预处理
        :return:
        """
    ## preprocess Data csv
    # preprocess algebra2006
    read_col = [["user_id", "skill_id", "problem_id", "correct","ms_first_response", "attempt_count"]]
    # read_col = ['order_id', 'assignmentId', 'studentId', 'assistment_id', 'problemId', 'correct',
    #             'skill', 'skill_name', 'original', 'schoolID',
    #             'startTime','endTime', 'attemptCount']
    target = 'correct'
    # read in the Data
    data_path = 'Data/algebra2006/my_algebra2006_train.csv'
    print("data_path:", data_path)
    df = pd.read_csv(data_path, low_memory=False, encoding="utf-8")
    # df = df.dropna(subset=["user_id", "skill_id", "start_time", "problem_id", "correct","ms_first_response"])
    df["index"] = range(len(df))
    # df = df.sort_values(by=['startTime', 'index'])
    df.ms_first_response = df.ms_first_response.astype('int64')


    # todo 由于数据太多了 目前为了学习 仅取1%玩一下
    # df = df.sample(frac=0.05, replace=False, axis=0, random_state=123)

    df.skill_id = df.skill_id.astype('int64')

    print('After removing empty skill_id,school_id, records number %d' % len(df))

    # NOTE delete scaffolding problems
    # df = df[df['original'].isin([1])]
    # print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 3
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'ms_first_response', 'attempt_count','is_repeat']]
    print('After deleting some users, records number %d' % len(df))

    # # NOTE 做实验的时候用
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int(len(uid) * 0.8)
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]

    return df


def data_preprocess_statics2011():
    """
        statics2011 的数据预处理
        :return:
        """
    ## preprocess Data csv
    # preprocess algebra2005
    read_col = [["user_id", "skill_id", "correct","ms_first_response", "attempt_count", "is_repeat"]]
    # read in the Data
    data_path = 'Data/statics2011/statics2011_Final.csv'
    print("data_path:", data_path)
    # df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
    df = pd.read_csv(data_path, low_memory=False)
    df["problem_id"] = "NA"
    df = df.dropna(subset=["user_id", "skill_id", "correct","ms_first_response"])
    df["index"] = range(len(df))
    # df = df.sort_values(by=['startTime', 'index'])
    df.ms_first_response = df.ms_first_response.astype('int64')


    # df['is_repeat'] = ['0']*len(df)

    # 建立映射
    # delete empty skill_id
    df = df.dropna(subset=['skill_id'])
    # df = df[~df['skill_id'].isin(['noskill'])]
    # NOTE 不存在school_id为空的
    # todo 由于数据太多了 目前为了学习 仅取1%玩一下
    # df = df.sample(frac=0.05, replace=False, axis=0, random_state=123)

    # TODO 两个方法，第二个是删除多skill的，第一个是将他划分开来
    # TODO 方法1
    # column_skill_id = df['skill_id'].str.split('_', expand=True).stack()
    # column_skill_id_1 = column_skill_id.reset_index(level=1, drop=True, name='skill_id')
    # column_skill_id_1.name = "skill_id"
    # df = df.drop(['skill_id'], axis=1).join(column_skill_id_1)
    df.skill_id = df.skill_id.astype('int64')

    print('After removing empty skill_id,school_id, records number %d' % len(df))

    # NOTE delete scaffolding problems
    # df = df[df['original'].isin([1])]
    # print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 3
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'ms_first_response', 'attempt_count','is_repeat']]
    print('After deleting some users, records number %d' % len(df))

    if args.SGC == 5:
        delet_school_min_inter_num = args.delet_school_min_inter_num
        schools = df.groupby(['school_id'], as_index=True)
        delete_schools = []
        for ss in schools:
            if len(ss[1]) < delet_school_min_inter_num:
                delete_schools.append(ss[0])
        print('deleted delete_schools number based min-inters %d' % len(delete_schools))
        df = df[~df['school_id'].isin(delete_schools)]
        print('After deleting some school, records number %d' % len(df))

    # # NOTE 做实验的时候用
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int(len(uid) * 0.8)
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]

    return df


def data_preprocess_ednet():
    """
        statics2011 的数据预处理
        :return:
        """
    ## preprocess Data csv
    # preprocess algebra2005
    read_col = [["user_id", "skill_id", "correct","ms_first_response", "attempt_count", "is_repeat"]]
    # read in the Data
    data_path = 'Data/ednet/ednet_sample.csv'
    data_path = 'Data/ednet/ednet_sample_process.csv'
    data_path = 'Data/ednet/ednet_Final.csv'

    print("data_path:", data_path)
    # df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
    df = pd.read_csv(data_path, low_memory=False)
    df["problem_id"] = "NA"
    df = df.dropna(subset=["user_id", "skill_id", "correct","ms_first_response"])
    df["index"] = range(len(df))
    # df = df.sort_values(by=['startTime', 'index'])
    df.ms_first_response = df.ms_first_response.astype('int64')


    # df['is_repeat'] = ['0']*len(df)

    # 建立映射
    # delete empty skill_id
    df = df.dropna(subset=['skill_id'])
    # df = df[~df['skill_id'].isin(['noskill'])]
    # NOTE 不存在school_id为空的
    # todo 由于数据太多了 目前为了学习 仅取1%玩一下
    # df = df.sample(frac=0.05, replace=False, axis=0, random_state=123)

    # TODO 两个方法，第二个是删除多skill的，第一个是将他划分开来
    # TODO 方法1
    # column_skill_id = df['skill_id'].str.split('_', expand=True).stack()
    # column_skill_id_1 = column_skill_id.reset_index(level=1, drop=True, name='skill_id')
    # column_skill_id_1.name = "skill_id"
    # df = df.drop(['skill_id'], axis=1).join(column_skill_id_1)
    df.skill_id = df.skill_id.astype('int64')

    print('After removing empty skill_id,school_id, records number %d' % len(df))

    # NOTE delete scaffolding problems
    # df = df[df['original'].isin([1])]
    # print('After removing scaffolding problems, records number %d' % len(df))

    min_inter_num = 3
    users = df.groupby(['user_id'], as_index=True)
    delete_users = []
    for u in users:
        if len(u[1]) < min_inter_num:
            delete_users.append(u[0])
    print('deleted user number based min-inters %d' % len(delete_users))
    df = df[~df['user_id'].isin(delete_users)]
    df = df[['user_id', 'problem_id', 'skill_id', 'correct', 'ms_first_response', 'attempt_count','is_repeat']]
    print('After deleting some users, records number %d' % len(df))

    if args.SGC == 5:
        delet_school_min_inter_num = args.delet_school_min_inter_num
        schools = df.groupby(['school_id'], as_index=True)
        delete_schools = []
        for ss in schools:
            if len(ss[1]) < delet_school_min_inter_num:
                delete_schools.append(ss[0])
        print('deleted delete_schools number based min-inters %d' % len(delete_schools))
        df = df[~df['school_id'].isin(delete_schools)]
        print('After deleting some school, records number %d' % len(df))

    # # NOTE 做实验的时候用
    # np.random.seed(2022)
    # uid = list(set(df["user_id"]))
    # np.random.shuffle(uid)
    # prop = int(len(uid) * 0.8)
    # uid_train = uid[:prop]
    # uid_test = uid[prop:]
    # train_df = df[df["user_id"].isin(uid_train)]
    # test_df = df[df["user_id"].isin(uid_test)]

    return df



def get_all_skill_prob(df):
    """
    由于出现纰漏，在这里补上所有群体的skill_prob, 需要在第二层编码阶段进行对齐不同群体的大小，作补0操作
    :param df: 全部DF
    :return:
    """
    all_stuGroup_inDiffKC = joblib.load(config.jobname_all_stuGroup_inDiffKC)
    print("loda file:", config.jobname_all_stuGroup_inDiffKC)
    print("KC_num:", KC_num)
    all_stuGroup = all_stuGroup_inDiffKC.iloc[:, KC_num]
    df['group'] = df['user_id'].apply(lambda r: all_stuGroup[r])
    all_skill_prob = []
    for k in range(K):

        cur_stuGroup_df = df[df["group"] == k]
        cur_skill_prob = cur_stuGroup_df[['skill_id', 'problem_id']].groupby(['skill_id'], as_index=True).apply(
            lambda r: np.array(list(set(r['problem_id'].values))))
        all_skill_prob.append(cur_skill_prob)
    joblib.dump(all_skill_prob, config.jobname_all_skill_prob)
    print("file save:", config.jobname_all_skill_prob)


def _get_skill_diff(cur_user_skill_martix):
    '''
    #不会被直接调用
    :param user_skill_martix:
    :return:
    '''
    # NOTE 方法1 计算平均值
    cur_skill_diff = cur_user_skill_martix[['user_id', 'skill_id', 'correct']].groupby(['skill_id'])['correct'].agg('mean')
    # job_name = "Data/assist2009/skill_diff.pkl.zip"
    # joblib.dump(skill_diff, job_name)
    # skill_diff
    # NOTE 方法2 还没想好
    return cur_skill_diff


def stu_cluster_assist09(df, NAN_full = 0):
    '''
    assist09参数：选定使用填充0.5 with epoch 全局定死初值聚类中心
    cluster_method = kmeanXmedian
    :param df:
    :param user_skill_martix: 根据df生成的每个学生对所有知识点掌握情况矩阵
    :return: NAN
    '''

    # NOTE 生成user_skill_martix
    skill_id = np.array(df['skill_id'])
    skills = set(skill_id)
    user_skill = df[['user_id', 'skill_id', 'correct']].groupby(['user_id', 'skill_id'])['correct'].agg('mean')
    user_id = np.array(df['user_id'])
    user = set(user_id)
    user_skill_martix = pd.DataFrame(index=list(user), columns=list(skills))
    for stu_kc in user_skill.index:
        user_skill_martix.loc[stu_kc[0]][stu_kc[1]] = user_skill[stu_kc]

    # NOTE 对知识点进行重新排序,并且新添维度 “rate”
    filter_KC_index = user_skill_martix.count().sort_values(ascending=False).index
    user_correct_rate = df[['user_id', 'skill_id', 'correct']].groupby(['user_id'])['correct'].agg('mean')
    user_correct_rate.sort_index
    USM_byfilter = user_skill_martix[filter_KC_index]
    USM_byfilter_rate = USM_byfilter.copy(deep=True)  # 深复制
    USM_byfilter_rate = USM_byfilter_rate.sort_index()
    # USM_byfilter_rate['rate'] = user_correct_rate.values
    USM_byfilter_rate.insert(loc=0, column='rate', value=user_correct_rate.values)

    # df['rate'] = USM_byfilter['problem_id'].apply(lambda r: user_correct_rate[r])
    # USM_byfilter
    # ①所有人的平均值
    # USM_byfilter_rate = USM_byfilter_rate.fillna(USM_byfilter_rate.mean())
    # ③ NOTE 填充
    USM_byfilter_rate = USM_byfilter_rate.fillna(NAN_full, inplace=False)

    print(USM_byfilter_rate.head())
    # # NOTE 设定聚类初始值---------------------------------------------------------------------
    # num_list = [1,2,3,4,5,8,10,15,20,25,30,35,40,45,50,55,60]
    num_list = range(1, USM_byfilter_rate.shape[1])  # 总共有USM_byfilter_rate。shape[1] 那么多个num_list
    i_index = 1  # 临时变量
    # K = 4  # 聚类参数，分几类
    epoch = 5  # 训练次数
    print("K=", K, " epoch=", epoch)

    step = 1 / K
    initCenter = pd.DataFrame()
    for s in range(K):
        # print("[",s*step,",",(s+1)*step+0.01,"]")
        tempUSM = USM_byfilter_rate[
            (USM_byfilter_rate['rate'] > s * step) & (USM_byfilter_rate['rate'] < ((s + 1) * step + 0.01))]
        # temp_init = tempUSM.sample(n = 1) # 在区间内随机选择初试聚类中心
        temp_init = tempUSM.iloc[int(tempUSM.shape[0] / 2) - 1:int(tempUSM.shape[0] / 2), :]  # tempUSM已经是有序的了，按“rate”排序
        initCenter = pd.concat([initCenter, temp_init], sort=False)
    print("聚类初始化中心点为:")
    print(initCenter.iloc[:, :5])
    # # NOTE 设定聚类初始值over---------------------------------------------------------------------

    # NOTE 开始训练---------------------------------------------------------------------
    all_stuGroup_inDiffKC = USM_byfilter_rate.iloc[:, :0].copy(deep=True)  # 创建空DF

    for numL in num_list:
        temp_stuGroup_inCurKC = USM_byfilter_rate.iloc[:, :0].copy(deep=True)  # 创建 一个
        for e in range(epoch):  # 控制变量，保证每个numL下的epoch次训练使用相同的聚类中心

            USM_values = USM_byfilter_rate.iloc[:, :numL].values
            # 当前epoch&numL下 模型构建训练，分类，结果
            # model = KMeans(n_clusters = K, init = all_init_center[e,:,:numL], n_init=1) # all_init_center是三维的
            model = KMeans(n_clusters=K, init=initCenter.iloc[:, :numL])    # n_init=1?
            # model.fit(USM_values)
            result = model.fit_predict(USM_values)  # fit_predict
            centers = model.cluster_centers_
            # 将result 插入
            # USM_byfilter_rate.insert(loc = 0, column = 'rate', value = user_correct_rate.values)
            temp_column = "epoch{}".format(e)
            temp_stuGroup_inCurKC.insert(loc=temp_stuGroup_inCurKC.shape[1], column=temp_column, value=result)

        # 将tempGroip 统计最多的那个数值，然后插入到all_stuGroup_inDiffKC中
        temp_column = "KC{}".format(numL)
        all_stuGroup_inDiffKC.insert(loc=all_stuGroup_inDiffKC.shape[1], column=temp_column,
                                     value=temp_stuGroup_inCurKC.T.max().values)

    print("所有学生在不同KC下的分类all_stuGroup_inDiffKC")
    print(all_stuGroup_inDiffKC)
    # note cluster_method = "kmeanXmedian" # 在全局参数里

    joblib.dump(all_stuGroup_inDiffKC, config.jobname_all_stuGroup_inDiffKC)
    print("file save:", config.jobname_all_stuGroup_inDiffKC)
    # NOTE 开始训练over---------------------------------------------------------------------
    # # 绘图一 ：绘制hist图 只绘制前20个图
    # plt.figure(figsize=(10, 8))
    # i_index = 1  # 临时变量
    # # for numL in num_list:
    # for numL in range(1, 20 + 1):
    #     plt.subplot(5, 4, i_index)
    #     i_index += 1
    #     result = all_stuGroup_inDiffKC.iloc[:, numL - 1].values
    #
    #     n, bins, patches = plt.hist(result, bins=np.arange(0, K + 1), rwidth=0.8,
    #                                 align='left')  # bins=np.arange(0.5,K+0.5)
    #     for i in range(len(n)):
    #         plt.text(bins[i], n[i] * 1.02, int(n[i]), horizontalalignment="center")  # 打标签，在合适的位置标注每个直方图上面样本数
    #     title = "num of KC:{}".format(numL)
    #     plt.title(title)
    # #     plt.xticks(np.arange(1, K+1))
    #
    # plt.tight_layout(pad=1.08)
    # plt.show()


def inferring_similarity(a, b, c, d):
    """
    :param a:
    :param b:
    :param c:
    :param d:
    :param _methods: inferring_methods的枚举类型
    :return:score of similarity
    """
    n = a + b + c + d
    # print("[n:", n, "]")
    # print("[n:", n, "a:", a, "b:", b, "c:", c, "d:", d, "]")
    result = 0
    if n == 0:
        return 0
    elif infering_method == "cohenKappa": # TODO: 这个公式有很大问题 出现0的概率太大了
    # else:
        if ((a+b)*(b+d)+(a+c)*(c+d)) == 0:
            return -1
        result = (2*(a*d-b*c)) / ((a+b)*(b+d)+(a+c)*(c+d))

        # Po = (a+d) / n
        # Pe = ((a+b)*(b+d)+(a+c)*(c+d)) / (n*n)
        # result = (Po - Pe) / (1 - Pe)
    elif infering_method == "adjusted_kappa":
        result = 0
    elif infering_method == "routine":
        result = a/(a+b+c+d)
    elif infering_method == "routineSGC1":
        result = a/(a+b+c+d)
    elif infering_method == "routineSGC2":
        result = a/(a+b+c+d)
    elif infering_method == "routineSGC3":
        result = a/(a+b+c+d)
    elif infering_method == "routineSGC4":
        result = a/(a+b+c+d)
    elif infering_method == "routineSGC5":
        result = a/(a+b+c+d)
    elif infering_method == "routineSGC11":
        result = a/(a+b+c+d)
    elif infering_method == "routineUndirected":
        result = a / (a + b + c + d)

    # print("result:", result)
    return result


def co_acc_skills_diredctional(data, temp):
    skill1 = temp[0]
    skill2 = temp[1]
    if skill1 == skill2:
        return 0.0 # NOTE skill是没有自己到自己的关系的
    # NOTE 测试----------------------------
    # if skill1 == 39 and skill2 == 101:
    #     return -39101
    # if skill1 == 101 and skill2 == 39:
    #     return -10139
    # NOTE 测试----------------------------

    # NOTE 初始化列联表数据
    # print("[", i, "/", "?", "]")
    # print("[s1:", skill1, " &s2:", skill2, "]")
    a = b = c = d = n = 0

    # NOTE 1 列出所有用户ID
    user_id = data.groupby(["user_id"])

    # NOTE 2 遍历用户是否同时有s1和s2的答题记录
    # for id in tqdm(user_id.indices):
    for id in user_id.indices:
        user_data = data[(data["user_id"] == id)]  # 得到当前用户的所有记录
        user_s1 = user_data[user_data["skill_id"].isin([skill1])]   # NOTE 该用户skill1的所有df
        user_s2 = user_data[user_data["skill_id"].isin([skill2])]
        # NOTE 3 遍历所有用户下的列联表数据 TODO: 可能会重复诶，每条s2记录都会重复多次，看看要不要优化
        if (user_s1.shape[0] > 0) & (user_s2.shape[0] > 0):
            # 则统计
            for s1_index, s1 in user_s1.iterrows():
                if s1["correct"] == 1:
                    # NOTE 根据有向的要求和推理证明需要：在答题序列上需要s1在S2之前出现
                    a += user_s2[(user_s2.index > s1_index) & (user_s2["correct"] == 1)].shape[0]
                    b += user_s2[(user_s2.index > s1_index) & (user_s2["correct"] == 0)].shape[0]
                if s1["correct"] == 0:
                    c += user_s2[(user_s2.index > s1_index) & (user_s2["correct"] == 1)].shape[0]
                    d += user_s2[(user_s2.index > s1_index) & (user_s2["correct"] == 0)].shape[0]

    return inferring_similarity(a, b, c, d)

    # (Data["user_id"] == 14) & (Data["skill_id"] == 13) # 再用data包含他 得到的就是我要的数据
    # Data[(Data["user_id"] == 14) & Data["skill_id"].isin([skill1,skill2])]
    # for i in range(0,len(user_s1)): print(user_s1.iloc[i])
    # user_s1.iloc[1].name # name前面是个series,name就是索引了
    # user_s2[ (user_s2.index>34377) & (user_s2["correct"] == 1 ) ].shape[0]


def co_acc_skills_undirected(data, temp):
    skill1 = temp[0]
    skill2 = temp[1]
    # NOTE 测试----------------------------
    # if skill1 == 39 and skill2 == 101:
    #     return -39101
    # if skill1 == 101 and skill2 == 39:
    #     return -10139
    # NOTE 测试----------------------------

    # NOTE 初始化列联表数据
    # print("[", i, "/", "?", "]")
    # print("[s1:", skill1, " &s2:", skill2, "]")
    a = b = c = d = n = 0

    # NOTE 1 列出所有用户ID
    user_id = data.groupby(["user_id"])

    # NOTE 2 遍历用户是否同时有s1和s2的答题记录
    # for id in tqdm(user_id.indices):
    for id in user_id.indices:
        user_data = data[(data["user_id"] == id)]  # 得到当前用户的所有记录
        user_s1 = user_data[user_data["skill_id"].isin([skill1])]
        user_s2 = user_data[user_data["skill_id"].isin([skill2])]
        # NOTE 3 遍历所有用户下的列联表数据 TODO: 可能会重复诶，每条s2记录都会重复多次，看看要不要优化
        if (user_s1.shape[0] > 0) & (user_s2.shape[0] > 0):
            # 则统计
            for s1_index, s1 in user_s1.iterrows():
                if s1["correct"] == 1:
                    # NOTE 无向
                    a += user_s2[(user_s2["correct"] == 1)].shape[0]
                    b += user_s2[(user_s2["correct"] == 0)].shape[0]
                if s1["correct"] == 0:
                    c += user_s2[(user_s2["correct"] == 1)].shape[0]
                    d += user_s2[(user_s2["correct"] == 0)].shape[0]

    return inferring_similarity(a, b, c, d)

    # (Data["user_id"] == 14) & (Data["skill_id"] == 13) # 再用data包含他 得到的就是我要的数据
    # Data[(Data["user_id"] == 14) & Data["skill_id"].isin([skill1,skill2])]
    # for i in range(0,len(user_s1)): print(user_s1.iloc[i])
    # user_s1.iloc[1].name # name前面是个series,name就是索引了
    # user_s2[ (user_s2.index>34377) & (user_s2["correct"] == 1 ) ].shape[0]


def extract_c2c_matrix_diredctional(df, skill_df):
    """
    :param df: user_skill_temp 需要df里的['user_id', 'skill_id', 'correct']
    :param method:  inferring_methods 枚举类型 设置成全局变量吧
    :return: void, 保存文件: joblib.dump(mat_skill, 'c2c'+method.name+'.pkl.zip')
    """
    cores = multiprocessing.cpu_count()
    print("cores:", cores)
    pool = multiprocessing.Pool(processes=cores)

    user_skill_temp = df[['user_id', 'skill_id', 'correct']].sort_values(by="user_id", axis=0, ascending=True)

    row = []
    col = []
    val = []
    print('Extracting c2c matrix...')
    # arg_skills = ((skill1, skill2) for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:])
    # arg_skills = ((skill1, skill2) for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:])
    arg_skills = [[skill1, skill2, i] for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:]]
    c2c = partial(co_acc_skills_diredctional, user_skill_temp)
    # val = pool.starmap(c2c, arg_skills)
    # tqdm(pool.starmap(c2c, arg_skills))
    # val = process_map(c2c, arg_skills)
    with multiprocessing.Pool(processes=cores) as p:
        val = list(tqdm(p.imap(c2c, arg_skills), total=len(arg_skills)))
    for i in range(len(skill_df.index)):
        for j in range(i, len(skill_df.index)):
            row.append(i)
            col.append(j)

    # NOTE 更改使用torch的coo_matrix
    # mat_skill = coo_matrix((val, (row, col)), shape=(len(skill_df.index), len(skill_df.index)))
    v = torch.tensor(val)
    i = torch.tensor(np.vstack((row, col)))
    shape = (len(skill_df.index), len(skill_df.index))
    mat_skill = torch.sparse_coo_tensor(i, v, shape)

    # NOTE 复制粘贴流：复制粘贴流：复制粘贴流：复制粘贴流：复制粘贴流：复制粘贴流：
    row2 = []
    col2 = []
    arg_skills2 = [[skill2, skill1, i] for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:]]
    c2c2 = partial(co_acc_skills_diredctional, user_skill_temp)
    with multiprocessing.Pool(processes=cores) as p2:
        val2 = list(tqdm(p2.imap(c2c2, arg_skills2), total=len(arg_skills2)))

    for i in range(len(skill_df.index)):
        for j in range(i, len(skill_df.index)):
            row2.append(j)
            col2.append(i)
    # NOTE 更改使用torch的coo_matrix
    # mat_skill = coo_matrix((val, (row, col)), shape=(len(skill_df.index), len(skill_df.index)))
    v = torch.tensor(val2)
    i = torch.tensor(np.vstack((row2, col2)))
    shape = (len(skill_df.index), len(skill_df.index))
    mat_skill2 = torch.sparse_coo_tensor(i, v, shape)
    # NOTE TODO 需要设置为0 对角线
    mat_add = mat_skill + mat_skill2



    # NOTE 保存
    # joblib.dump(mat_add, 'c2c_' + args.infering_method + '.pkl.zip')
    # joblib.dump(mat_add, 'c2c_' + "cohens_kappa" + '.pkl.zip')
    print("extract_c2c_DONE!")
    return mat_add


def extract_c2c_matrix_undirected(df, skill_df):
    """
    :param df: user_skill_temp 需要df里的['user_id', 'skill_id', 'correct']
    :param method:  inferring_methods 枚举类型 设置成全局变量吧
    :return: void, 保存文件: joblib.dump(mat_skill, 'c2c'+method.name+'.pkl.zip')
    """
    cores = multiprocessing.cpu_count()
    print("cores:", cores)
    pool = multiprocessing.Pool(processes=cores)

    user_skill_temp = df[['user_id', 'skill_id', 'correct']].sort_values(by="user_id", axis=0, ascending=True)

    row = []
    col = []
    val = []
    print('Extracting c2c matrix...')
    # arg_skills = ((skill1, skill2) for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:])
    # arg_skills = ((skill1, skill2) for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:])
    arg_skills = [[skill1, skill2, i] for (i, skill1) in enumerate(skill_df.index) for skill2 in skill_df.index[i:]]
    c2c = partial(co_acc_skills_undirected, user_skill_temp)
    # val = pool.starmap(c2c, arg_skills)
    # tqdm(pool.starmap(c2c, arg_skills))
    # val = process_map(c2c, arg_skills)
    with multiprocessing.Pool(processes=cores) as p:
        val = list(tqdm(p.imap(c2c, arg_skills), total=len(arg_skills)))
    for i in range(len(skill_df.index)):
        for j in range(i, len(skill_df.index)):
            row.append(i)
            col.append(j)

    # NOTE 更改使用torch的coo_matrix
    # mat_skill = coo_matrix((val, (row, col)), shape=(len(skill_df.index), len(skill_df.index)))
    v = torch.tensor(val)
    i = torch.tensor(np.vstack((row, col)))
    shape = (len(skill_df.index), len(skill_df.index))
    mat_skill = torch.sparse_coo_tensor(i, v, shape)

    # NOTE 无向直接对称反正相加
    mat_skill_t = mat_skill.t()
    # mat_skill_t.setdiag(0)
    mat_add = mat_skill + mat_skill_t #TODO 对角矩阵没有搞定

    print("extract_c2c_DONE!")
    return mat_add


def extract_c2c_byStuGroup(df):  # TODO: 有待测试
    # 命名规则  Data/{数据集名}/数据集路径/c2c_计算方法_K=?_KCnum=?_.pkl.zip
    # jobname = "{}all_stuGroup_inDiffKC_K={}_NAN=0.5_cluster_method_methods={}.pkl.zip".format(dataset_path, K, cluster_method)
    all_stuGroup_inDiffKC = joblib.load(config.jobname_all_stuGroup_inDiffKC)
    print("loda file:", config.jobname_all_stuGroup_inDiffKC)
    print("KC_num:", KC_num)
    all_stuGroup = all_stuGroup_inDiffKC.iloc[:, KC_num]
    df['group'] = df['user_id'].apply(lambda r: all_stuGroup[r])

    all_c2c_bystuGroup = []
    all_skill_diff = []

    # NOTE 这里的skill不用all_skill_prob是因为之前遗漏了，但是这样写也没问题，因为这个skill_df就是从对应group里的df得到的
    # NOTE 2 反正重写了，不如直接把skill_df改成总的
    skill_df = df[['skill_id', 'problem_id']].groupby(['skill_id'], as_index=True).apply(
        lambda r: np.array(list(set(r['problem_id'].values))))
    joblib.dump(skill_df, config.jobname_skill_df)
    print("flie save:",config.jobname_skill_df)

    for k in range(K):
        print("NOTE :当前正在训练第{}类群体知识点状态图".format(k))
        cur_stuGroup_df = df[df["group"] == k]
        # NOTE 插入到DF中
        cur_c2c = extract_c2c_matrix_diredctional(cur_stuGroup_df,skill_df)  # 传入K是多少
        all_c2c_bystuGroup.append(cur_c2c)

        # NOTE 同时计算当前群体K的skill_diff, 在当前 KCnum值、K值、cluster_method 下得到的所有群体的难度
        cur_skill_diff = _get_skill_diff(cur_stuGroup_df)
        all_skill_diff.append(cur_skill_diff)
    # 保存
    # jobname = "{}{}all_c2c_bystuGroup_infering_methods={}_K={}_KCnum={}_.pkl.zip".format(
    #     dataset_path, dataset_name, infering_method, K, KC_num)
    joblib.dump(all_c2c_bystuGroup, config.jobname_all_c2c_bystuGroup)
    print("file save:", config.jobname_all_c2c_bystuGroup)
    # jobname = "{}{}all_skill_diff_cluster_method={}_K={}_KCnum={}_.pkl.zip".format(
    #     dataset_path, dataset_name, cluster_method, K, KC_num)
    joblib.dump(all_skill_diff, config.jobname_all_skill_diff)
    print("file save:", config.jobname_all_skill_diff)

    print("extract_c2cByStuGroup_DONE!")


def extract_c2c_byStuGroup_undirected(df):
    # 命名规则  Data/{数据集名}/数据集路径/c2c_计算方法_K=?_KCnum=?_.pkl.zip
    # jobname = "{}all_stuGroup_inDiffKC_K={}_NAN=0.5_cluster_method_methods={}.pkl.zip".format(dataset_path, K, cluster_method)
    K = args.K
    if args.SGC == 1 or args.SGC == 2:   # TODO: 记得改
        all_stuGroup_inDiffKC = joblib.load(config.jobname_all_stuGroup_inDiffKC)
        print("loda file:", config.jobname_all_stuGroup_inDiffKC)
        print("KC_num:", KC_num)
        all_stuGroup = all_stuGroup_inDiffKC.iloc[:, KC_num]
        df['group'] = df['user_id'].apply(lambda r: all_stuGroup[r])
    elif args.SGC == 3:
        # TODO 等待测试
        print("未经过测试")
        all_stuGroup = joblib.load(config.jobname_all_stuGroup_bySGC)
        df['group'] = df['user_id'].apply(lambda r: all_stuGroup.loc[r])
    elif args.SGC == 5:
        # NOTE school_id是学校代码，需要转化为
        # TODO K=school_id数
        school_id = np.array(df['school_id'])
        schools_map = {}
        schools = set(school_id)
        schools_map = {}
        ii = 0
        for s in schools:
            schools_map[s] = ii
            ii = ii + 1
        K = len(schools_map)  # 改编
        df['group'] = df['school_id'].apply(lambda r: schools_map[r])      # NOTE 根据school_id添加group
        joblib.dump(schools_map, config.jobname_schoolsID_index)
        print("file save:", config.jobname_schoolsID_index)
    elif args.SGC == 11:
        all_stuGroup = joblib.load(config.jobname_all_stuGroup_bySGC)
        df['group'] = df['user_id'].apply(lambda r: all_stuGroup.loc[r])
    else:
        print("BIG ERROR: no SGC")
        print("BIG ERROR: no SGC")
        print("BIG ERROR: no SGC")
        print("BIG ERROR: no SGC")
        print("BIG ERROR: no SGC")
        print("BIG ERROR: no SGC")

    all_c2c_bystuGroup = []
    all_skill_diff = []
    all_skill_group_time = []
    all_skill_group_fitst_reacttion = []
    # NOTE 这里的skill不用all_skill_prob是因为之前遗漏了，但是这样写也没问题，因为这个skill_df就是从对应group里的df得到的
    # NOTE 2 反正重写了，不如直接把skill_df改成总的
    skill_df = df[['skill_id', 'problem_id']].groupby(['skill_id'], as_index=True).apply(
        lambda r: np.array(list(set(r['problem_id'].values))))
    joblib.dump(skill_df, config.jobname_skill_df)
    print("flie save:", config.jobname_skill_df)
    print("K=", K)
    for k in range(K):
        print("NOTE :当前正在训练第{}类群体知识点状态图".format(k))
        cur_stuGroup_df = df[df["group"] == k]
        # NOTE 插入到DF中
        # cur_c2c = extract_c2c_matrix_diredctional(cur_stuGroup_df, skill_df)  # NOTE 邮箱
        cur_c2c =extract_c2c_matrix_undirected(cur_stuGroup_df, skill_df)   # NOTE 无向
        all_c2c_bystuGroup.append(cur_c2c)

        # NOTE 同时计算当前群体K的skill_diff, 在当前 KCnum值、K值、cluster_method 下得到的所有群体的难度
        #  群体认知难度
        cur_skill_diff = _get_skill_diff(cur_stuGroup_df)
        all_skill_diff.append(cur_skill_diff)
        # note 群体反应时间
        cur_skill_group_time = QuantificationOfGroupDiff._get_skill_group_time_onegroup(cur_stuGroup_df)
        all_skill_group_time.append(cur_skill_group_time)
        # note 群体首次反应情况
        cur_skill_group_fitst_reacttion = QuantificationOfGroupDiff._get_skill_group_fitst_reacttion_onegroup(cur_stuGroup_df)
        all_skill_group_fitst_reacttion.append(cur_skill_group_fitst_reacttion)
        # note 群体首次IRT  # TODO 先不完成吧 过于麻烦了吧


    # NOTE 保存
    joblib.dump(all_c2c_bystuGroup, config.jobname_all_c2c_bystuGroup)
    print("file save:", config.jobname_all_c2c_bystuGroup)

    joblib.dump(all_skill_diff, config.jobname_all_skill_diff)
    print("file save:", config.jobname_all_skill_diff)

    joblib.dump(all_skill_group_time, config.jobname_all_skill_group_time)
    print("file save:", config.jobname_all_skill_group_time)

    joblib.dump(all_skill_group_fitst_reacttion, config.jobname_all_skill_group_first_reaction)
    print("file save:", config.jobname_all_skill_group_first_reaction)

    print("extract_c2cByStuGroup_DONE!")


def temp_get_skill_diff():
    all_stuGroup_inDiffKC = joblib.load(config.jobname_all_stuGroup_inDiffKC)
    print("loda file:", config.jobname_all_stuGroup_inDiffKC)
    print("KC_num:", KC_num)
    all_stuGroup = all_stuGroup_inDiffKC.iloc[:, KC_num]
    df['group'] = df['user_id'].apply(lambda r: all_stuGroup[r])
    all_skill_diff = []
    for k in range(K):
        print("NOTE :当前正在训练第{}类群体知识点状态图".format(k))
        cur_stuGroup_df = df[df["group"] == k]
        # NOTE 同时计算当前群体K的skill_diff, 在当前 KCnum值、K值、cluster_method 下得到的所有群体的难度
        cur_skill_diff = _get_skill_diff(cur_stuGroup_df)
        all_skill_diff.append(cur_skill_diff)
    # 保存
    joblib.dump(all_skill_diff, config.jobname_all_skill_diff)
    print("file save:", config.jobname_all_skill_diff)


if __name__ == '__main__':
    # multiprocessing to speedup matrix extraction

    # cores = multiprocessing.cpu_count()
    # print("cores:", cores)
    # pool = multiprocessing.Pool(processes=cores)

    # NOTE 处理数据集
    # df = data_preprocess_assist2009()
    # df = data_preprocess_assist2012()
    # df = data_preprocess_assist2017()
    # df = data_preprocess_algebra2005()
    if args.dataset_path in ["Data/assist2009/", "Data/assist2009_test/"]:
        df = data_preprocess_assist2009()
        print("lodad")
    elif args.dataset_path in ["Data/assist2017/"]:
        df = data_preprocess_assist2017()
    elif args.dataset_path in ["Data/assist2012/"]:
        df = data_preprocess_assist2012()
    elif args.dataset_path in ["Data/algebra2005/"]:
        df = data_preprocess_algebra2005()
    elif args.dataset_path in ["Data/statics2011/"]:
        df = data_preprocess_statics2011()
    elif args.dataset_path in ["Data/ednet/"]:
        df = data_preprocess_ednet()
    else:
        print("big error!!!!!!!!!!!!!!!!!!!!!!, not the dataset")
        print("big error!!!!!!!!!!!!!!!!!!!!!!, not the dataset")
        print("big error!!!!!!!!!!!!!!!!!!!!!!, not the dataset")

    # NOTE 学生聚类分类
    # stu_cluster_assist09(df, NAN_FULL) # OLD METHOD UNUSEFUL

    # NOTE 计算群体知识点图  TODO 根据学生分类完成知识点结构图训练 先用jupyter   E2E 想想怎么编写代码
    # extract_c2c_byStuGroup(df)        # NOTE  计算有向图
    extract_c2c_byStuGroup_undirected(df)   # NOTE 计算无向图

    # NOTE 得到群体答知识点难度
    # temp_get_skill_diff()

    # NOTE 解决遗漏，群体skill_prob列表 每个skill对应哪些prob TODO 删除吧 没用了
    # get_all_skill_prob(df)

    # NOTE 不知道要执行多久 autoDL自动关机
    # os.system("shutdown")

