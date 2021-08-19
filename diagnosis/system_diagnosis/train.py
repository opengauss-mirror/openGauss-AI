/*
 * Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 *
 * openGauss is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
 
 

import pickle, json
import numpy as np


anomaly_type_num = 10
num_cases = 25
n_neighbors = 5

def generate_X_y(data_path="./data/new_dataset_numerical.pickle"):
    with open(data_path, 'rb') as f_train:
        dataset = np.array(pickle.load(f_train))
        X = dataset[:,0:len(dataset[0]) - 1]
        y = dataset[:,len(dataset[0]) - 1]
        np.save("./model/X_train.npy", X)
        np.save("./model/y_train.npy", y)


'''生成重要性权重

Args:

data_file: 需要读取的数据文件路径
shuffle_idx: 训练集的下标
verbose: 是否可视化结果
treshold: 异常向量预处理阈值

Returns:

anomaly_alpha: 异常向量权重组成的矩阵，维度是field_num * anomaly_case

Raise:

None

'''
def generate_anomaly_alpha(num_cases = 25, data_file="./data/new_dataset_bool.pickle", threshold = 0.1, verbose=False):
    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)                                            # 从pickle文件中加载数据
        labels = dataset[:,len(dataset[0]) - 1]
        label_list = sorted(set(labels))
        label_map = dict(zip(label_list, list(range(len(label_list)))))     # 由于标签可能存在缺失，这一步是构造映射填补空隙
        inv_label_map = dict(zip(list(range(len(label_list))), label_list))
        dataset = dataset[:, :len(dataset[0]) - 1]                          # 去掉dataset中最后一行标签

        anomaly_field_vec = [[] for _ in range(anomaly_type_num)]           # 异常向量初始化构建
        shuffle_idx = range(len(dataset))
        for id in shuffle_idx:
            anomaly_field_vec[label_map[labels[id]]].append(dataset[id])    #

        local_ratio = np.zeros((0, len(dataset[0])))
        for vecs in anomaly_field_vec:
            res = np.sum(vecs, axis=0) / len(vecs)
            res[res<threshold] = 0.0
            local_ratio = np.row_stack((local_ratio, res))                  # 每一类聚合成local vector，然后计算距离

        dist_mat = np.ones((anomaly_type_num, anomaly_type_num))

        for id1, v1 in enumerate(local_ratio):
            for id2, v2 in enumerate(local_ratio):
                if id1 != id2:
                    dist = np.linalg.norm(v1 - v2)**2
                    dist_mat[id1,id2] = dist

                if verbose==True:
                    if id1==id2:
                        print('{:2f}'.format(0.0), end=' ')
                        continue
                    print('{:2f}'.format(dist), end=' ')
            if verbose==True:
                print()

        dist_mat = 1/dist_mat
        # for i in range(anomaly_type_num):
        #     dist_mat[i][i] = 0

        global_ratio = []
        for i in range(anomaly_type_num):
            tmp_ratio = 1 / np.sum(dist_mat[i]) * (sum(map(lambda j: dist_mat[i][j]*local_ratio[j], range(anomaly_type_num))))
            global_ratio.append(tmp_ratio)

        anomaly_alpha = np.abs((global_ratio - local_ratio))

        # alpha normalization
        alpha_sum = np.sum(anomaly_alpha, axis=1)
        anomaly_alpha = len(dataset[0]) * anomaly_alpha / alpha_sum[:,None]
        # anomaly_alpha = anomaly_alpha / alpha_sum[:,None]
        np.save("./model/anomaly_vec.npy", anomaly_alpha)
        return anomaly_alpha, label_map, inv_label_map

if __name__=="__main__":
    generate_X_y()
    generate_anomaly_alpha()