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
 
 

import pickle, os, json
import numpy as np
import train
from argparse import ArgumentParser
from pprint import pprint
anomaly_type_num = 10
n_neighbors = 5



# data_explore()
# 0"cpu_saturation",
# 1"io_saturation",
# 2"database_backup",
# 3"table_restore",
# 4"poorly_physical_design",
# 5"poorly_written_query",
# 6"workload_spike",
# 7"flush_log",
# 8"vacuum_analyze",
# 9"lock_contention",


def kNN(alpha_vec, X_train, y_train, new_vec):
    res_distance = []
    # print(alpha_vec)
    for i in range(len(X_train)):
        idx = int(y_train[i])
        res = np.sqrt(np.dot((X_train[i] - new_vec)**2, alpha_vec[idx]))
        res_distance.append(res)
    idx_res = np.argsort(res_distance)
    # print(idx_res)
    int_y = y_train.astype(int)

    return np.argmax(np.bincount(int_y[idx_res[: n_neighbors]]))

def anomaly_metrics(alpha_vec, new_vec):
    feature_vec = alpha_vec * new_vec
    # threshold = 
    idx_list = np.argsort(feature_vec)[::-1]
    return idx_list[:5]


def build_description(root_cause_id):
    with open("./config/anomaly_type.json", "r") as f1, \
        open("./config/anomaly_info.json", "r") as f2: 
        anomaly_lookup    = json.load(f1)
        desc_lookup = json.load(f2)

    res = desc_lookup[anomaly_lookup[str(root_cause_id)]]
    pprint(res)

X_train_path   = "./model/X_train.npy"
y_train_path   = "./model/y_train.npy"
alpha_vec_path = "./model/anomaly_vec.npy"


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("--vec_path")
    args = parser.parse_args()

    X_train, y_train, alpha_vec = np.array([]), np.array([]), np.array([])
    if os.path.isfile(X_train_path)==False or os.path.isfile(y_train_path)==False:
        train.generate_X_y()

    if os.path.isfile(alpha_vec_path)==False:
        train.generate_anomaly_alpha()

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    alpha_vec = np.load(alpha_vec_path)

    new_vec = np.load(args.vec_path)
    root_cause_id = kNN(alpha_vec, X_train, y_train, new_vec)
    build_description(root_cause_id)
