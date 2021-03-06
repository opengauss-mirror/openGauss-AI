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
 
 

# -*- coding: utf-8 -*-
"""
desciption: system variables or other constant information
"""
import os
import requests
import argparse

CONFIG = {
    'url': 'jdbc:postgresql://166.111.121.62:5432/',
    'host': '166.111.121.62',
    'port': 5432,
    'driver': 'org.postgresql.Driver',
    'username': 'postgres',
    'password': 'postgres',
    'schema': 'tpch1x',
    'sqldir': 'tpch',
    'logdir': 'rewrite_results'
}


class model_parameters():
    def __init__(self):
        self.cuda = False
        self.fastmode = False
        self.seed = 42
        self.epochs = 100
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5


def parse_cmd_args():

    parser = argparse.ArgumentParser()


    # benchmark
    parser.add_argument('--iteration_num', type=int, default=230, help='')
    parser.add_argument('--workload_num', type=int, default=3184, help='The number of queries')


    parser.add_argument('--feature_num', type=int, default=2, help='The number of vertex features')
    parser.add_argument('--node_dim', type=int, default=30, help='The size of intermediate network layers')

    args = parser.parse_args()
    argus = vars(args)

    return argus
