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

def parse_cmd_args():

    parser = argparse.ArgumentParser()

    # benchmark
    parser.add_argument('--workload', type=str, default='tpch', help='[tpch, user]')

    args = parser.parse_args()
    argus = vars(args)

    return argus
