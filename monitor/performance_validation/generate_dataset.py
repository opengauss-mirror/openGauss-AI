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
 
 


import numpy as np
import pandas as pd
import os
import json



from config  import parse_cmd_args
args = parse_cmd_args()


mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5}
oid = 0


def compute_cost(node):
    return float(node["Total Cost"]) - float(node["Startup Cost"])


def compute_time(node):
    return float(node["Actual Total Time"]) - float(node["Actual Startup Time"])


def extract_plan(sample):
    global mp_optype, oid
    # function: extract SQL feature
    # return: start_time, node feature, edge feature

    plan = sample["plan"]
    while isinstance(plan, list):
        plan = plan[0]
    # Features: print(plan.keys())
    # start time = plan["start_time"]
    # node feature = [Node Type, Total Cost:: Actual Total Time]
    # node label = [Actual Startup Time, Actual Total Time]

    plan = plan["Plan"]  # root node
    node_matrix = []
    edge_matrix = []

    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        parent["oid"] = oid
        oid = oid + 1

        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)

    cnt = 0
    maxnm = 0
    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        maxnm = maxnm + 1
        run_cost = compute_cost(parent)
        run_time = compute_time(parent)
        if parent["Node Type"] not in mp_optype:
            mp_optype[parent["Node Type"]] = len(mp_optype)

        node_feature = [parent["oid"], mp_optype[parent["Node Type"]], run_cost, float(parent["Actual Startup Time"]),
                        run_time]
        node_matrix = [node_feature] + node_matrix
        cnt = cnt + 1
        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)
                edge_matrix = [[node["oid"], parent["oid"], 1]] + edge_matrix
    # node: 18 * featuers
    # edge: 18 * 18

    return float(sample["start_time"]), node_matrix, edge_matrix


def generate_graph(wid, path):

    vmatrix = []
    ematrix = []

    with open(os.path.join(path,"sample-plan-" + str(wid) + ".txt"), "r") as f:
        for sample in f.readlines():
            sample = json.loads(sample)
            start_time, vertex, edge = extract_plan(sample)
            vmatrix = vmatrix + vertex
            ematrix = ematrix + edge
    return vmatrix, ematrix


base_dir = os.path.abspath(os.curdir)

data_path = os.path.join(base_dir, 'data/query_plan/job-pg')

for wid in range(args['workload_num']):
    vmatrix, ematrix = generate_graph(wid, data_path)
    with open(os.path.join(data_path,"test","sample-plan-"+str(wid)+".content"), "w") as wf:
            for v in vmatrix:
                wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
    with open(os.path.join(data_path,"test","sample-plan-"+str(wid)+".cites"), "w") as wf:
            for e in ematrix:
                wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")

print("Data Generated!!")