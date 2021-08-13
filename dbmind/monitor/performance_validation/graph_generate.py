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
 
 


import json
from configs.base import base_dir, get_file, table_job

'''
step-1
'''

dataset = 'job-pg'
data_path = base_dir.joinpath('data/query_plan/'+dataset)
graph_path = base_dir.joinpath('data/graph/'+dataset)

table_buf = table_job
weight_type = {'passing':1, 'sharing':1, 'conflict':-1, 'resource':0.1}

workload_size = 10
workload_num = 3184

edge_dim = 30
node_dim = 30

# generate graph data
mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5}
oid = 0


def compute_cost(node):
    return float(node["Total Cost"]) - float(node["Startup Cost"]) 

def compute_time(node):
    return float(node["Actual Total Time"]) - float(node["Actual Startup Time"]) 

def extract_plan(oid, sample, table_buf):
    global mp_optype
    # function: extract SQL feature
    # return: start_time, node feature, edge feature
    
    plan = sample["plan"]
    while isinstance(plan, list):
        plan = plan[0]
    # Features: print(plan.keys()) 
        # start time = plan["start_time"]
        # node feature = [Node Type, Total Cost:: Actual Total Time]
        # node label = [Actual Startup Time, Actual Total Time]

    plan = plan["Plan"] # root node
    node_matrix = []    
    edge_matrix = []   

    # mark operator id
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
                
        node_feature = [parent["oid"], mp_optype[parent["Node Type"]], run_cost, float(parent["Actual Startup Time"]), run_time]
        node_matrix = [node_feature] + node_matrix
        cnt = cnt + 1
        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)
                edge_matrix = [[node["oid"], parent["oid"], weight_type['passing']]] + edge_matrix

        # table conflict
        if 'Relation Name' in parent:
            table_buf[parent['Relation Name']].append(parent["oid"])   #todo

    # node: 18 * featuers
    # edge: 18 * 18

#    return float(sample["start_time"]), node_matrix, edge_matrix, cnt, maxnm
    return oid, node_matrix, edge_matrix, table_buf

def generate_graph(file_name):
    # todo: timestamp

    vmatrix = []
    ematrix = []
    access_buf = []

    oid = 0         # operator id
    # total = 0       # operator number
    # num = 0
    # mnm = 0         # max operator number

    with open(file_name, "r") as f:
        for sample in f.readlines():
            sample = json.loads(sample)
#            start_time, vertex, edge, cnt, tnm = extract_plan(oid, sample)
            vertices, edges, oid, table_conflict = extract_plan(oid, sample, table_job)
            vmatrix.append(vertices)
            ematrix.append(edges)

    #  data-sharing relation
    for tbl in table_buf:
        for i1 in tbl:
            for i2 in tbl:
                ematrix.append([i1, i2, weight_type['sharing']])


    #        total = total + cnt
    #        num = num + 1
    #        if tnm > mnm:
    #            mnm = tnm
#    return vmatrix, ematrix, total/num, mnm
    return vmatrix, ematrix

workloads = get_file(data_path, 'sample-plan-*')
for wid, workload in enumerate(workloads):
    # generate (V,E)
    vmatrix, ematrix = generate_graph(workload)

    # write (V,E) into files
    with open(graph_path + "sample-plan-" + str(wid) + ".content", "w") as wf:
        for v in vmatrix:
            wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
    with open(graph_path + "sample-plan-" + str(wid) + ".cites", "w") as wf:
        for e in ematrix:
            wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
    # print(level, num)



