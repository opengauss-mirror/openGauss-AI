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
import time


data_path = "/Users/xuanhe/Documents/mypaper/workload-performance/pygcn-master/pmodel_data/"
plan = data_path
workload_size = 10
workload_num = 3184

edge_dim = 30
node_dim = 30

# generate graph data

mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5}
oid = 0


def split_planset(fname, data_path = data_path):
    # split the plan samples into the workload samples    
    wid = 0
    planbuf = []
    i = 0
    with open(data_path + fname,  "r") as rf:
        for f in rf.readlines():
            planbuf.append(f)
            i = i + 1
            if i%workload_size == 0:
                with open(data_path + "sample-plan-" + str(wid) + ".txt", "w") as wf:
                    for plan in planbuf:
                        wf.write(plan)
                planbuf = []
                wid = wid + 1
        

# split plan dataset
split_planset("tim-plan-job-11-20")

for wid in range(1):
    st = time.time()
    vmatrix, ematrix = generate_graph(wid) 
    print(time.time() - st)

    
    with open(data_path + "graph/" + "sample-plan-" + str(wid) + ".content", "w") as wf:
            for v in vmatrix:
                wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
    with open(data_path + "graph/" + "sample-plan-" + str(wid) + ".cites", "w") as wf:
            for e in ematrix:
                wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")                