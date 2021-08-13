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
 
 

import sys, os
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(o_path+"/tools" )

from tools.sql_parser import Sql_parser

workload_paths={
    "JOB":"./dataset/JOB/job_workload.sql"
}

def load_workload(workload_name):
    file_path=workload_paths[workload_name]

    sql_par=Sql_parser()
    with open(file_path,"r") as fp:
        workload_str=fp.read()
    workload=sql_par.split_sqls(workload_str)

    return workload