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