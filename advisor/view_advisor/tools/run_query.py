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

import database
import sql_parser
import formattrans
import random, json, csv, re

sql_dir=o_path+"./dataset/JOB/join-order-benchmark/"
res_dir=o_path+"./result/job/"
mv_dir=o_path+"./result/job/mv/"
q_mv_dir=o_path+"./dataset/JOB/job-mv-q-mv/"
res_q_mv_dir=o_path+"./result/job/q-mv/"
res_dir=o_path+"./result/job/q/"
res_mv_scan_time_dir=o_path+"./result/job/mv_scan_time/"

sql_q_pt_dir=o_path+"./dataset/JOB/daji/1to10_0420/plans/"
res_q_pt_dir=o_path+"./result/job/q_pt/"

def run_q():
    file_list=os.listdir(sql_dir)
    rc=re.compile(r'(\d*)[a-z].sql')
    lis=[sql_dir+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    id_sql_lis=list(zip(ids,sqls))
    id_sql_lis2=id_sql_lis+id_sql_lis
    random.shuffle(id_sql_lis2)

    i_id_sql_lis2=[(i,id_sql_lis2[i][0],id_sql_lis2[i][1]) for i in range(len(id_sql_lis2))]

    db=database.Database(section='local_pg')
    db.connect()
    for i, id, sql in i_id_sql_lis2:
        result=db.get_explain(sql, analyze=True, costs=True, fmt='json')
        result=formattrans.dict2json(result[0][0][0])
        res_filename=res_dir+"{0}_{1}.json".format(i,id)
        with open(res_filename,"w") as fp:
            fp.write(result)
        print(i, id, "done")
    db.close()

def run_q_mv():
    file_list=os.listdir(q_mv_dir)
    rc=re.compile(r'(\d*)[a-z]-(\d*).sql')
    lis=[q_mv_dir+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    # ids: 1a-1
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    id_sql_lis=list(zip(ids,sqls))
    id_sql_lis2=id_sql_lis+id_sql_lis
    random.shuffle(id_sql_lis2)

    i_id_sql_lis2=[(i,id_sql_lis2[i][0],id_sql_lis2[i][1]) for i in range(len(id_sql_lis2))]

    db=database.Database(section='local_pg')
    db.connect()
    for i, id, sql in i_id_sql_lis2:
        print(i, id, "begin")
        sys.stdout.flush()
        try:
            result=db.get_explain(sql, analyze=True, costs=True, fmt='json')
        except Exception as e:
            print(e)
            # db.close()
            # break
            db.commit()
            continue
        result=formattrans.dict2json(result[0][0][0])
        res_filename=res_q_mv_dir+"{0}_{1}.json".format(i,id)
        with open(res_filename,"w") as fp:
            fp.write(result)
        print(i, id, "done")
    db.close()

def trans_pt_json_to_result_json():
    file_list=os.listdir(sql_q_pt_dir)
    rc=re.compile(r'(\d*)[a-z]_modified.json')
    lis=[sql_q_pt_dir+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    

    for filename in lis:
        id=os.path.splitext(os.path.basename(filename))[0]
        result=get_plan_from_json_files_pt_ver([filename])
        # [{}]\n[{}] -> [{},{}]
        plan_json_lis=[plan[0] for plan in result]
        res=formattrans.dict2json(plan_json_lis)   # actual list2json
        res_filename=res_q_pt_dir+"{0}.json".format(id)
        with open(res_filename,"w") as fp:
            fp.write(res)
        print(id, "done") 

def get_plan_from_json_files(file_lis):
    res_lis=[]
    for filename in file_lis:
        with open(filename) as fp:
            js=json.load(fp)
        res_lis.append(js)
    return res_lis

def get_plan_from_json_files_pt_ver(file_lis):
    res_lis=[]
    for filename in file_lis:
        with open(filename) as fp:
            lines=fp.readlines()
            for line in lines:
                js=json.loads(line)
                res_lis.append(js)
    return res_lis

def get_cost_and_time_from_json_files(dir_path):
    file_list=os.listdir(dir_path)
    rc=re.compile(r'(\d*)_((\d*)[a-z]).json')
    lis=[dir_path+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(3)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]    
    
    plans=get_plan_from_json_files(lis)
    cost_time_dic=dict()
    for i_id, plan in zip(ids, plans):
        id=re.search(r'(\d*)_((\d*)[a-z])',i_id).group(2)
        cost=plan['Plan']['Total Cost']
        time=plan['Plan']['Actual Total Time']
        if id not in cost_time_dic:
            cost_time_dic[id]=[(cost,time)]
        else:
            cost_time_dic[id].append((cost,time))


    res_lis=[]
    for id, ct_lis in cost_time_dic.items():
        res_lis.append([id]+list(ct_lis[0])+list(ct_lis[1]))
        
    with open(o_path+"tools/"+"q_cost_time.csv","w",newline='') as fp:
        csv_writer=csv.writer(fp)
        csv_writer.writerows(res_lis)
    

def check_query_legal(dir_path):
    file_list=os.listdir(dir_path)
    rc=re.compile(r'((\d*)[a-z])-(\d*).sql')
    lis=[dir_path+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(2)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]    

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    rc2=re.compile(r';$')
    rc3=re.compile(r',[ |\t|\n]*WHERE')

    res_lis=[]
    for qid_mvid, sql in zip(ids, sqls):
        tables=sql_par.extract_tables(sql)
        table_alias=[x[1] for x in tables]

        join_conditions=sql_par.extract_join_conditions(sql)
        res_legal=True
        for lef, rig in join_conditions:
            if lef[0] not in table_alias or rig[0] not in table_alias:
                res_legal=False
                break

        sql2=sql.strip()
        if not rc2.search(sql2) or rc3.search(sql2):
            res_legal=False

        res_lis.append((qid_mvid,res_legal))

    flag=True
    for qid_mvid, res in res_lis:
        print(qid_mvid+"\t"+str(res))
        if not res:
            flag=False
    print("All legal", flag)

    return res_lis

def check_mv_legal(dir_path):
    file_list=os.listdir(dir_path)
    rc=re.compile(r'mv(\d*).sql')
    lis=[dir_path+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]    

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    rc2=re.compile(r';$')
    rc3=re.compile(r',[ |\t|\n]*WHERE')

    res_lis=[]
    for mvid, sql in zip(ids, sqls):
        tables=sql_par.extract_tables(sql)
        table_alias=[x[1] for x in tables]

        join_conditions=sql_par.extract_join_conditions(sql)
        res_legal=True
        for lef, rig in join_conditions:
            if lef[0] not in table_alias or rig[0] not in table_alias:
                res_legal=False
                break

        sql2=sql.strip()
        if not rc2.search(sql2) or rc3.search(sql2):
            res_legal=False

        res_lis.append((mvid,res_legal))

    flag=True
    for mvid, res in res_lis:
        print(mvid+"\t"+str(res))
        if not res:
            flag=False
    print("All legal", flag)

    return res_lis

def create_mv(dir_path):
    file_list=os.listdir(dir_path)
    rc=re.compile(r'mv(\d*).sql')
    lis=[dir_path+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    id_sql_lis=list(zip(ids,sqls))

    db=database.Database(section='local_pg')
    db.connect()
    for id, sql in id_sql_lis:
        sql=format_create_mv(id, sql)
        db.execute_stmt_no_fetch(sql)
        db.commit()
        print(id, "done")

        # EXPLAIN
        # result=db.get_explain(sql, analyze=True, costs=True, fmt='json')
        # result=formattrans.dict2json(result[0][0][0])
        # res_filename=mv_dir+"{0}.json".format(id)
        # with open(res_filename,"w") as fp:
        #     fp.write(result)
        # print(id, "done")

    db.close()

def get_mv_size(dir_path):
    file_list=os.listdir(dir_path)
    rc=re.compile(r'mv(\d*).sql')
    lis=[dir_path+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]

    db=database.Database(section='local_pg')
    db.connect()

    stmt="SELECT pg_relation_size('{0}'), pg_indexes_size('{0}'), pg_total_relation_size('{0}');"
    mv_sizes=[]
    for id in ids:
        result=db.execute_stmt(stmt.format(id))
        mv_sizes.append([id]+list(result[0]))
    
    with open(mv_dir+"mv_sizes.csv","w",newline='') as fp:
        csv_writer=csv.writer(fp)
        csv_writer.writerows(mv_sizes)

    db.close()

# id, "Execution Time", "Plan Rows","Plan Width"
def get_mv_scan_time(dir_path):
    if not os.path.exists(res_mv_scan_time_dir):
        os.makedirs(res_mv_scan_time_dir)
    file_list=os.listdir(dir_path)
    rc=re.compile(r'mv(\d*).sql')
    lis=[dir_path+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    id_sql_lis=list(zip(ids,sqls))

    db=database.Database(section='local_pg')
    db.connect()

    # stmt="SELECT pg_relation_size('{0}'), pg_indexes_size('{0}'), pg_total_relation_size('{0}');"
    stmt="SELECT * from {0};"
    mv_scan_data=[]
    for id, sql in id_sql_lis:
        sql=stmt.format(id)
        result=db.get_explain(sql, analyze=True, costs=True, fmt='json')
        result=result[0][0][0]
        r_js=formattrans.dict2json(result)
        res_filename=res_mv_scan_time_dir+"{0}.json".format(id)
        with open(res_filename,"w") as fp:
            fp.write(r_js)
        # print(result)
        # print(result["Plan"], type(result["Plan"]))
        entry=[id,result["Execution Time"],result["Plan"]["Plan Rows"],result["Plan"]["Plan Width"]]
        mv_scan_data.append(entry)
        print(id, "done")


    
    with open(res_mv_scan_time_dir+"mv_scan.csv","w",newline='') as fp:
        csv_writer=csv.writer(fp)
        csv_writer.writerows(mv_scan_data)

    db.close()
    

def format_create_mv(mv_name, sql):
    sql=sql.strip()
    sql=re.sub(r';$','',sql)
    template=\
"""CREATE MATERIALIZED VIEW IF NOT EXISTS {0} AS
{1}
WITH DATA;"""
    sql=template.format(mv_name, sql)
    return sql

def fix_mv_rows():
    # iter all querys and build all_tb_att {'ct':{'company_type','baabala'}}
    file_list=os.listdir(sql_dir)
    rc=re.compile(r'(\d*)[a-z].sql')
    lis=[sql_dir+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    all_tb_att=dict()
    for sql in sqls:
        tables=sql_par.extract_tables(sql)

        identifiers=sql_par.extract_identifiers(sql)
        for identi in identifiers:
            tb1=identi[0]

            if not tb1 in all_tb_att:
                all_tb_att[tb1]=set([identi[1]])
            else:
                all_tb_att[tb1].add(identi[1])

    # iter all mvs and fix them
    file_list=os.listdir(q_mv_dir)
    rc=re.compile(r'mv(\d*).sql')
    lis=[q_mv_dir+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]    

    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    for id, sql in zip(ids,sqls):
        tables=sql_par.extract_tables(sql)

        tb_att_lis=[]
        for table, table_alias in tables:
            new_lis=[table_alias+"."+x for x in list(all_tb_att[table_alias])]
            tb_att_lis.extend(new_lis)

        tb_att_string=",\n        ".join(tb_att_lis)
        new_sql=sql.replace("*",tb_att_string)
        with open(q_mv_dir+id+"+.sql","w") as fp:
            fp.write(new_sql)

def create_index_script_on_mv():
    def format_create_idx(id, col):
        ss="create index {1}_{0} on {0}({1});"
        return ss.format(id,col)

    file_list=os.listdir(q_mv_dir)
    rc=re.compile(r'mv(\d*).sql')
    lis=[q_mv_dir+x for x in file_list if rc.match(x)]
    lis.sort(key=lambda x:int(rc.search(x).group(1)))
    ids=[os.path.splitext(os.path.basename(x))[0] for x in lis]

    sql_par=sql_parser.Sql_parser()
    sqls=sql_par.get_sql_list_from_file_or_string(sqls_files=lis)

    id_sql_lis=list(zip(ids,sqls))

    idx_col_lis=['id','company_id','company_type_id','info_type_id','keyword_id','kind_id','linked_movie_id','link_type_id','movie_id','person_id','person_role_id','role_id']  

    script_lis=[]
    for id, sql in id_sql_lis:
        identi_lis=sql_par.extract_selects(sql)
        for tb_alias, col in identi_lis:
            if col in idx_col_lis:
                script_lis.append(format_create_idx(id,col))
    
    with open(q_mv_dir+"fkindex_mv.sql","w") as fp:
        fp.write("\n".join(script_lis))
    


if __name__=="__main__":
    # get_cost_and_time_from_json_files(res_dir)
    # res_lis=check_query_legal(q_mv_dir)
    # res_lis=check_mv_legal(q_mv_dir)
    
    # create_mv(q_mv_dir)
    # fix_mv_rows()
    
    # get_mv_size(q_mv_dir)
    
    # run_q_mv()
    # create_index_script_on_mv()

    # pretrain
    # trans_pt_json_to_result_json()


    # test mv scan time
    # get_mv_scan_time(q_mv_dir)

    print("end")

