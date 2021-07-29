import sys, os
o_path = os.getcwd()
o_path=o_path+"/./" # single run on server
sys.path.append(o_path)

import database
from tools import sql_parser, expr_parser, formattrans
import random, json, csv, re

class Preprocess_data(object):
    def __init__(self):
        self.expr_parser=expr_parser.Expr_parser()
        self.alias_table_map=dict()

    # pattern: regex pattern like r'\d*_((\d*)[a-z]).json'   group(1) is id, group(2) is sort key
    # return plans: [(id, plan),...]
    def get_plans_from_dir_with_pattern(self, dir_path, pattern):
        file_list=os.listdir(dir_path)
        rc=re.compile(pattern)
        file_lis=[dir_path+x for x in file_list if rc.match(x)]
        file_lis.sort(key=lambda x:int(rc.search(x).group(2)))

        ids=[rc.search(x).group(1) for x in file_lis]
        basename_lis=[os.path.splitext(os.path.basename(x))[0] for x in file_lis]

        plans=[]
        for id, filename in zip(ids, file_lis):
            with open(filename) as fp:
                plan=json.load(fp)
            plans.append((id, plan))
        
        return plans


    # add this pt_ver because pretrain data has multiple plan in one file
    # pattern: regex pattern like r'\d*_((\d*)[a-z]).json'   group(1) is id, group(2) is sort key
    # return plans: [(id_0, plan),(id_1, plan),...]
    def get_plans_from_dir_with_pattern_pt_ver(self, dir_path, pattern):
        file_list=os.listdir(dir_path)
        rc=re.compile(pattern)
        file_lis=[dir_path+x for x in file_list if rc.match(x)]
        file_lis.sort(key=lambda x:int(rc.search(x).group(2)))

        ids=[rc.search(x).group(1) for x in file_lis]
        basename_lis=[os.path.splitext(os.path.basename(x))[0] for x in file_lis]

        plans=[]
        for id, filename in zip(ids, file_lis):
            with open(filename) as fp:
                plans2=json.load(fp)
                cnt=0
                for plan in plans2:
                    plans.append(("{0}_{1}".format(id, cnt), plan))
                    cnt+=1
        return plans
    
    # used to check, in plan, which key we need to extract and what value pattern we need to consider in parsing expression
    def list_all_interest_entry_in_plan(self):
        q_dir=o_path+"result/job/q/"
        mv_dir=o_path+"result/job/mv/"
        q_mv_dir=o_path+"result/job/q-mv/"

        q_id_plans=self.get_plans_from_dir_with_pattern(q_dir, r'\d+_((\d+)[a-z]).json')
        mv_id_plans=self.get_plans_from_dir_with_pattern(mv_dir, r'(mv(\d+)).json')
        
        result_seq=[]
        for id, plan in q_id_plans:
            self.traverse_plantree(plan['Plan'], result_seq, self.extract_interest_key_value)

        key_set=set()
        for key, value in result_seq:
            key_set.add(key)
        print("\n".join(list(key_set)))

        for key, value in result_seq:
            print(key, value)
        
        # print([x[0] for x in q_id_plans])
        # print([x[0] for x in mv_id_plans])

    # make dataset querydata.csv mvdata.csv
    def make_dataset(self):
        q_dir=o_path+"result/job/q/"
        mv_dir=o_path+"result/job/mv/"
        q_mv_dir=o_path+"result/job/q-mv/"
        q_pt_dir=o_path+"result/job/q_pt/"

        trainset_dir=o_path+"dataset/JOB/trainset/"

        # # DEAL with q

        # # q_id_plans2: [(id, plan),]
        # q_id_plans2=self.get_plans_from_dir_with_pattern(q_dir, r'\d+_((\d+)[a-z]).json')
        
        # q_id_plans_c_t2=[(x[0],x[1],*self.get_cost_and_time_from_plan(x[1])) for x in q_id_plans2]
        # q_id_plans_ctlis_lis=self.merge_multiple_run(q_id_plans_c_t2)
        # q_id_seq_c_t=[]
        # for id, plan, ctlis in q_id_plans_ctlis_lis:
        #     cos_lis=[x[0] for x in ctlis]
        #     tim_lis=[x[1] for x in ctlis]
        #     avg_cos=sum(cos_lis)/len(cos_lis)
        #     avg_tim=sum(tim_lis)/len(tim_lis)
        #     q_id_seq_c_t.append((id, json.dumps(self.plantree2seq(plan)), avg_cos, avg_tim))
        # rc=re.compile(r'(\d+)[a-z]')
        # q_id_seq_c_t.sort(key=lambda x: int(rc.search(x[0]).group(1)))
        
        # # write to csv: [id, seq_json, cost, time]
        # with open(trainset_dir+"querydata.csv","w",newline='') as fp:
        #     csv_writer=csv.writer(fp)
        #     csv_writer.writerows(q_id_seq_c_t)

        # # DEAL with mv
        # with open(mv_dir+"mv_sizes.csv") as fp:
        #     csv_reader=csv.reader(fp)
        #     mv_sizes=dict()
        #     for line in csv_reader:
        #         mv_sizes[line[0]]=line[3]

        # mv_id_plans2=self.get_plans_from_dir_with_pattern(mv_dir, r'(mv(\d+)).json')
        
        # mv_id_plans_c_t2=[(x[0],x[1],*self.get_cost_and_time_from_plan(x[1])) for x in mv_id_plans2]
        # mv_id_plans_ctlis_lis=self.merge_multiple_run(mv_id_plans_c_t2)
        # mv_id_seq_c_t=[]
        # for id, plan, ctlis in mv_id_plans_ctlis_lis:
        #     cos_lis=[x[0] for x in ctlis]
        #     tim_lis=[x[1] for x in ctlis]
        #     avg_cos=sum(cos_lis)/len(cos_lis)
        #     avg_tim=sum(tim_lis)/len(tim_lis)
        #     mv_size=mv_sizes[id]
        #     mv_id_seq_c_t.append((id, json.dumps(self.plantree2seq(plan)), avg_cos, avg_tim, mv_size))
        # rc=re.compile(r'mv(\d+)')
        # mv_id_seq_c_t.sort(key=lambda x: int(rc.search(x[0]).group(1)))
        
        # # write to csv: [id, seq_json, cost, time, size]
        # with open(trainset_dir+"mvdata.csv","w",newline='') as fp:
        #     csv_writer=csv.writer(fp)
        #     csv_writer.writerows(mv_id_seq_c_t)

        # # DEAL with q-mv

        # # q_id_plans2: [(id, plan),]
        # q_mv_id_plans2=self.get_plans_from_dir_with_pattern(q_mv_dir, r'\d+_((\d+)[a-z]-\d+).json')
        
        # q_mv_id_plans_c_t2=[(x[0],x[1],*self.get_cost_and_time_from_plan(x[1])) for x in q_mv_id_plans2]
        # q_mv_id_plans_ctlis_lis=self.merge_multiple_run(q_mv_id_plans_c_t2)
        # q_mv_id_seq_c_t=[]
        # for id, plan, ctlis in q_mv_id_plans_ctlis_lis:
        #     cos_lis=[x[0] for x in ctlis]
        #     tim_lis=[x[1] for x in ctlis]
        #     avg_cos=sum(cos_lis)/len(cos_lis)
        #     avg_tim=sum(tim_lis)/len(tim_lis)
        #     q_mv_id_seq_c_t.append((id, json.dumps(self.plantree2seq(plan)), avg_cos, avg_tim))
        # rc=re.compile(r'(\d+)[a-z]')
        # q_mv_id_seq_c_t.sort(key=lambda x: int(rc.search(x[0]).group(1)))
        
        # # write to csv: [id, seq_json, cost, time]
        # with open(trainset_dir+"query-mvdata.csv","w",newline='') as fp:
        #     csv_writer=csv.writer(fp)
        #     csv_writer.writerows(q_mv_id_seq_c_t)


        # DEAL with q_pt

        # q_id_plans: [(id, plan),]
        q_id_plans=self.get_plans_from_dir_with_pattern_pt_ver(q_pt_dir, r'((\d+)[a-z]_modified).json')
        
        q_id_plans_c_t=[(x[0],x[1],*self.get_cost_and_time_from_plan(x[1])) for x in q_id_plans]
        q_id_seq_c_t=[]
        for id, plan, cos, tim in q_id_plans_c_t:
            q_id_seq_c_t.append((id, json.dumps(self.plantree2seq(plan)), cos, tim))
        rc=re.compile(r'(\d+)[a-z]_modified')
        q_id_seq_c_t.sort(key=lambda x: int(rc.search(x[0]).group(1)))
        
        # write to csv: [id, seq_json, cost, time]
        with open(trainset_dir+"query_pretraindata.csv","w",newline='') as fp:
            csv_writer=csv.writer(fp)
            csv_writer.writerows(q_id_seq_c_t)


    # [(id, plan, cost, time),] => [(id, plan, [(cost, time),]),]
    def merge_multiple_run(self, id_plans_c_t):
        # reduce mutiple runs and calculate average cost and time
        id_ctlis_dic=dict()
        result_lis=[]
        for x in id_plans_c_t:
            if x[0] not in id_ctlis_dic:
                id_ctlis_dic[x[0]]=[(x[2],x[3])]
                result_lis.append((x[0],x[1],[]))
            else:
                id_ctlis_dic[x[0]].append((x[2],x[3]))
        for x in result_lis:
            x[2].extend(id_ctlis_dic[x[0]])
        
        return result_lis

    def get_cost_and_time_from_plan(self, plan):
        cost=plan['Plan']['Total Cost']
        time=plan['Plan']['Actual Total Time']
        return cost, time

    def plantree2seq(self, plan):
        # print("begin ex tb alias")
        result_seq=[]
        self.traverse_plantree(plan['Plan'], result_seq, self.extract_table_alias)
        self.alias_table_map=dict()
        for table, alias in result_seq:
            self.alias_table_map[alias]=table
        # print("begin ex seq")
        
        result_seq=[]
        self.traverse_plantree(plan['Plan'], result_seq, self.extract_seq)
        return result_seq

    def traverse_plantree(self, plan, result_seq, extract_node_fun):
        if 'Plans' in plan:
            for sub_plan in plan['Plans']:
                self.traverse_plantree(sub_plan, result_seq, extract_node_fun)
        try:
            result_seq.extend(extract_node_fun(plan))
        except Exception as e:
            print(e)
            print(plan)
            raise Exception("errere")

    def extract_seq(self, plan):
        key_list=[("Relation Name","identifier"), 
                ("Index Name","identifier"), 
                ("Filter","expression"),
                ("Index Cond","expression"), 
                ("Hash Cond","expression"), 
                ("Join Filter","expression"),
                ("Join Cond","expression"),
                ("Join Type","keyword"), 
                ("Recheck Cond", "expression"),
                ("Node Type","keyword")]
        result_lis=[]
        for key, e_type in key_list:
            if key in plan:
                if e_type=="expression":
                    table_name=plan['Alias'] if 'Alias' in plan else None
                    result_lis.extend(self.expr_parser.parse_expr(plan[key], table_name, self.alias_table_map))
                else:
                    result_lis.append((plan[key],e_type))
        return result_lis

    def extract_table_alias(self, plan):
        if "Relation Name" in plan and "Alias" in plan:
            return [(plan["Relation Name"],plan["Alias"])]
        else:
            return []

    def extract_interest_key_value(self, plan):
        def is_interest(key, value):
            interest_key=['Cond', 'Index', 'Filter', 'Join', 'Name', 'Node', 'Type', 'Alias']
            for itrs_key in interest_key:
                if itrs_key in key:
                    return True
            return False

        result_seq=[]
        for key, value in plan.items():
            if is_interest(key, value):
                result_seq.append((key, value))

        return result_seq

        


if __name__=="__main__":
    pre_data=Preprocess_data()
    pre_data.make_dataset()