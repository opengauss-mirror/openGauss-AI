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
 
 

#!/usr/bin/python
from __future__ import division, print_function, unicode_literals

import csv
import json
import os
import random
import re
import string
import sys
import unicodedata
from io import open
import time
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np

o_path = os.getcwd()
sys.path.append(o_path)

import config as cfg

import database

from tools.sql_parser import Sql_parser
from tools.run_query import format_create_mv
from tools.preprocess_data import Preprocess_data

from utils import Query, View, Query_view_pair
from algorithms.encoder_reducer import Encoder_reducer, EncoderRNN, AttnReducerRNN, Lang, Scale_log_std
from algorithms.DQN import Environment, DQN

class Autoview(object):

    def __init__(self,**kw_args):

        self.workload=[]
        self.views=[]
        self.MVs=[]
        self.estimated_pairs=[]
        self.constraint={
            "size":100,
            "number":3
        }

        self.database=None
        self.view_minner=None

        self.encoder_reducer=Encoder_reducer()
        self.encoder_reducer.prepare_data(o_path+"/"+cfg.enc_rdc_trainset_dir)
        self.encoder_reducer.load_model(o_path+"/"+cfg.enc_rdc_model_path)

        self.env=Environment(budget=cfg.dqn_budget)
        self.DQN=DQN(self.env)

        self.preprocess_data=Preprocess_data()

        self.db=database.Database(section=cfg.local_pg)


    def mine_views(self, workload):
        pass
        #self.views=self.view_minner()
        
        # test
        query_lis=workload[:min(3, len(workload))]
        view_lis=[]
        for query in query_lis:
            view=View(sql=query.sql)
            view.frequency=1
            view.related_queries=[query]
            view_lis.append(view)
        self.views=view_lis

        return self.views
        
    def estimate_views(self, workload, views):

        # serialize
        self.db.connect()
        for query in workload: self.serialize_sql_plan(query)
        for view in views: self.serialize_sql_plan(view)
        for query in workload: self.extract_info(query)
        for view in views: self.extract_info(view)
        self.db.close()

        # estimate
        estimated_pairs=[]
        for view in views:
            for query in view.related_queries:
                evl_q_time, evl_q_mv_time, attn, hidd=self.encoder_reducer.evaluate(query.serialized_plan, view.serialized_plan)
                print("estimated q_tim", evl_q_time)
                print("estimated q_mv_tim", evl_q_mv_time)

                qv_pair=Query_view_pair(query, view)
                query.execution_time=evl_q_time
                qv_pair.query_view_execution_time=evl_q_mv_time
                qv_pair.benefit=evl_q_time-evl_q_mv_time

                qv_pair.embedding=hidd
                estimated_pairs.append(qv_pair)

        self.estimated_pairs=estimated_pairs

        return self.estimated_pairs

    def select_views(self, workload, views, estimated_pairs):
        # select MVs from views

        # build environment
        self.env.build(workload, views, estimated_pairs)

        self.DQN=DQN(self.env)

        self.DQN.evaluate(1)

        result_view_list=self.DQN.get_result_view_list()

        return result_view_list

    def recommend_MVs(self, workload):

        # test
        # import random

        # MVs=random.sample(workload, min(self.constraint["number"], len(workload)))

        # self.MVs=[format_create_mv("mv"+str(hash(MV))[-5:], MV) for MV in MVs]

        # return self.MVs

        # load workload
        self.load_workload(workload)

        # mine view candidates
        self.views=self.mine_views(self.workload)

        # encoder reducer
        self.estimated_pairs=self.estimate_views(self.workload, self.views)

        # DQN
        self.MVs=self.select_views(self.workload, self.views, self.estimated_pairs)

        # output
        self.MVs_create=[format_create_mv("mv"+str(hash(MV))[-5:], MV.sql) for MV in self.MVs]
        return self.MVs_create

    
    def load_workload(self, workload):
        self.workload=[Query(sql=x) for x in workload]
        return self.workload


    def get_query_plan(self, query):

        result=self.db.get_explain(query.sql, analyze=False, costs=True, fmt='json')
        result=result[0][0][0]
        query.sql_psy_plan=result
        return result
        

    
    def serialize_sql_plan(self, query):
        # for query and view
        
        if not query.sql_psy_plan:
            query.sql_psy_plan = self.get_query_plan(query)

        query.serialized_plan=self.preprocess_data.plantree2seq(query.sql_psy_plan)

        return query.serialized_plan

    def extract_info(self, query):
        query.execution_time=query.sql_psy_plan['Plan']['Total Cost']*0.01
        if hasattr(query, "size"):
            query.size=query.sql_psy_plan['Plan']['Plan Rows']*0.001
        



if __name__=="__main__":
    import workloads
    workload=workloads.load_workload("JOB")

    autoview=Autoview()

    MVs=autoview.recommend_MVs(workload)
    print(MVs)

    # query=Query()
    # print(query.sql)

    