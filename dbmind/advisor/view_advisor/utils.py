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
 
 

import time


class Query(object):
    def __init__(self, sql="", q_id=None):
        self.sql=sql
        self.id=q_id if q_id else hash(sql+str(time.time()))
        self.execution_time=None
        self.sql_psy_plan=None
        self.serialized_plan=None

class View(object):
    def __init__(self, sql="", v_id=None):
        self.sql=sql
        self.id=v_id if v_id else hash(sql+str(time.time()))
        self.frequency=0
        self.related_queries=[]
        self.execution_time=None    # TODO
        self.sql_psy_plan=None
        self.serialized_plan=None
        self.size=None              # TODO

class Query_view_pair(object):
    def __init__(self, query, view):
        self.query=query
        self.view=view
        self.query_view_execution_time=None
        self.benefit=0
        self.embedding=None