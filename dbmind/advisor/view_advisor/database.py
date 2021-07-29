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
#test python connect postgresql/openGauss
import psycopg2
import config as cfg
from tools.expr_parser import Expr_parser
from tools import formattrans

class Database(object):
    def __init__(self,**kw_args):
        self.conn=None
        self.expr_parser=Expr_parser()
        self.db_params = kw_args['section'] if 'section' in kw_args else cfg.local_pg
    
    def connect(self):
        """ Connect to the PostgreSQL/openGauss database server """
        try:
            if self.conn is not None and not self.conn.closed:
                self.conn.close()

            # read connection parameters
            # postgresql/openGauss: local
            # local_pg: .61 server
            params = self.db_params
    
            # connect to the PostgreSQL/openGauss server
            print('Connecting to the openGauss database...')
            self.conn = psycopg2.connect(**params)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            if self.conn is not None:
                self.conn.close()
                return
    
    def close(self):
        if self.conn is not None and not self.conn.closed:
            self.conn.close()

    def commit(self):
        self.conn.commit()

    def execute_stmt(self, stmt):
        with self.conn.cursor() as cur:
            cur.execute(stmt)
            result=cur.fetchall()
            return result

    def execute_stmt_no_fetch(self, stmt):
        with self.conn.cursor() as cur:
            cur.execute(stmt)

    def get_explain(self, stmt, analyze=False, costs=True, fmt='json'):
        prefix='EXPLAIN (ANALYZE {0}, COSTS {1}, FORMAT {2})\n'.format(analyze, costs, fmt)
        explain_stmt=prefix+stmt
        result=self.execute_stmt(explain_stmt)
        
        # print(formattrans.dict2json(result[0][0][0]))
        return result

    def get_planseq(self, stmt, **kw_args):
        return self.plantree2seq(self.get_explain(stmt,**kw_args))

    def plantree2seq(self, plan):
        seq_lis=[]
        self.traverse_plantree(plan[0][0][0]['Plan'],seq_lis)
        return seq_lis
    
    def traverse_plantree(self, plan, result_seq):
        if 'Plans' in plan:
            for sub_plan in plan['Plans']:
                self.traverse_plantree(sub_plan, result_seq)
        result_seq.extend(self.node2seq(plan))

    def node2seq(self, plan):
        key_list=[("Relation Name",False), ("Index Name",False), ("Index Cond",True), ("Hash Cond",True), ("Filter",True), ("Join Cond",True),("Join Type",False), ("Node Type",False), ("Join Filter",True),("Recheck Cond", True)]
        result_lis=[]
        for key, need_parse in key_list:
            if key in plan:
                if need_parse:
                    result_lis.extend(self.expr_parser.parse_expr(plan[key]))
                else:
                    result_lis.append(plan[key])
        return result_lis
 
if __name__ == '__main__':
    db=Database(section='laptop')
    db.connect()
    stmt="select e.fname, e.minit, e.lname from employee e, works_on w, project p where e.dno = 5 and e.ssn = w.essn and w.pno = p.pnumber and p.pname = 'ProductX' and w.hours > 10;"
    # stmt="select e.fname, d.dname, e.salary from employee e, department d;"
    # stmt="select * from company_name limit 3;"
#     stmt="""
# SELECT MIN(t.title) AS movie_title
# FROM keyword AS k,
#      movie_info AS mi,
#      movie_keyword AS mk,
#      title AS t
# WHERE k.keyword LIKE '%sequel%'
#   AND mi.info IN ('Sweden',
#                   'Norway',
#                   'Germany',
#                   'Denmark',
#                   'Swedish',
#                   'Denish',
#                   'Norwegian',
#                   'German')
#   AND t.production_year > 2005
#   AND t.id = mi.movie_id
#   AND t.id = mk.movie_id
#   AND mk.movie_id = mi.movie_id
#   AND k.id = mk.keyword_id;
# """
    explain=db.get_explain(stmt)
    print(formattrans.dict2json(explain[0][0][0]))

    # result=db.execute_stmt(stmt)
    # print(result)
    seq_lis=db.get_planseq(stmt)
    #print(' '.join(seq_lis))
    print(', '.join(seq_lis))