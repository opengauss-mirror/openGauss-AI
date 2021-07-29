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
 
 

import psycopg2
import sql_metadata
import configparser
import copy
import math

conf_path = "./config.ini"
config_raw = configparser.RawConfigParser()
config_raw.read(conf_path)


class postgresql_env(object):
    def __init__(self, dbname, schema_name, budget_size):
        self.dbname = dbname
        self.schema_name = schema_name
        self.budget_size = budget_size

        self.db_table_list = []
        self.db_column_list = []

        self.workload = None
        self.workload_raw_cost = 0

        conn = psycopg2.connect(database=config_raw.get(self.dbname, 'DB_name'),
                                user=config_raw.get(self.dbname, 'DB_username'),
                                password=config_raw.get(self.dbname, 'DB_password'),
                                host=config_raw.get('server', 'ip'),
                                port=config_raw.get('server', 'port'))
        cur = conn.cursor()

        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' and table_type = 'BASE TABLE';")
        res = cur.fetchall()

        for table in res:
            self.db_table_list.append(table[0])

            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='%s';" % table)
            col_res = cur.fetchall()
            col_tmp = []
            for col in col_res:
                col_tmp.append(col[0])
            self.db_column_list.append(col_tmp)

        cur.close()
        conn.close()

        self.len_db_column_list = [len(self.db_column_list[i]) for i in range(len(self.db_column_list))]

        self.action_space = []
        self.cur_state = None
        self.cur_virtual_index = []
        self.cur_cost = None
        self.cur_size_use = 0

        self.termial_wait_num = 0

    def get_input_num(self):
        return sum(self.len_db_column_list)

    def set_workload(self, workload):
        self.workload = workload
        self.original_action_space = self.sqlparse(self.workload)
        self.original_action_acum_pos = []
        for table_pos, col_pos in self.original_action_space:
            tmp_pos_num = sum(self.len_db_column_list[0:table_pos]) + col_pos
            self.original_action_acum_pos.append(tmp_pos_num)
        self.cur_virtual_index = []
        self.workload_raw_cost = self.cal_cur_cost()
        self.reset()

    def sqlparse(self, workload):
        """
        Extract table name and column name in  from the sql statement
        :param sql: sql statement
        :return: [action_space]
        """
        actionspace = []

        for sql in workload:
            sql_table_list = sql_metadata.get_query_tables(sql)
            tmp_col_list = sql_metadata.get_query_columns(sql)

            for cond in tmp_col_list:
                cond_list = cond.split('.')
                if len(cond_list) == 2:
                    if cond_list[0] in self.db_column_list:
                        table_pos = self.db_table_list.index(cond_list[0])
                        if cond_list[1] in self.db_column_list[table_pos]:
                            col_pos = self.db_column_list[table_pos].index(cond_list[1])
                            if [table_pos, col_pos] not in actionspace:
                                actionspace.append([table_pos, col_pos])
                else:
                    for table in sql_table_list:
                        if table not in self.db_table_list:
                            continue
                        table_pos = self.db_table_list.index(table)
                        if cond_list[0] in self.db_column_list[table_pos]:
                            col_pos = self.db_column_list[table_pos].index(cond_list[0])
                            if [table_pos, col_pos] not in actionspace:
                                actionspace.append([table_pos, col_pos])
                            break
        return actionspace

    def reset(self):
        origin = []
        for i in range(0, len(self.db_table_list)):
            origin.append([0] * len(self.db_column_list[i]))
        self.action_space = copy.deepcopy(self.original_action_space)
        self.action_acum_pos = copy.deepcopy(self.original_action_acum_pos)
        self.cur_state = origin
        self.cur_virtual_index = []
        self.cur_cost = self.cal_cur_cost()
        self.cur_size_use = 0
        self.termial_wait_num = 0

    def step(self, action):
        s = self.cur_state
        scost = self.cur_cost

        conn = psycopg2.connect(database=config_raw.get(self.dbname, 'DB_name'),
                                user=config_raw.get(self.dbname, 'DB_username'),
                                password=config_raw.get(self.dbname, 'DB_password'),
                                host=config_raw.get('server', 'ip'),
                                port=config_raw.get('server', 'port'))
        cur = conn.cursor()

        cur.execute("SELECT * FROM hypopg_create_index('CREATE INDEX ON %s (%s)');"
                    % (self.db_table_list[action[0]], self.db_column_list[action[0]][action[1]]))
        index_id = cur.fetchall()[0][0]

        cur.execute("SELECT * FROM hypopg_relation_size(%s);" % str(index_id))
        cur_action_size = cur.fetchall()[0][0]

        cur.close()
        conn.close()

        if action in self.cur_virtual_index:
            reward = 0
            done = False
            s_ = s
        else:
            if cur_action_size > self.budget_size - self.cur_size_use:
                self.termial_wait_num += 1
                if self.termial_wait_num > 5:
                    reward = 0
                    done = True
                    s_ = 'terminal'
                else:
                    reward = 0
                    done = False
                    s_ = s
            else:
                self.cur_state[action[0]][action[1]] = 1
                self.cur_virtual_index.append(action)
                self.cur_cost = self.cal_cur_cost()

                s_ = self.cur_state
                s_cost = self.cur_cost

                self.cur_size_use += cur_action_size

                reward = float(scost) - float(s_cost)
                self.action_acum_pos.pop(self.action_space.index(action))
                self.action_space.remove(action)
                done = False

        return s_, reward, done

    def cal_cur_cost(self, print_info=False):
        conn = psycopg2.connect(database=config_raw.get(self.dbname, 'DB_name'),
                                user=config_raw.get(self.dbname, 'DB_username'),
                                password=config_raw.get(self.dbname, 'DB_password'),
                                host=config_raw.get('server', 'ip'),
                                port=config_raw.get('server', 'port'))
        cur = conn.cursor()

        for table_pos, col_pos in self.cur_virtual_index:
            cur.execute("SELECT * FROM hypopg_create_index('CREATE INDEX ON %s (%s)');" %
                        (self.db_table_list[table_pos], self.db_column_list[table_pos][col_pos]))

        workload_cost = 0
        for sql in self.workload:
            cur.execute('EXPLAIN ' + sql)
            res = cur.fetchall()[0][0]

            start_pos = res.find('cost=') + 5
            end_pos = res.find('rows=')

            time_interval = res[start_pos:end_pos].strip()

            workload_cost += float(time_interval.split('..')[1])

        if print_info:
            cur.execute("SELECT * FROM hypopg_list_indexes();")
            res = cur.fetchall()
            print(res)

        cur.close()
        conn.close()

        return workload_cost


if __name__ == '__main__':
    test_sql = "select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority \
            from customer, orders, lineitem\
            where c_mktsegment = 'AUTOMOBILE' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-22' and l_shipdate > date '1995-03-22' \
            group by l_orderkey, o_orderdate, o_shippriority \
            order by revenue desc, o_orderdate;"
    env = postgresql_env('tpch', 'public', 200 * 1024 * 1024)
    env.set_sql(test_sql)
    print(env.get_input_num())
