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

from config import CONFIG

# execute sql
def execute_sql(sql):
    conn = psycopg2.connect(database=CONFIG['schema'],  # tpch1x (0.1m, 10m), tpch100m (100m)
                            user=CONFIG['username'],
                            password=CONFIG['password'],
                            host=CONFIG['host'],
                            port=CONFIG['port'])
    fail = 0
    cur = conn.cursor()
    try:
        cur.execute(sql)
    except:
        fail = 1
    res = []
    if fail == 0:
        res = cur.fetchall()

    conn.commit()  # todo
    cur.close()
    conn.close()

    return res

def fetch_execution_time(res):
    for r in res:
        if "Execution" in r[0]:
            return r[0]

    return -1