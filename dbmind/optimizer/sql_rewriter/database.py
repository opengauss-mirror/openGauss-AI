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