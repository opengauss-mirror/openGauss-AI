from postgres_env import postgresql_env
import torch
from RL_brain import DQN
from sys import argv

test_sql = "select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority \
            from customer, orders, lineitem\
            where c_mktsegment = 'AUTOMOBILE' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-22' and l_shipdate > date '1995-03-22' \
            group by l_orderkey, o_orderdate, o_shippriority \
            order by revenue desc, o_orderdate;"

size_budget_mb = 17
dbname = 'tpch'

with open(argv[1], 'r') as work:
    sql_list = work.readlines()

env = postgresql_env(dbname, 'public', size_budget_mb * 1024 * 1024)
env.set_workload(sql_list)


res_path = argv[2]
res_file_name = argv[3]


dqn = DQN(env, traincsv=res_path + '/train_' + res_file_name +
          '.csv', reccsv=res_path + '/rec_' + res_file_name + '.csv')
dqn.train_workload()
torch.save(dqn.eval_net.state_dict(), res_path + '/eval_' + res_file_name + '.pkl')
torch.save(dqn.target_net.state_dict(), res_path + '/target_' + res_file_name + '.pkl')
