from postgres_env import postgresql_env
from RL_brain import DQN
from sys import argv

env = postgresql_env('tpch', 'public', 17 * 1024 * 1024)

dqn = DQN(env, None, None)

model = argv[1]

workload_size = argv[2]

profit_list = []
time_cost = []
index_rec_list = []

# for i in range(1,11):
# with open('../workload/tpc-h/workload_' + str(workload_size) + '/workload_' + str(workload_size) + '_' + str(i) + '.sql', 'r') as work:
for i in range(1, 2):
    with open('tpch-stream.sql', 'r') as work:
        sql_list = work.readlines()

    print("Workload" + str(i) + ":")
    profit, time, index_rec = dqn.recommend_for_test(sql_list, model)
    print("\n")
    profit_list.append(profit)
    time_cost.append(time)
    index_rec_list.append(index_rec)

for i in range(0, len(time_cost)):
    print(str(i + 1) + ',' + str(profit_list[i]) + ',' + str(time_cost[i]) + ', ' + index_rec_list[i])
