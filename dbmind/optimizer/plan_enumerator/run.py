
from PGUtils import pgrunner
from sqlSample import sqlInfo
import numpy as np
from itertools import count
from math import log
import random
import time
from DQN import DQN,ENV
from TreeLSTM import SPINN
from JOBParser import DB
import copy
import torch
from torch.nn import init
from ImportantConfig import Config

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() and config.usegpu==1 else "cpu")


with open(config.schemaFile, "r") as f:
    createSchema = "".join(f.readlines())

db_info = DB(createSchema)

featureSize = 128

policy_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
target_net = SPINN(n_classes = 1, size = featureSize, n_words = 50,mask_size= len(db_info)*len(db_info),device=device).to(device)
policy_net.load_state_dict(torch.load("CostFinal.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# pgrunner = PGRunner(config.dbName,config.userName,config.password,config.ip,config.port,isCostTraining=False,latencyRecord = True,latencyRecordFile = "Latency.json")

DQN = DQN(policy_net,target_net,db_info,pgrunner,device)

if __name__=='__main__':
    ###No online update now
    print("Enter each query in one line")
    print("---------------------")
    while (1):
        # print(">",end='')
        query = input(">")
        sqlSample = sqlInfo(pgrunner,query,"input")
        # pg_cost = sql.getDPlantecy()
        env = ENV(sqlSample,db_info,pgrunner,device,run_mode = True)
        print("-----------------------------")
        for t in count():
                action_list, chosen_action,all_action = DQN.select_action(env,need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)

                reward, done = env.reward_new()
                if done:
                    for row in reward:
                        print(row)
                    break
        print("-----------------------------")


