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
 
 

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import os, sys, re, csv, json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

use_tensorboardX=False
if use_tensorboardX:
    from tensorboardX import SummaryWriter

import networkx as nx

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(o_path+"/algorithms")

import config as cfg

from encoder_reducer import Encoder_reducer, EncoderRNN, AttnReducerRNN, Lang, Scale_log_std
from utils import Query, View, Query_view_pair


trainset_dir=o_path+"dataset/JOB/trainset/"
res_savepath=o_path+"result/job/dqn/2021.4.27-2/"
# res_savepath=o_path+"result/job/dqn/test/"
enc_rdc_model_path=o_path+"result/job/encoder_reducer/2021.4.27-1/"
model_dir=o_path+"result/job/dqn/2021.4.27-2/"

if not os.path.exists(res_savepath):
    os.makedirs(res_savepath)

if use_tensorboardX: writer=SummaryWriter()

MAX_MV_NUM=30
MAX_Q_MV_EDGE_NUM=200

MAX_TURNS=500
MAX_EPISODE=100

torch.cuda.set_device(1)

class Environment(object):
    # budget unit MB
    def __init__(self, budget=100000, max_mv_num=MAX_MV_NUM, max_q_mv_edge_num=MAX_Q_MV_EDGE_NUM, use_real_time=False):
        self.budget = budget
        self.budget_used = 0
        self.max_mv_num = max_mv_num
        self.max_q_mv_edge_num = max_q_mv_edge_num
        self.q_data = dict()
        self.mv_data = dict()
        self.q_mv_data = dict()

        self.workload=None
        self.views=None
        self.estimated_pairs=None


        # self.triples = []
        # self.q_id_lis=[]
        # self.mv_id_lis=[]
        self.selected_mv_bitmap = []
        self.selected_q_mv_edge_bitmap = []
        self.current_edge_index = None
        self.total_delta_time = None
        self.use_real_time=use_real_time
        
        self.encoder_reducer=Encoder_reducer()

        self.query_index=dict()
        self.view_index=dict()
        
    
    def reset(self):
        self.budget_used=0
        self.selected_mv_bitmap=[0]*(len(self.mv_data) if self.mv_data else self.max_mv_num)
        self.selected_q_mv_edge_bitmap=[0]*(len(self.q_mv_data) if self.q_mv_data else self.max_q_mv_edge_num)
        self.current_edge_index = 0
        self.total_delta_time = 0

        state=self.state_cstr()

        return state

    def step(self, action):
        if action not in {0, 1}: raise Exception("illegal action value")
        qv_pair = self.estimated_pairs[self.current_edge_index]
        query = qv_pair.query
        view =  qv_pair.view

        cur_q_index=self.workload.index(query)
        cur_mv_index=self.views.index(view)

        # update edge bitmap
        if action == 0:
            self.selected_q_mv_edge_bitmap[self.current_edge_index] &= 0
        else :
            if self.selected_mv_bitmap[cur_mv_index]==1 or view.size+self.budget_used <= self.budget:
                self.selected_q_mv_edge_bitmap[self.current_edge_index] |= 1
        

        q_candi_q_mv=dict()
        for q_id in self.query_index.keys():
            q_candi_q_mv[q_id]=[]
        
        # update mv bitmap
        self.selected_mv_bitmap=[0]*len(self.views)
        for qv_pair, selected in zip(self.estimated_pairs, self.selected_q_mv_edge_bitmap):
            if selected:
                query = qv_pair.query
                view = qv_pair.view

                mv_index=self.views.index(view)
                self.selected_mv_bitmap[mv_index] |= 1

                q_candi_q_mv[query.id].append(qv_pair)

         
        # print(self.selected_mv_bitmap)

        # update used budget
        self.budget_used=0
        for view, selected in zip(self.views, self.selected_mv_bitmap):
            if selected:
                self.budget_used+=view.size
        
        # cal total delta time
        total_workload_time=0
        new_total_delta_time = 0
        for q_id in self.query_index.keys():
            origin_tim=self.query_index[q_id].execution_time
            total_workload_time+=origin_tim
            if q_candi_q_mv[q_id]:
                new_tim=min([qv_pair.query_view_execution_time for qv_pair in q_candi_q_mv[q_id]])
                # print("choosed, ",origin_tim,new_tim)
            else:
                # print("unchoosed")
                new_tim=origin_tim
            new_total_delta_time+=origin_tim-new_tim
        # print(new_total_delta_time)
        # print("total_workload_time", total_workload_time)
        reward=(new_total_delta_time-self.total_delta_time)
        self.total_delta_time=new_total_delta_time
        if self.budget_used > self.budget:
            reward = -100

        # go to next edge
        self.current_edge_index=(self.current_edge_index+1)%len(self.estimated_pairs)
        next_state=self.state_cstr()

        return [next_state, reward, False, ""]


    # edge [index, triples, mv bitmap, edge bitmap, budget, used budget, q mv delta time, mv size, edge tensor]
    def state_cstr(self):

        qv_pair = self.estimated_pairs[self.current_edge_index]
        q_mv_delta_time = qv_pair.benefit

        state = {
            "edge_index":self.current_edge_index,
            "mv_index":self.views.index(qv_pair.view),
            "edge_list":self.estimated_pairs,
            "mv_bitmap":self.selected_mv_bitmap,
            "edge_bitmap":self.selected_q_mv_edge_bitmap,
            "budget":self.budget,
            "budget_used":self.budget_used,
            "q_mv_delta_time":q_mv_delta_time,
            "mv_size":qv_pair.view.size,
            "edge_tensor":qv_pair.embedding,
            "total_delta_time":self.total_delta_time,
            "q_list":self.workload,
            "mv_list":self.views
        }

        return state


    def close(self):
        self.reset()


    """
    def load_data(self):
        def csv_data_loader(filename, col_id_lis):
            res_data=dict()
            with open(filename) as fp:
                csv_reader=csv.reader(fp)
                for line in csv_reader:
                    tmp_lis=[]
                    for i in col_id_lis:
                        if i==1:
                            tmp_lis.append(json.loads(line[i]))
                        elif i==2 or i==3 or i==4:
                            tmp_lis.append(float(line[i]))
                        else:
                            tmp_lis.append(line[i])
                    res_data[line[0]]=tmp_lis
            return res_data



        # read q: seq, time
        self.q_data=csv_data_loader(trainset_dir+"querydata.csv", [1,3])
        
        # read mv: seq, time, size
        self.mv_data=csv_data_loader(trainset_dir+"mvdata.csv", [1,3,4])
        # rescale size from B to MB
        for data in self.mv_data.values():
            data[2]=data[2]/1e6
  
        # read q-mv time
        # later it will append hidden tensor
        self.q_mv_data=csv_data_loader(trainset_dir+"query-mvdata.csv", [3])

        # read q_mv_q-mv index
        with open(trainset_dir+"query_mv_q_mv_index.csv") as fp:
            csv_reader=csv.reader(fp)
            self.triples=list(csv_reader)

        # sort q mv q_mv index by q id
        rc=re.compile(r'(\d+)[a-z]')
        self.triples.sort(key=lambda x:int(rc.search(x[0]).group(1)))

        q_set=set()
        mv_set=set()
        for triple in self.triples:
            q_set.add(triple[0])
            mv_set.add(triple[1])
        self.q_id_lis=list(q_set)
        self.mv_id_lis=list(mv_set)
        rc_mv=re.compile(r"mv(\d+)")
        self.q_id_lis.sort(key=lambda x:int(rc.search(x).group(1)))
        self.mv_id_lis.sort(key=lambda x:int(rc_mv.search(x).group(1)))

        # load encoder reducer model from file
        self.encoder_reducer.load_model(enc_rdc_model_path)
        # pridict q_tim and q_mv_tim with enc rdc model, if not use real data
        for q_id, mv_id, q_mv_id in self.triples:
            evl_q_tim, evl_q_mv_tim, _, hidden = self.encoder_reducer.evaluate(self.q_data[q_id][0], self.mv_data[mv_id][0])
            if not self.use_real_time:
                self.q_data[q_id][1]=evl_q_tim
                self.q_mv_data[q_mv_id][0]=evl_q_mv_tim
            self.q_mv_data[q_mv_id].append(hidden)
    """



    def build(self, workload, views, estimated_pairs):

        self.workload=workload
        self.views=views
        self.estimated_pairs=estimated_pairs

        # build index
        self.query_index=dict()
        for query in self.workload:
            self.query_index[query.id]=query
        
        self.view_index=dict()
        for view in self.views:
            self.view_index[view.id]=view

        



        

class DQN(object):
    def __init__(self, env):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env=env
        self.Transition=namedtuple("Transition",("state","action","next_state","reward"))

        self.BATCH_SIZE=128
        self.GAMMA=0.999
        self.EPS_START=0.9
        self.EPS_END=0.01
        self.EPS_DECAY=10000
        self.TARGET_UPDATE=200
        self.MEMORY_CAPACITY=100000
        
        self.policy_net=self.Net().to(self.device)
        self.target_net=self.Net().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer=optim.Adam(self.policy_net.parameters())
        
        self.memory=self.memory_create(self.MEMORY_CAPACITY)
        
        self.episodes_optimize_tim=[]
        self.max_optimize_tim=0
        self.max_opti_solution=dict()

        self.epoch=0

    class Net(nn.Module):
        def __init__(self):
            super(DQN.Net,self).__init__()
            self.fc1=nn.Linear(263,128)
            # self.bn1=nn.BatchNorm1d(8)
            self.fc2=nn.Linear(128,64)
            self.fc3=nn.Linear(64,16)
            # self.bn2=nn.BatchNorm1d(16)
            self.fc4=nn.Linear(16,2)
            self.LeakyReLU=nn.LeakyReLU(negative_slope=0.01)

        def forward(self,x):
            # x=F.relu(self.bn1(self.fc1(x)))
            # x=F.relu(self.bn2(self.fc2(x)))
            # relu
            # x=F.relu(self.fc1(x))
            # x=F.relu(self.fc2(x))
            # x=F.relu(self.fc3(x))
            # leakyrelu
            x=self.LeakyReLU(self.fc1(x))
            x=self.LeakyReLU(self.fc2(x))
            x=self.LeakyReLU(self.fc3(x))
            x=self.fc4(x)
            return x
        
    def memory_create(self,capacity):
        outter_class=self
        class Memory(object):
            def __init__(self,capacity):
                self.memory=[]
                self.capacity=capacity
                self.tail=0
            
            def push(self,*args):
                a_piece_of_memory=outter_class.Transition(*args)
                if len(self.memory)<self.capacity:
                    self.memory.append(a_piece_of_memory)
                else:
                    self.memory[self.tail]=a_piece_of_memory
                self.tail=(self.tail+1)%self.capacity
            
            def sample(self,num):
                return random.sample(self.memory,num)
            
            def __len__(self):
                return len(self.memory)
                
        return Memory(capacity)
    
    def select_action(self,state,steps_done):
        threshold=self.EPS_END+(self.EPS_START-self.EPS_END)*math.exp(-1.0*steps_done/self.EPS_DECAY)
        rnd=random.random()
        if state[0,6]<=0:
            return torch.tensor(0, device=self.device, dtype=torch.long).view(1,1)
        if rnd>threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1)
        else:
            return torch.tensor(random.randint(0,1),device=self.device,dtype=torch.long).view(1,1)

    def select_action_evaluate(self,state,steps_done):
        threshold=self.EPS_END+(self.EPS_START-self.EPS_END)*math.exp(-1.0*steps_done/self.EPS_DECAY)
        rnd=random.random()
        if state[0,6]<=0:
            return torch.tensor(0, device=self.device, dtype=torch.long).view(1,1)
        if rnd>threshold:
            with torch.no_grad():
                return self.target_net(state).max(1)[1].view(1,1)
        else:
            return torch.tensor(random.randint(0,1),device=self.device,dtype=torch.long).view(1,1)
    
    def optimize_model(self):
        if len(self.memory)<self.BATCH_SIZE:
            return
        
        batch=self.memory.sample(self.BATCH_SIZE)
        batch=self.Transition(*zip(*batch))
        
        batch_state=torch.cat(batch.state)
        batch_action=torch.cat(batch.action)
        batch_reward=torch.cat(batch.reward)
        next_state_exist_mask=torch.tensor([x is not None for x in batch.next_state], device=self.device, dtype=torch.uint8)
        batch_next_state=torch.cat([x for x in batch.next_state if x is not None])
        
        state_action_values=self.policy_net(batch_state).gather(1,batch_action)
        
        expect_next_state_values=torch.zeros((self.BATCH_SIZE),device=self.device)
        expect_next_state_values[next_state_exist_mask]=self.target_net(batch_next_state).max(1)[0].detach()
        expect_next_state_values=expect_next_state_values*self.GAMMA+batch_reward
        # expect_next_state_values=expect_next_state_values+batch_reward
        
        loss=F.smooth_l1_loss(state_action_values,expect_next_state_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if use_tensorboardX: writer.add_scalar('Train/Loss',loss,self.epoch)
        self.epoch+=1
        # for param in self.policy_net.parameters():
            # param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    def train(self,num_episodes, preprocecssed_selected_mv_bitmap=None, preprocecssed_selected_q_mv_edge_bitmap=None, use_pre_DP=True):
        steps_done=0
        self.epoch=0
        self.episodes_optimize_times=[]

        if use_pre_DP:
            choosed_mv_list=self.DP(self.env.budget,False)
            preprocecssed_selected_mv_bitmap=gen_mv_bitmap_from_mv_list(self.env, choosed_mv_list)
            preprocecssed_selected_q_mv_edge_bitmap=gen_edge_bitmap_from_mv_bitmap(self.env, preprocecssed_selected_mv_bitmap)
            self.env.reset()
        
        
        for episodes_id in range(num_episodes):
            tmp_max_opti_time=0
            env_state=self.env.reset()
            if preprocecssed_selected_mv_bitmap is not None:
                self.env.selected_mv_bitmap=preprocecssed_selected_mv_bitmap.copy()
                self.env.selected_q_mv_edge_bitmap=preprocecssed_selected_q_mv_edge_bitmap.copy()
                env_state,reward,done,info=self.env.step(0)
                if env_state["total_delta_time"]>tmp_max_opti_time:
                    tmp_max_opti_time=env_state["total_delta_time"]
                if env_state["total_delta_time"]>self.max_optimize_tim:
                    self.max_optimize_tim=env_state["total_delta_time"]
                    self.max_opti_solution=env_state
        
            state=self.env_state2tensor(env_state).unsqueeze(0)
            # state=torch.tensor(,device=self.device,dtype=torch.float32).unsqueeze(0)
            

            start=time.time()

            for turns in count():
                # self.env.render()
                action=self.select_action(state,steps_done)
                # print("q action",state[0][6], action.item())
                steps_done+=1
                env_next_state,reward,done,info=self.env.step(action.item())
                next_state=self.env_state2tensor(env_next_state).unsqueeze(0)

                reward=torch.tensor([reward],dtype=torch.float, device=self.device)
                if done: next_state=None
                self.memory.push(state,action,next_state,reward)

                if env_next_state["total_delta_time"]>tmp_max_opti_time:
                    tmp_max_opti_time=env_next_state["total_delta_time"]
                if env_next_state["total_delta_time"]>self.max_optimize_tim:
                    self.max_optimize_tim=env_next_state["total_delta_time"]
                    self.max_opti_solution=env_next_state
                                
                state=next_state
                self.optimize_model()

                # round as episode
                if env_next_state["edge_index"]==0:
                    self.episodes_optimize_times.append(env_next_state["total_delta_time"])
                
                if done or turns>MAX_TURNS:
                    print(tmp_max_opti_time)
                    # self.episodes_optimize_times.append(tmp_max_opti_time)
                    if use_tensorboardX: writer.add_scalar('Train/optimize_time',tmp_max_opti_time,episodes_id)
                    # dynamically draw opti times
                    # self.plot_durations()
                    break
                    
            stop=time.time()
            self.log_latency(start, stop, turns)

                    
            if turns%self.TARGET_UPDATE==0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.env.close()
        # final draw opti times
        # self.plot_durations()
        self.plot_durations2()
        # self.plot_solution()


    def evaluate(self,num_episodes, preprocecssed_selected_mv_bitmap=None, preprocecssed_selected_q_mv_edge_bitmap=None, use_pre_DP=True):
        steps_done=0
        self.epoch=0
        self.episodes_optimize_times=[]

        if use_pre_DP:
            choosed_mv_list=self.DP(self.env.budget,False)
            preprocecssed_selected_mv_bitmap=gen_mv_bitmap_from_mv_list(self.env, choosed_mv_list)
            preprocecssed_selected_q_mv_edge_bitmap=gen_edge_bitmap_from_mv_bitmap(self.env, preprocecssed_selected_mv_bitmap)
            self.env.reset()
        
        
        for episodes_id in range(num_episodes):
            tmp_max_opti_time=0
            env_state=self.env.reset()
            if preprocecssed_selected_mv_bitmap is not None:
                self.env.selected_mv_bitmap=preprocecssed_selected_mv_bitmap.copy()
                self.env.selected_q_mv_edge_bitmap=preprocecssed_selected_q_mv_edge_bitmap.copy()
                env_state,reward,done,info=self.env.step(0)
                if env_state["total_delta_time"]>tmp_max_opti_time:
                    tmp_max_opti_time=env_state["total_delta_time"]
                if env_state["total_delta_time"]>self.max_optimize_tim:
                    self.max_optimize_tim=env_state["total_delta_time"]
                    self.max_opti_solution=env_state
        
            state=self.env_state2tensor(env_state).unsqueeze(0)
            # state=torch.tensor(,device=self.device,dtype=torch.float32).unsqueeze(0)
            

            start=time.time()

            for turns in count():
                # self.env.render()
                action=self.select_action_evaluate(state,steps_done)
                # print("q action",state[0][6], action.item())
                steps_done+=1
                env_next_state,reward,done,info=self.env.step(action.item())
                next_state=self.env_state2tensor(env_next_state).unsqueeze(0)

                reward=torch.tensor([reward],dtype=torch.float, device=self.device)
                if done: next_state=None
                self.memory.push(state,action,next_state,reward)

                if env_next_state["total_delta_time"]>tmp_max_opti_time:
                    tmp_max_opti_time=env_next_state["total_delta_time"]
                if env_next_state["total_delta_time"]>self.max_optimize_tim:
                    self.max_optimize_tim=env_next_state["total_delta_time"]
                    self.max_opti_solution=env_next_state
                                
                state=next_state
                #self.optimize_model()

                # round as episode
                if env_next_state["edge_index"]==0:
                    self.episodes_optimize_times.append(env_next_state["total_delta_time"])
                
                if done or turns>MAX_TURNS:
                    print("Tot benefit", tmp_max_opti_time)
                    # self.episodes_optimize_times.append(tmp_max_opti_time)
                    if use_tensorboardX: writer.add_scalar('Train/optimize_time',tmp_max_opti_time,episodes_id)
                    # dynamically draw opti times
                    # self.plot_durations()
                    break
                    
            stop=time.time()
            # self.log_latency(start, stop, turns)

                    
            # if turns%self.TARGET_UPDATE==0:
            #     self.target_net.load_state_dict(self.policy_net.state_dict())
        self.env.close()
        # final draw opti times
        # self.plot_durations()
        # self.plot_durations2()
        # self.plot_solution()

        

    def env_state2tensor(self, e):
        e_budget=max(e["budget"],1e-6)
        lis=[
            e["edge_bitmap"][e["edge_index"]],
            e["mv_bitmap"][e["mv_index"]],
            e["edge_index"],
            e["mv_index"],
            e["budget_used"]/e_budget,
            e["mv_size"]/e_budget,
            e["q_mv_delta_time"]
        ]
        res_tensor=torch.tensor(lis, dtype=torch.float, device=self.device)
        res_tensor=torch.cat((res_tensor,e["edge_tensor"]), dim=0)
        return res_tensor
                
    def plot_durations(self):
        plt.figure(1,figsize=(15,10))
        plt.clf()
        opti_tim_t = torch.tensor(self.episodes_optimize_times, dtype=torch.float, device=self.device)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Training',fontsize=20)
        plt.xlabel('Episode',fontsize=20)
        plt.ylabel('Benefit',fontsize=20)
        plt.plot(opti_tim_t.numpy())
        # Take 100 episode averages and plot them too
        # avg_num=30
        # if len(opti_tim_t) >= avg_num:
        #     means = opti_tim_t.unfold(0, avg_num, 1).mean(1).view(-1)
        #     means = torch.cat((torch.zeros(avg_num-1), means))
        #     plt.plot(means.numpy())
        
        plt.savefig(res_savepath+"episode-opti_times-budget{0}.pdf".format(self.env.budget))
        with open(res_savepath+"episode-opti_times-budget{0}.json".format(self.env.budget), "w") as fp:
            json.dump(self.episodes_optimize_times, fp)

        # plt.pause(0.001)  # pause a bit so that plots are updated

    def log_latency(self, start, stop, turns):
        with open(res_savepath+"log_latency.json".format(self.env.budget), "w") as fp:
            json.dump([stop-start, turns],fp)


    def plot_durations2(self):
        plt.figure(1,figsize=(15,10))
        plt.clf()
        opti_tim_t = torch.tensor(self.episodes_optimize_times, dtype=torch.float, device=self.device)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title('Training',fontsize=22)
        plt.xlabel('Rounds',fontsize=22)
        plt.ylabel('Benefit',fontsize=22)
        plt.plot(opti_tim_t.numpy())
        # Take 100 episode averages and plot them too
        # avg_num=30
        # if len(opti_tim_t) >= avg_num:
        #     means = opti_tim_t.unfold(0, avg_num, 1).mean(1).view(-1)
        #     means = torch.cat((torch.zeros(avg_num-1), means))
        #     plt.plot(means.numpy())
        
        plt.savefig(res_savepath+"rounds-opti_times-budget{0}.pdf".format(self.env.budget))
        with open(res_savepath+"rounds-opti_times-budget{0}.json".format(self.env.budget), "w") as fp:
            json.dump(self.episodes_optimize_times, fp)

        # plt.pause(0.001)  # pause a bit so that plots are updated

    def plot_solution(self, max_opti_solution=None):
        if max_opti_solution is None: max_opti_solution=self.max_opti_solution
        plt.figure("solution",figsize=(10,35))
        plt.clf()
        plt.title("solution(budget={0})".format(max_opti_solution["budget"]))
        q_list=self.env.q_id_lis
        mv_list=self.env.mv_id_lis
        triples=self.env.triples
        edge_list=[(x[0],x[1]) for x in triples]
        pos_q=[(0,y) for y in range(0,-len(q_list),-1)]
        pos_mv=[(5,y*3) for y in range(0,-len(mv_list),-1)]
        node_pos=dict()
        for pos, name in zip(pos_q,q_list):
            node_pos[name]=pos
        for pos, name in zip(pos_mv,mv_list):
            node_pos[name]=pos
        node_pos_l={}
        node_pos_r={}
        for key,value in node_pos.items():
            node_pos_l[key]=(value[0]-0.25,value[1])
            node_pos_r[key]=(value[0]+0.3,value[1])
        labels_q={}
        labels_mv={}
        for q_id in q_list:labels_q[q_id]=q_id
        for mv_id in mv_list:labels_mv[mv_id]=mv_id
        choosed_mv_list=[]
        for mv_id, choosed in zip(mv_list, max_opti_solution["mv_bitmap"]):
            if choosed:
                choosed_mv_list.append(mv_id)
        choosed_edge_list=[]
        for edge, choosed in zip(edge_list, max_opti_solution["edge_bitmap"]):
            if choosed:
                choosed_edge_list.append(edge)      

        G=nx.Graph()
        G.add_nodes_from(q_list)
        G.add_nodes_from(mv_list)
        G.add_edges_from(edge_list)
        nx.draw_networkx(G, pos=node_pos, node_color='dodgerblue', with_labels=False)
        nx.draw_networkx_nodes(G, pos=node_pos,nodelist=choosed_mv_list, node_color='r')
        nx.draw_networkx_edges(G, pos=node_pos,edgelist=choosed_edge_list, edge_color='r', width=3)
        nx.draw_networkx_labels(G, pos=node_pos_l, labels=labels_q)
        nx.draw_networkx_labels(G, pos=node_pos_r, labels=labels_mv)
        # nx.draw_networkx_labels(G, pos=node_pos_r, nodelist=mv_list)
        plt.savefig(res_savepath+"solution-budget{0}.pdf".format(max_opti_solution["budget"]))
        
        # plt.show()

    def save_model(self,budget):
        torch.save(self.policy_net, res_savepath+"policy_net{0}.pth".format(budget))

    def load_model(self,budget):
        self.policy_net = torch.load(model_dir+"policy_net{0}.pth".format(budget), map_location="cpu")
        self.policy_net.to(device)

        self.target_net=self.Net().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer=optim.Adam(self.policy_net.parameters())

    def get_result_view_list(self):
        mv_bitmap=self.max_opti_solution["mv_bitmap"]
        result_view_list=[]
        for view, selected in zip(self.env.views, mv_bitmap):
            if selected==1:
                result_view_list.append(view)
        
        return result_view_list


    def DP(self, budget=500, save_result=True):
        def get_mv_info():
            mv_info=dict()
            for view in self.env.views: mv_info[view.id]=[view.size,0]
            for qv_pair in self.env.estimated_pairs:
                mv_id=qv_pair.view.id
                q_mv_delta_time = qv_pair.benefit
                if q_mv_delta_time>0:
                    mv_info[mv_id][1]+=q_mv_delta_time

            return mv_info

        def get_total_delta_time(choosed_mv_list):
            env_state=self.env.reset()
            while True:
                mv_id=env_state["mv_list"][env_state["mv_index"]]
                q_mv_delta=env_state["q_mv_delta_time"]
                if mv_id in choosed_mv_list and q_mv_delta>0:
                    action=1
                else:
                    action=0
                self.env_state, _, _, _=self.env.step(action)
                
                if env_state["edge_index"]==0:
                    break
            return env_state["total_delta_time"]
        
        mv_list=[view.id for view in self.env.views]
        # mv_info: mv_id:[mv_size, value]
        mv_info=get_mv_info()

        mv_info_list=[[mv_id, x[0], x[1]] for mv_id,x in mv_info.items()]
        for x in mv_info_list:
            x[1]=int(math.ceil(x[1]))
        budget=int(budget)

        f=[[[0,0] for j in range(budget+1)] for i in range(len(mv_info_list))]
        for j in range(budget+1):
            if j>=mv_info_list[0][1]:
                f[0][j][0]=mv_info_list[0][2]
                f[0][j][1]=1
        for i in range(1,len(mv_info_list)):
            mv_id, size, value=mv_info_list[i]
            for j in range(budget+1):
                f[i][j][0]=f[i-1][j][0]
                if j>=size:
                    if f[i-1][j-size][0]+value>f[i][j][0]:
                        f[i][j][0]=f[i-1][j-size][0]+value
                        f[i][j][1]=1
        
        rest_budget=budget
        choosed_mv_list=[]
        for i in range(len(mv_info_list)-1,-1,-1):
            if f[i][rest_budget][1]==1:
                choosed_mv_list.append(mv_info_list[i][0])
                rest_budget-=mv_info_list[i][1]
        choosed_mv_list=choosed_mv_list[::-1]

        total_delta_time=get_total_delta_time(choosed_mv_list)

        # print("DP budget={0} total_delta={1}".format(budget,total_delta_time))

        return choosed_mv_list


def gen_edge_bitmap_from_mv_bitmap(env, selected_mv_bitmap):
    selected_mv_bitmap=selected_mv_bitmap.copy()
    selected_q_mv_edge_bitmap=[0]*(len(env.estimated_pairs))
    tot=0
    # print("mv_id_lis", env.mv_id_lis)
    # print("selected_mv_bitmap", selected_mv_bitmap)
    for qv_pair in env.estimated_pairs:
        mv_index=env.views.index(qv_pair.view)
        if selected_mv_bitmap[mv_index] == 1:
            selected_q_mv_edge_bitmap[tot]=1
        tot+=1
    return selected_q_mv_edge_bitmap

def gen_mv_bitmap_from_mv_list(env, choosed_mv_list):
    selected_mv_bitmap=[0]*len(env.workload)
    for mv_id in choosed_mv_list:
        mv_index=env.views.index(env.view_index[mv_id])
        selected_mv_bitmap[mv_index] |= 1
    return selected_mv_bitmap
        
if __name__=="__main__":
    args = sys.argv
    if len(args)==2:
        budget=float(args[1])
    else:
        budget=100

    use_pre_DP=False

    env=Environment(budget=budget,use_real_time=True)
    env.load_data()

    # dynamic plot
    # plt.ion()

    # train
    print("use GAMMA0.98, decay 1000, EPS end 0.1, LeakyReLU negative_slope=0.01, use_pre_DP")
    print("begin")
    # train_budget_list=[5, 10, 20, 50, 70, 100, 200, 500, 1000]
    # train_budget_list=train_budget_list+[150, 250, 750, 2000, 3000, 4000, 5000]
    # train_budget_list=list(set(train_budget_list))
    # train_budget_list.sort()
    train_budget_list=[500]
    for budget in train_budget_list:
        print('Budget', budget)
        env.budget=budget
        dqn=DQN(env)

        if use_pre_DP:
            choosed_mv_list=dqn.DP(budget,False)
            selected_mv_bitmap=gen_mv_bitmap_from_mv_list(env, choosed_mv_list)
            selected_q_mv_edge_bitmap=gen_edge_bitmap_from_mv_bitmap(env, selected_mv_bitmap)
        else:
            choosed_mv_list=None
            selected_mv_bitmap=None
            selected_q_mv_edge_bitmap=None

        
        dqn.train(MAX_EPISODE,preprocecssed_selected_mv_bitmap=selected_mv_bitmap, preprocecssed_selected_q_mv_edge_bitmap=selected_q_mv_edge_bitmap)
        
        with open(res_savepath+"budget-max_opti.txt","a") as fp:
            fp.write(str(budget)+" "+str(dqn.max_optimize_tim)+"\n")
            fp.write(" ".join(map(str, dqn.max_opti_solution["mv_bitmap"]))+"\n")
            fp.write(" ".join(map(str, dqn.max_opti_solution["edge_bitmap"]))+"\n")
        dqn.save_model(budget)
        del dqn
    
    # tensorboardx
    # writer.add_graph(dqn.policy_net,(batch_state,))
    # writer.close()

    # test env step
    # state=env.reset()
    # print("first state", state)
    # for i in range(10):
    #     next_state, reward, _, _ =env.step(1)
    #     print("next state", next_state)
    #     print("reward", reward)

    # dynamic plot
    # plt.ioff()
    # plt.show()

    # draw solution from result file
    # budget=0
    # spl_list=[]
    # dqn=DQN(env)
    # with open(res_savepath+"budget-max_opti2.txt") as fp:
    #     for line in fp.readlines():
    #         spl_list.append(line.strip().split(" "))
    # for spl in spl_list:
    #     if len(spl)==2:
    #         budget=int(spl[0])
    #     elif len(spl)<=30:
    #         mv_bitmap=list(map(int, spl))
    #     elif len(spl)>=150:
    #         edge_bitmap=list(map(int, spl))
    #         curr_solution={
    #             "budget":budget,
    #             "mv_bitmap":mv_bitmap,
    #             "edge_bitmap":edge_bitmap
    #         }
    #         if budget==5 or budget==500:
    #             dqn.plot_solution(curr_solution)

    # draw zero solution
    # spl_list=[]
    # dqn=DQN(env)

    # zero_solution={
    #     "budget":0,
    #     "mv_bitmap":[0]*27,
    #     "edge_bitmap":[0]*199
    # }
    # dqn.plot_solution(zero_solution)


        
            

et==5 or budget==500:
    #             dqn.plot_solution(curr_solution)

    # draw zero solution
    # spl_list=[]
    # dqn=DQN(env)

    # zero_solution={
    #     "budget":0,
    #     "mv_bitmap":[0]*27,
    #     "edge_bitmap":[0]*199
    # }
    # dqn.plot_solution(zero_solution)


        
            

