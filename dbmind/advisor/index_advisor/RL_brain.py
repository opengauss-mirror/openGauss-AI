import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import copy
import csv


class Net(nn.Module):
    def __init__(self, input_num):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_num, 2 ** 10)
        self.fc1.weight.data.normal_(0, 0.01)

        self.fc2 = nn.Linear(2 ** 10, 2 ** 10)
        self.fc2.weight.data.normal_(0, 0.01)

        self.out = nn.Linear(2 ** 10, input_num)
        self.out.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):
    def __init__(self, env, traincsv, reccsv, learning_rate=0.1, reward_decay=0.9, e_greedy=1, update_freq=50, mem_cap=10000,
                 batch_size=32):
        self.env = env
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.update_freq = update_freq

        self.mem_counter = 0
        self.memory_capacity = mem_cap
        self.mem = np.zeros((self.memory_capacity, self.env.get_input_num() * 2 + 2))

        self.eval_net = Net(self.env.get_input_num())
        self.target_net = Net(self.env.get_input_num())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.train_csv = traincsv
        self.rec_csv = reccsv

    def choose_action(self):
        if np.random.uniform() > self.epsilon:
            x = []
            for table in self.env.cur_state:
                x.extend(table)

            x = torch.unsqueeze(torch.Tensor(x), 0)
            actions_value_tmp = self.eval_net.forward(x)
            actions_value = actions_value_tmp[0, self.env.action_acum_pos]

            #max_action_val = torch.max(actions_value)[0]
            max_action_val = torch.max(actions_value)
            y = torch.ones(1, len(actions_value)) * -10000
            max_actions = torch.where(actions_value == max_action_val, actions_value, y)
            candidate_action = []
            for i in range(len(actions_value)):
                if max_actions[0, i] != -10000:
                    candidate_action.append(i)

            if len(candidate_action) > 1:
                action_num = candidate_action[np.random.randint(len(candidate_action))]
                action = self.env.action_space[action_num]
            else:
                action = self.env.action_space[candidate_action[0]]
        else:
            action_num = np.random.randint(len(self.env.action_space))
            action = [self.env.action_space[action_num][0], self.env.action_space[action_num][1]]
        return action

    def store_transition(self, s, a, r, s_):
        action_pos = self.env.original_action_acum_pos[self.env.original_action_space.index(a)]
        s_seq = []
        for table in s:
            s_seq.extend(table)
        s__seq = []
        for table in s_:
            s__seq.extend(table)
        trans = np.hstack((s_seq, [action_pos, r], s__seq))
        index = self.mem_counter % self.memory_capacity
        self.mem[index, :] = trans
        self.mem_counter += 1

    def learn(self):
        sample_index = np.random.choice(min(self.mem_counter, self.memory_capacity), self.batch_size)
        b_memory = self.mem[sample_index, :]
        state_len = self.env.get_input_num()
        b_s = torch.tensor(b_memory[:, :state_len], dtype=torch.float)
        b_a = torch.tensor(b_memory[:, state_len: state_len + 1], dtype=torch.long)
        b_r = torch.tensor(b_memory[:, state_len + 1: state_len + 2], dtype=torch.float)
        b_s_ = torch.tensor(b_memory[:, -state_len:], dtype=torch.float)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

    def train_workload(self):
        episode_num = 1024
        const_a = math.pow(0.001, 1 / episode_num)
        for episode in range(episode_num):
            time_start = time.time()
            self.env.reset()

            cur_state = copy.deepcopy(self.env.cur_state)

            while True:
                action = self.choose_action()
                state_next, reward, done = self.env.step(action)

                if done:
                    time_end = time.time()

                    index_plan = ""
                    for table_pos, col_pos in self.env.cur_virtual_index:
                        index_plan += self.env.db_table_list[table_pos] + '.' + \
                            self.env.db_column_list[table_pos][col_pos] + ' '
                    index_plan.strip()

                    print("Epsilon: " + str(self.epsilon))
                    print("Indexes: " + str(episode))
                    print(index_plan)
                    print("Size: " + str(self.env.cur_size_use))
                    print("Benefit: " + str(max(0, self.env.workload_raw_cost - self.env.cur_cost)))
                    print("Time: " + str(time_end - time_start))

                    csv_file = open(self.train_csv, 'a+', newline='')
                    csv_write = csv.writer(csv_file, dialect='excel')
                    csv_write.writerow([episode, self.epsilon, index_plan, self.env.cur_size_use, max(
                        0, self.env.workload_raw_cost - self.env.cur_cost), time_end - time_start])
                    csv_file.close()
                    break

                self.store_transition(cur_state, action, reward, state_next)
                if self.mem_counter > self.batch_size:
                    self.learn()

                cur_state = state_next

            if episode % 10 == 0 and episode > 0:
                self.recommend()

            if self.epsilon >= episode_num - 200:
                self.epsilon = 0.0001
            else:
                self.epsilon = math.pow(const_a, episode)

    def recommend(self):
        time_start = time.time()
        self.env.reset()

        cur_state = copy.deepcopy(self.env.cur_state)

        while True:
            x = []
            for table in self.env.cur_state:
                x.extend(table)

            x = torch.unsqueeze(torch.Tensor(x), 0)
            actions_value_tmp = self.eval_net.forward(x)
            actions_value = actions_value_tmp[0, self.env.action_acum_pos]
            max_action_val = torch.max(actions_value)
            y = torch.ones(1, len(actions_value)) * -10000
            max_actions = torch.where(actions_value == max_action_val, actions_value, y)
            candidate_action = []
            for i in range(len(actions_value)):
                if max_actions[0, i] != -10000:
                    candidate_action.append(i)

            if len(candidate_action) > 1:
                action_num = candidate_action[np.random.randint(len(self.env.action_space))]
                action = self.env.action_space[action_num]
            else:
                action = self.env.action_space[candidate_action[0]]

            state_next, reward, done = self.env.step(action)

            if cur_state == state_next:
                time_end = time.time()

                index_rec = ""
                for table_pos, col_pos in self.env.cur_virtual_index:
                    index_rec += self.env.db_table_list[table_pos] + '.' + \
                        self.env.db_column_list[table_pos][col_pos] + ' '

                csvr = open(self.rec_csv, 'a+', newline='')
                csvr_write = csv.writer(csvr, dialect='excel')
                csvr_write.writerow([index_rec, self.env.cur_size_use, max(
                    0, self.env.workload_raw_cost - self.env.cur_cost), time_end - time_start])
                csvr.close()

                print("-------------------------------------Indexes----------------------------------------")
                print(index_rec)
                print("Size: " + str(self.env.cur_size_use))
                print("Benefit: " + str(max(0, self.env.workload_raw_cost - self.env.cur_cost)))
                print("Time: " + str(time_end - time_start))
                print("-------------------------------------Finished---------------------------------------")
                break

            cur_state = copy.deepcopy(state_next)

    def recommend_for_test(self, workload, target_model):
        self.env.set_workload(workload)
        self.target_net.load_state_dict(torch.load(target_model))

        time_cost = 0
        time_start = time.time()
        self.env.reset()

        cur_state = copy.deepcopy(self.env.cur_state)

        while True:
            x = []
            for table in self.env.cur_state:
                x.extend(table)

            x = torch.unsqueeze(torch.Tensor(x), 0)
            actions_value_tmp = self.target_net.forward(x)
            actions_value = actions_value_tmp[0, self.env.action_acum_pos]
            max_action_val = torch.max(actions_value)
            y = torch.ones(1, len(actions_value)) * -10000
            max_actions = torch.where(actions_value == max_action_val, actions_value, y)
            candidate_action = []
            for i in range(len(actions_value)):
                if max_actions[0, i] != -10000:
                    candidate_action.append(i)

            if len(candidate_action) > 1:
                action_num = candidate_action[np.random.randint(len(candidate_action))]
                action = self.env.action_space[action_num]
            else:
                action = self.env.action_space[candidate_action[0]]

            state_next, reward, done = self.env.step(action)

            if cur_state == state_next:
                time_end = time.time()

                index_rec = ""
                for table_pos, col_pos in self.env.cur_virtual_index:
                    index_rec += self.env.db_table_list[table_pos] + '.' + self.env.db_column_list[table_pos][
                        col_pos] + ' '

                print("-------------------------------------Indexes----------------------------------------")
                print(index_rec)
                print("Indexes: " + str(self.env.cur_size_use))
                print("Benefit: " + str(max(0, self.env.workload_raw_cost - self.env.cur_cost)))
                print("Time: " + str(time_end - time_start))
                print("-------------------------------------Finished---------------------------------------")
                time_cost = time_end - time_start
                return max(0, self.env.workload_raw_cost - self.env.cur_cost), time_cost, index_rec

            cur_state = copy.deepcopy(state_next)
