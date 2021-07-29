#!/usr/bin/python
from __future__ import division, print_function, unicode_literals

import csv
import json
import os
import random
import re
import string
import sys
import unicodedata
from io import open
import time
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

o_path = os.getcwd()
o_path=o_path+"/./" # single run on server
sys.path.append(o_path)

# res_savepath=o_path+"result/job/encoder_reducer/eval_all/"
res_savepath=o_path+"result/job/encoder_reducer/2021.4.27-2/"
enc_rdc_model_path=o_path+"result/job/encoder_reducer/2021.4.27-1/"

pretrain_path=o_path+"result/job/encoder_pretrain/2021.4.27-1/"

if not os.path.exists(res_savepath):
    os.makedirs(res_savepath)

# torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

MAX_LENGTH=206
# SOS_token = 0
# EOS_token = 1

class Scale_log_std(object):
    def __init__(self):
        self.mean=None
        self.std=None
        
    def fit(self, lis):
        arr=np.log(lis)
        self.mean=np.mean(arr)
        self.std=np.std(arr)
    
    def transform(self, num):
        x=num
        x=(math.log(x)-self.mean)/self.std
        return x
    
    def detransform(self, num):
        x=num
        x=math.exp(x*self.std+self.mean)
        return x

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0 

    # origin addSentense
    def add_plan_seq(self, seq):
        tags_need_embedding={'identifier','keyword'}
        for content, tag in seq:
            if tag in tags_need_embedding:
                self.add_word(content)

    # origin addWord
    def add_word(self, word):
        if word not in self.word2index:
            # give word an increasing id
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1  

class EncoderRNN(nn.Module):
    def __init__(self, input_size, gru_input_size, gru_hidden_size, embedding):
        super(EncoderRNN, self).__init__()
        self.gru_hidden_size = gru_hidden_size
        self.gru_input_size = gru_input_size

        self.embedding = embedding

        self.conv1=nn.Conv1d(1,8,3)
        self.conv2=nn.Conv1d(8,16,3)
        self.fc=nn.Linear(128,self.gru_input_size)


        self.gru = nn.GRU(gru_input_size, gru_hidden_size)

        self.fc2=nn.Linear(self.gru_hidden_size, 64)
        self.out = nn.Linear(64, 1)        # output actual runtime, a number

    def forward(self, input, hidden):
        input, tag = input
        if tag=="identifier" or tag=="keyword":
            # embedding input=index:int
            input=torch.tensor([[input]], dtype=torch.long, device=device)
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
        elif tag=="number":
            # number input=number:float
            output=self.float2bin_tensor(input, vec_size=self.gru_input_size).view(1,1,-1)
        elif tag=="string":
            # string input=string:string
            input=self.string2tensor(input).view(1,1,-1)
            input=F.max_pool1d(F.relu(self.conv1(input)),2)
            input=F.max_pool1d(F.relu(self.conv2(input)),2)
            input=input.view(1,-1)
            input=F.relu(self.fc(input)).view(1,1,-1)
            output=input
        else:
            raise Exception("not seen tag")
        
        output, hidden = self.gru(output, hidden)

        tim_output = F.relu(self.fc2(output[0]))
        tim_output = self.out(tim_output)

        return output, hidden, tim_output

    def initHidden(self):
        return torch.zeros(1, 1, self.gru_hidden_size, device=device)

    def string2tensor(self, ss, padding_to_size=38):
        lis=[0]*padding_to_size
        lis[:len(ss)]=[ord(c) for c in ss]
        tensor=torch.tensor(lis, dtype=torch.float, device=device)
        tensor=tensor/128
        return tensor


    def float2bin_tensor(self, num, vec_size=128):
        eps=1e-8
        num=float(num)
        deci, intg=math.modf(num)
        intg=int(intg)
        half_size=vec_size//2
        intg_lis=[0]*half_size
        deci_lis=[0]*half_size
        for i in range(0,half_size):
            intg_lis[i]=intg%2
            intg=intg//2
            if intg<=0:break
        for i in range(half_size-1,0,-1):
            deci=deci*2
            deci, deci_lis[i]=math.modf(deci)
            deci_lis[i]=int(deci_lis[i])
            if abs(deci)<eps:break
        res_lis=deci_lis+intg_lis
        # for display
        # intg_lis_reverse=intg_lis[::-1]
        # deci_lis_reverse=deci_lis[::-1]
        # print("".join(map(str,intg_lis_reverse))+"."+"".join(map(str,deci_lis_reverse)))
        tensor=torch.tensor(res_lis, dtype=torch.float, device=device)
        return tensor

class AttnReducerRNN(nn.Module):
    def __init__(self, input_size, gru_input_size, gru_hidden_size, embedding, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnReducerRNN, self).__init__()
        self.gru_input_size=gru_input_size
        self.gru_hidden_size=gru_hidden_size
        self.input_size=input_size
        self.max_length=max_length
        self.dropout_p=dropout_p

        self.embedding = embedding
        self.attn = nn.Linear(self.gru_input_size+self.gru_hidden_size, self.gru_hidden_size)
        self.attn_combine = nn.Linear(self.gru_input_size+self.gru_hidden_size, self.gru_input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size)
        self.fc2=nn.Linear(self.gru_hidden_size, 64)
        self.out = nn.Linear(64, 1)        # output actual runtime, a number
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

        self.conv1=nn.Conv1d(1,8,3)
        self.conv2=nn.Conv1d(8,16,3)
        self.fc=nn.Linear(128,self.gru_input_size)

    def forward(self, input, hidden, encoder_outputs):
        input, tag = input
        if tag=="identifier" or tag=="keyword":
            # embedding input=index:int
            input=torch.tensor([[input]], dtype=torch.long, device=device)
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)
        elif tag=="number":
            # number input=number:float
            embedded=self.float2bin_tensor(input, vec_size=self.gru_input_size).view(1,1,-1)
        elif tag=="string":
            # string input=string:string
            input=self.string2tensor(input).view(1,1,-1)
            input=F.max_pool1d(F.relu(self.conv1(input)),2)
            input=F.max_pool1d(F.relu(self.conv2(input)),2)
            input=input.view(1,-1)
            input=F.relu(self.fc(input)).view(1,1,-1)
            embedded=input
        else:
            raise Exception("not seen tag")

        attn_weights = self.attn(torch.cat((embedded[0], hidden[0]), 1))
        # 1. dot similarity
        # attn_weights = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0).transpose(1,2))[0]
        # 2. cos similarity
        attn_weights = self.cos_sim(attn_weights, encoder_outputs).unsqueeze(0)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.relu(self.fc2(output[0]))
        output = self.out(output)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.gru_hidden_size, device=device)

    def string2tensor(self, ss, padding_to_size=38):
        lis=[0]*padding_to_size
        lis[:len(ss)]=[ord(c) for c in ss]
        tensor=torch.tensor(lis, dtype=torch.float, device=device)
        tensor=tensor/128
        return tensor

    def float2bin_tensor(self, num, vec_size=128):
        eps=1e-8
        num=float(num)
        deci, intg=math.modf(num)
        intg=int(intg)
        half_size=vec_size//2
        intg_lis=[0]*half_size
        deci_lis=[0]*half_size
        for i in range(0,half_size):
            intg_lis[i]=intg%2
            intg=intg//2
            if intg<=0:break
        for i in range(half_size-1,0,-1):
            deci=deci*2
            deci, deci_lis[i]=math.modf(deci)
            deci_lis[i]=int(deci_lis[i])
            if abs(deci)<eps:break
        res_lis=deci_lis+intg_lis
        # for display
        # intg_lis_reverse=intg_lis[::-1]
        # deci_lis_reverse=deci_lis[::-1]
        # print("".join(map(str,intg_lis_reverse))+"."+"".join(map(str,deci_lis_reverse)))
        tensor=torch.tensor(res_lis, dtype=torch.float, device=device)
        return tensor


class Encoder_reducer(object):
    def __init__(self):
        self.q_data=dict()
        self.mv_data=dict()
        self.q_mv_data=dict()
        self.triples=[]
        self.triples_vali=[]
        self.all_triples=[]
        self.dictionary=None
        self.gru_input_size=128
        self.gru_hidden_size=256
        self.embedding = None
        self.encoder = None
        self.reducer = None
        self.max_length=MAX_LENGTH
        self.scale=Scale_log_std()
        self.validate_size=0.1


    # Word contains SQL plan keyword, table identifier, column identifier
      

    # origin readLangs
    def read_langs(self, trainset_dir=None):
        def csv_data_loader(filename, col_id_lis):
            res_data=dict()
            with open(filename) as fp:
                csv_reader=csv.reader(fp)
                for line in csv_reader:
                    tmp_lis=[]
                    for i in col_id_lis:
                        if i==1:
                            tmp_lis.append(json.loads(line[i]))
                        elif i==2 or i==3:
                            tmp_lis.append(float(line[i]))
                        else:
                            tmp_lis.append(line[i])
                    res_data[line[0]]=tmp_lis
            return res_data
        
        if not trainset_dir:
            trainset_dir=o_path+"dataset/JOB/trainset/"
        # print("Reading data...from "+trainset_dir)

        # read q
        self.q_data=csv_data_loader(trainset_dir+"querydata.csv", [1,3])
        
        # read mv
        self.mv_data=csv_data_loader(trainset_dir+"mvdata.csv", [1,3])
        
        # read q-mv
        self.q_mv_data=csv_data_loader(trainset_dir+"query-mvdata.csv", [3])

        # rescale
        lis_q=[data[1] for data in self.q_data.values()]
        lis_q_mv=[data[0] for data in self.q_mv_data.values()]
        lis_all=lis_q+lis_q_mv
        self.scale.fit(lis_all)
        for data in self.q_data.values():
            data[1]=self.scale.transform(data[1])
        for data in self.q_mv_data.values():
            data[0]=self.scale.transform(data[0])


        # read q_mv_q-mv index
        with open(trainset_dir+"query_mv_q_mv_index.csv") as fp:
            csv_reader=csv.reader(fp)
            self.all_triples=list(csv_reader)



        
        return self.all_triples

    def split_train_validate(self, validate_size=None):
        def random_train_vali_index(lis, vali_rate):
            n=len(lis)
            indexes=list(range(n))
            random.shuffle(indexes)
            train_num=int(n*(1-vali_rate))
            vali_num=n-train_num
            part_num=(n+vali_num-1)//vali_num
            train_vali_idx_lis=[]
            for i in range(part_num):
                train_index=indexes[:i*vali_num]+indexes[(i+1)*vali_num:]
                vali_index=indexes[i*vali_num:(i+1)*vali_num]
                train_vali_idx_lis.append([train_index, vali_index])

            return train_vali_idx_lis

        if validate_size is not None: self.validate_size=validate_size

        # split train validate
        train_vali_idx_lis = random_train_vali_index(self.all_triples, self.validate_size)

        return train_vali_idx_lis


    # origin prepareData
    def prepare_data(self, trainset_dir=None):
        self.read_langs(trainset_dir)
        # print("Read %s q mv q-mv triples" % len(self.all_triples))

        # print("Counting words...")
        self.dictionary=Lang("job")
        self.max_length = 0
        for id, data in self.q_data.items():
            self.dictionary.add_plan_seq(data[0])
            if len(data[0])>self.max_length: self.max_length=len(data[0])
        for id, data in self.mv_data.items():
            self.dictionary.add_plan_seq(data[0])
        
        # print("Counted words:")
        # print(self.dictionary.name, self.dictionary.n_words)

        return self.dictionary

    def re_dictionary(self):
        for id, data in self.q_data.items():
            self.dictionary.add_plan_seq(data[0])
            if len(data[0])>self.max_length: self.max_length=len(data[0])
        for id, data in self.mv_data.items():
            self.dictionary.add_plan_seq(data[0])
        
        # print("redic Counted words:")
        # print(self.dictionary.name, self.dictionary.n_words)        

    # teacher_forcing_ratio = 0.5
    def index_from_word(self, word):
        return self.dictionary.word2index[word] if word in self.dictionary.word2index else 1


    # def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    def train(self, q_seq, mv_seq, q_tim, q_mv_tim, encoder, reducer, encoder_optimizer, reducer_optimizer, criterion, max_length=MAX_LENGTH):
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        reducer_optimizer.zero_grad()

        q_length = len(q_seq)
        mv_length = len(mv_seq)

        encoder_outputs = torch.zeros(max_length, encoder.gru_hidden_size, device=device)

        loss = 0

        for ei in range(q_length):
            element=q_seq[ei]
            if element[1]=="identifier" or element[1]=="keyword":
                element=(self.index_from_word(element[0]), element[1])
            encoder_output, encoder_hidden, tim_output = encoder(element, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        q_target_tim=torch.tensor([[q_tim]], dtype=torch.float, device=device)
        loss += criterion(tim_output, q_target_tim)
        # decoder_input = torch.tensor([[SOS_token]], device=device)

        reducer_hidden = encoder_hidden

        for di in range(mv_length):
            element=mv_seq[di]
            if element[1]=="identifier" or element[1]=="keyword":
                element=(self.index_from_word(element[0]), element[1])
            reducer_output, reducer_hidden, reducer_attention = reducer(element, reducer_hidden, encoder_outputs)

        
        
        q_mv_target_tim=torch.tensor([[q_mv_tim]], dtype=torch.float, device=device)
        # print(reducer_output, q_mv_target_tim)
        loss += criterion(reducer_output, q_mv_target_tim)

        loss.backward()

        encoder_optimizer.step()
        reducer_optimizer.step()

        return loss.item() / 2 

    # validate
    def validate(self, q_seq, mv_seq, q_tim, q_mv_tim, encoder, reducer,  criterion, max_length=MAX_LENGTH):
        with torch.no_grad():
            encoder_hidden = encoder.initHidden()

            q_length = len(q_seq)
            mv_length = len(mv_seq)

            encoder_outputs = torch.zeros(max_length, encoder.gru_hidden_size, device=device)

            loss = 0

            for ei in range(q_length):
                element=q_seq[ei]
                if element[1]=="identifier" or element[1]=="keyword":
                    element=(self.index_from_word(element[0]), element[1])
                encoder_output, encoder_hidden, tim_output = encoder(element, encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            q_target_tim=torch.tensor([[q_tim]], dtype=torch.float, device=device)
            loss += criterion(tim_output, q_target_tim)
            # decoder_input = torch.tensor([[SOS_token]], device=device)

            reducer_hidden = encoder_hidden

            for di in range(mv_length):
                element=mv_seq[di]
                if element[1]=="identifier" or element[1]=="keyword":
                    element=(self.index_from_word(element[0]), element[1])
                reducer_output, reducer_hidden, reducer_attention = reducer(element, reducer_hidden, encoder_outputs)

            
            
            q_mv_target_tim=torch.tensor([[q_mv_tim]], dtype=torch.float, device=device)
            # print(reducer_output, q_mv_target_tim)
            loss += criterion(reducer_output, q_mv_target_tim)

            return loss.item() / 2


    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    #
    def trainIters(self, encoder, reducer, n_iters, print_every=1000, plot_every=100, enc_learning_rate=0.001, rdc_learning_rate=0.001, log_id=""):
        start = time.time()
        plot_losses = []
        plot_validate_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        last_validate_loss = -1

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=enc_learning_rate)
        reducer_optimizer = optim.Adam(reducer.parameters(), lr=rdc_learning_rate)
        training_triples = [random.choice(self.triples) for i in range(n_iters)]
        criterion = nn.SmoothL1Loss()

        

        for iter in range(1, n_iters + 1):
            training_triple = training_triples[iter - 1]
            q_seq, q_tim = self.q_data[training_triple[0]]
            mv_seq = self.mv_data[training_triple[1]][0]
            q_mv_tim = self.q_mv_data[training_triple[2]][0]

            loss = self.train(q_seq, mv_seq, q_tim, q_mv_tim, encoder, reducer, encoder_optimizer, reducer_optimizer, criterion, self.max_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

                if len(self.triples_vali)>0:
                    vali_loss = 0
                    for triple_vali in self.triples_vali:
                        q_seq, q_tim = self.q_data[triple_vali[0]]
                        mv_seq = self.mv_data[triple_vali[1]][0]
                        q_mv_tim = self.q_mv_data[triple_vali[2]][0]
                        vali_loss += self.validate(q_seq, mv_seq, q_tim, q_mv_tim, encoder, reducer, criterion)
                    vali_loss = vali_loss / len(self.triples_vali)
                    plot_validate_losses.append(vali_loss)
                    last_validate_loss = vali_loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) trainLoss: %.4f, valiLoss: %.4f' % (self.timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, last_validate_loss))
                sys.stdout.flush()


        self.showPlot(plot_losses, "Train Loss"+log_id)
        self.showPlot(plot_validate_losses, "Validate Loss"+log_id)

    def showPlot(self, points, title):
        fig, ax = plt.subplots(figsize=(15,10))
        # this locator puts ticks at regular intervals
        # loc = ticker.MultipleLocator(base=0.2)
        # ax.yaxis.set_major_locator(loc)
        ax.set_title(title)
        plt.plot(points)
        plt.savefig(res_savepath+title.replace(" ","_")+".png")

        plt.close('all')

    def evaluate(self, q_seq, mv_seq, encoder = None, reducer = None, max_length=MAX_LENGTH):
        if encoder is None: encoder=self.encoder
        if reducer is None: reducer=self.reducer
        with torch.no_grad():
            q_length = len(q_seq)
            mv_length = len(mv_seq)
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(max_length, encoder.gru_hidden_size, device=device)

            for ei in range(q_length):
                element=q_seq[ei]
                if element[1]=="identifier" or element[1]=="keyword":
                    element=(self.index_from_word(element[0]), element[1])
                encoder_output, encoder_hidden, tim_output = encoder(element, encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]            
            evl_q_tim=tim_output.item()

            reducer_hidden = encoder_hidden

            reducer_attentions = torch.zeros(max_length, max_length)

            for di in range(mv_length):
                element=mv_seq[di]
                if element[1]=="identifier" or element[1]=="keyword":
                    element=(self.index_from_word(element[0]), element[1])
                reducer_output, reducer_hidden, reducer_attention = reducer(element, reducer_hidden, encoder_outputs)
                reducer_attentions[di] = reducer_attention.data
            evl_q_mv_tim=reducer_output.item()

            evl_q_tim=self.scale.detransform(evl_q_tim)
            evl_q_mv_tim=self.scale.detransform(evl_q_mv_tim)

            return evl_q_tim, evl_q_mv_tim, reducer_attentions[:mv_length,:q_length], reducer_hidden[0,0]


    def evaluateRandomly(self, encoder=None, reducer=None, triples=None, n=10):
        if triples is not None and not triples:
            print("None\n")
            return
        if triples is None:
            triples=self.triples   
        for i in range(n):
            triple = random.choice(triples)
            q_seq, q_tim = self.q_data[triple[0]]
            mv_seq, mv_tim = self.mv_data[triple[1]]
            q_mv_tim = self.q_mv_data[triple[2]][0]
            print('> q = {0}, mv = {1}'.format(triple[0],triple[1]))
            q_tim=self.scale.detransform(q_tim)
            q_mv_tim=self.scale.detransform(q_mv_tim)
            print('= q_time = {0}, mv_time = {1}, q_mv_time = {2}'.format(q_tim, mv_tim, q_mv_tim))
            evl_q_tim, evl_q_mv_tim, attentions, _ = self.evaluate(q_seq, mv_seq, encoder, reducer)
            print('< q_time = {0}, q_mv_time = {1}'.format(evl_q_tim, evl_q_mv_tim))
            print('')

    def evaluateAll(self, encoder=None, reducer=None, triples=None, n=10, log_id=""):
        if triples is not None and not triples:
            print("None\n")
            return
        if triples is None:
            triples=self.triples  

        res_lis=[] 
        for triple in triples:
            q_seq, q_tim = self.q_data[triple[0]]
            mv_seq, mv_tim = self.mv_data[triple[1]]
            q_mv_tim = self.q_mv_data[triple[2]][0]
            print('> q = {0}, mv = {1}'.format(triple[0],triple[1]))
            q_tim=self.scale.detransform(q_tim)
            q_mv_tim=self.scale.detransform(q_mv_tim)
            print('= q_time = {0}, mv_time = {1}, q_mv_time = {2}'.format(q_tim, mv_tim, q_mv_tim))
            evl_q_tim, evl_q_mv_tim, attentions, _ = self.evaluate(q_seq, mv_seq, encoder, reducer)
            print('< q_time = {0}, q_mv_time = {1}'.format(evl_q_tim, evl_q_mv_tim))
            res_lis.append([triple[2], q_tim, q_mv_tim, evl_q_tim, evl_q_mv_tim])
        with open(res_savepath+"eval_all{0}.csv".format(log_id),"w",newline='') as fp:
            csv_writer=csv.writer(fp)
            csv_writer.writerows(res_lis)

            

    def showAttention(self, q_id, mv_id, q_seq, mv_seq, attentions):
        # Set up figure with colorbar
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + [x[0] for x in q_seq], rotation=90)
        ax.set_yticklabels([''] + [x[0] for x in mv_seq])

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.savefig(res_savepath+"{0}-{1}.pdf".format(q_id,mv_id))

        # plt.show()
        plt.close('all')


    def evaluateAndShowAttention(self, q_id, mv_id, encoder=None, reducer=None):
        q_seq, q_tim = self.q_data[q_id]
        mv_seq, mv_tim = self.mv_data[mv_id]
        evl_q_tim, evl_q_mv_tim, attentions, _ = self.evaluate(q_seq, mv_seq, encoder, reducer)
        print('input: q = {0}, mv = {1}'.format(q_id, mv_id))
        self.showAttention(q_id, mv_id, q_seq, mv_seq, attentions)

    def save_model(self):
        torch.save(self.encoder, res_savepath+"encoder.pth")
        torch.save(self.reducer, res_savepath+"reducer.pth")
        with open(res_savepath+"dic.pk", "wb") as fp:
            pickle.dump(self.dictionary, fp)
        with open(res_savepath+"scale.pk", "wb") as fp:
            pickle.dump(self.scale, fp)

    def load_model(self, model_dir):
        self.encoder = torch.load(model_dir+"encoder.pth", map_location="cpu")
        self.encoder.to(device)
        self.reducer = torch.load(model_dir+"reducer.pth", map_location="cpu")
        self.reducer.to(device)
        self.encoder.eval()
        self.reducer.eval()
        with open(model_dir+"dic.pk", "rb") as fp:
            self.dictionary=pickle.load(fp)
        with open(model_dir+"scale.pk", "rb") as fp:
            self.scale=pickle.load(fp)

    def load_pretrain_encoder_model(self, model_dir):
        self.encoder = torch.load(model_dir+"encoder.pth", map_location="cpu")
        self.encoder.to(device)
        self.encoder.train()
        with open(model_dir+"dic.pk", "rb") as fp:
            self.dictionary=pickle.load(fp)
           
    def do(self):
        self.prepare_data()
        # print("dictionary.n_words = ",self.dictionary.n_words)
        # print("example triple ",random.choice(self.all_triples))

        train_vali_idx_lis=self.split_train_validate(0.1)
        part_num=len(train_vali_idx_lis)

        folders=10
        for turn in range(folders):
            tmp_triples=self.all_triples[:]
            train_index=train_vali_idx_lis[turn][0]
            vali_index=train_vali_idx_lis[turn][1]
            self.triples = [tmp_triples[x] for x in train_index]
            self.triples_vali = [tmp_triples[x] for x in vali_index]

            self.gru_hidden_size = 256
            self.embedding = nn.Embedding(self.dictionary.n_words, self.gru_input_size)
            self.encoder = EncoderRNN(self.dictionary.n_words, self.gru_input_size, self.gru_hidden_size, self.embedding).to(device)
            self.reducer = AttnReducerRNN(self.dictionary.n_words, self.gru_input_size, self.gru_hidden_size, self.embedding, dropout_p=0.1).to(device)

            self.trainIters(self.encoder, self.reducer, 40000, print_every=1000, log_id=str(turn))

            self.encoder.eval()
            self.reducer.eval()

            print("evaluate vali set")
            self.evaluateAll(triples=self.triples_vali, log_id=str(turn))

            print("evaluate train randomly")
            self.evaluateRandomly(self.encoder, self.reducer)
            print("\nevaluate validation set")
            self.evaluateRandomly(self.encoder, self.reducer, self.triples_vali)

            self.evaluateAndShowAttention("1a", "mv1", self.encoder, self.reducer)

            self.evaluateAndShowAttention("1a", "mv2", self.encoder, self.reducer)

            self.evaluateAndShowAttention("2a", "mv3", self.encoder, self.reducer)

            self.evaluateAndShowAttention("2a", "mv6", self.encoder, self.reducer)

            if turn < folders-1:
                del self.encoder
                del self.reducer

    def do_with_pretrain(self):
        self.prepare_data()
        print("dictionary.n_words = ",self.dictionary.n_words)
        print("example triple ",random.choice(self.all_triples))

        train_vali_idx_lis=self.split_train_validate(0.1)
        part_num=len(train_vali_idx_lis)

        folders=3
        for turn in range(folders):
            tmp_triples=self.all_triples[:]
            train_index=train_vali_idx_lis[turn][0]
            vali_index=train_vali_idx_lis[turn][1]
            self.triples = [tmp_triples[x] for x in train_index]
            self.triples_vali = [tmp_triples[x] for x in vali_index]

            self.gru_hidden_size = 256
            
            # self.encoder = EncoderRNN(self.dictionary.n_words, self.gru_input_size, self.gru_hidden_size, self.embedding).to(device)
            self.load_pretrain_encoder_model(pretrain_path)
            self.re_dictionary()
            weight=torch.zeros(self.dictionary.n_words, self.gru_input_size, device=device, dtype=torch.float)
            n, w=self.encoder.embedding.weight.shape
            print("n, w",n,w)
            weight[:n, :w]=self.encoder.embedding.weight
            self.embedding = nn.Embedding.from_pretrained(weight)
            self.encoder.embedding=self.embedding
            self.reducer = AttnReducerRNN(self.dictionary.n_words, self.gru_input_size, self.gru_hidden_size, self.embedding, dropout_p=0.1).to(device)

            self.trainIters(self.encoder, self.reducer, 40000, print_every=1000, log_id=str(turn), enc_learning_rate=0.0001)

            self.encoder.eval()
            self.reducer.eval()

            print("evaluate vali set")
            self.evaluateAll(triples=self.triples_vali, log_id=str(turn))

            print("evaluate randomly")
            self.evaluateRandomly(self.encoder, self.reducer)
            print("\nevaluate validation set")
            self.evaluateRandomly(self.encoder, self.reducer, self.triples_vali)

            self.evaluateAndShowAttention("1a", "mv1", self.encoder, self.reducer)

            self.evaluateAndShowAttention("1a", "mv2", self.encoder, self.reducer)

            self.evaluateAndShowAttention("2a", "mv3", self.encoder, self.reducer)

            self.evaluateAndShowAttention("2a", "mv6", self.encoder, self.reducer)

            if turn < folders-1:
                del self.encoder
                del self.reducer


if __name__=="__main__":
    print("Use 0.1 validate rate, 3-folder, save evaluate result, with pretrain")
    # encoder_reducer=Encoder_reducer()
    # encoder_reducer.do_with_pretrain()

    # encoder_reducer.save_model()

    enc_red2=Encoder_reducer()
    enc_red2.prepare_data()
    enc_red2.load_model(o_path+"result/job/encoder_reducer/8.6-1/")
    # # enc_red2.evaluateAll(triples=[['22b','mv8','22b-8'],['16a','mv28','16a-28']])
    # enc_red2.evaluateAll()
    
    evl_q_time, evl_q_mv_time, attn, hidd=enc_red2.evaluate(enc_red2.q_data["3a"][0], enc_red2.mv_data["mv8"][0])
    print("q_tim", evl_q_time)
    print("q_mv_tim", evl_q_mv_time)
