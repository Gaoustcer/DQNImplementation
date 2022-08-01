import torch
import gym
import torch.nn as nn
import numpy as np

import random
class Net(nn.Module):
    def __init__(self,input_nums:4,output_nums:2,device:torch.device('cuda:0')) -> None:
        self.net = nn.Sequential(
            nn.Linear(input_nums,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,output_nums),
            nn.ReLU()
        ).to(device)
        self.device = device

    def forward(self,input):
        input = torch.from_numpy(input).to(self.device)
        return self.net(input)
def make_env():
    return gym.make('CartPole-v0')
class DQNAgent:
    def __init__(self,train_epoch=400,MAX_SIZE=1024,device = torch.device('cuda:0')) -> None:
        self.trainenv = make_env()
        self.testenv = make_env()
        self.targetnet = Net(device=device)
        self.net = Net(device=device)
        self.device = device
        self.memory = np.random.random((MAX_SIZE,10))
        self.memoryindex = 0
        self.epsilon = 0.1
        self.maxsize = MAX_SIZE
        self.batchsize = 32
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.001)
    
    def collectinfo(self):
        COUNT = 128
        numcollect = 0
        while True:
            done = False
            current_state = self.trainenv.reset()
            while done == False:
                if random.random() < self.epsilon:
                    action = random.randint(0,1)
                else:
                    action = torch.argmax(self.net(current_state))
                next_state,reward,done,_ = self.trainenv.step(action)
                self.memory[self.memoryindex,:] = np.vstack((current_state,(reward,action),next_state))
                current_state = next_state
                self.memoryindex += 1
                self.memoryindex %= self.maxsize
                numcollect += 1
                if numcollect >= COUNT:
                    return
    def learn(self):
        for _ in range(8):
            index = np.random.choice(self.maxsize,self.batchsize)
            data = self.memory[index,:]
            current_state = data[:,:4]
            action = torch.from_numpy(data[:,4]).to(self.device)
            reward = torch.from_numpy(data[:,5]).to(self.device)
            next_state = data[:,6:]
            TDreward = reward * self.gamma + torch.max(self.targetnet(next_state)[0],-1)[0].detach()
            expect_current_action_value = self.net(current_state)[0]
            expect_current_action_value = torch.gather(expect_current_action_value,-1,action.unsqueeze(-1))
            loss = torch.dist(expect_current_action_value,TDreward)
            # print("shape of TD is",TDreward.shape,"shape of current_action",expect_current_action_value.shape)
            # loss = torch.dist(torch.gather(self.net(current_state),-1,action).to(self.device).unsqueeze(-1)),TDreward)
            loss.backward()
            self.optimizer.step()