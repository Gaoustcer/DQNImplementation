import torch
import gym
import torch.nn as nn
import torch
import numpy as np
from replaybuffer.replay import buffer
import random
from torch.utils.tensorboard import SummaryWriter
# class net(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

def initial(layer):
    if isinstance(layer,nn.Linear):
        nn.init.xavier_normal_(layer.weight)
class Net(nn.Module):
    def __init__(self,input_nums=4,output_nums=2,device=torch.device('cuda:0')) -> None:
        super(Net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_nums,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,output_nums)
        ).to(device)
        self.net.apply(initial)
        self.device = device

    def forward(self,input):
        if isinstance(input,np.ndarray):
            input = torch.from_numpy(input).to(self.device).to(torch.float32)
        return self.net(input)
def make_env():
    return gym.make('CartPole-v0')
class AgentBase:
    def __init__(self,train_epoch = 400,MAX_SIZE = 1024,sample_size = 32,tau = 0.005,lr = 0.0001,gamma = 0.99,logdir = './log/base',shaping = False) -> None:
        self.train_env = make_env()
        self.test_env = make_env()
        self.targetnet = Net().cuda()
        self.net = Net().cuda()
        self.sample_size = sample_size
        self.tau = tau
        self.EPOCH = train_epoch
        self.writer = SummaryWriter(logdir)
        self.testfreq = 32
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr = self.lr)
        self.buffer = buffer(max_size=MAX_SIZE)
        self.gamma = gamma
        self.rewardshape = shaping
        self._bufferinit()
        self.tderrorfunction = nn.MSELoss()
        
    def interactwithenv(self,state=None):
        # def _decisionmake():
    
        if state is None:
            state = self.train_env.reset()
        action = torch.argmax(self.net(state)).item()
        return action,state
    def _getrewardshapeing(self,next_state):
        x,x_dot,theta,theta_dot = next_state
        r1 = (self.train_env.x_threshold - abs(x)) / self.train_env.x_threshold - 0.8
        r2 = (self.train_env.theta_threshold_radians - abs(theta)) / self.train_env.theta_threshold_radians - 0.5
        return r1 + r2
    def _pushintoreplaybuffer(self,state):
        action,state = self.interactwithenv(state)
        nextstate,reward,done,info = self.train_env.step(action)
        if self.rewardshape:
            reward = self._getrewardshapeing(nextstate)
        self.buffer.push_memory(state,action,reward,nextstate)
        return nextstate,done
        
    def _bufferinit(self):
        while self.buffer.full == False:
            done = False
            state = None
            while done == False:
                state,done = self._pushintoreplaybuffer(state)
    
    def collectnewdata(self,K=4):
        done = False
        for _ in range(K):
            state = None
            done = False
            while done == False:
                state,done = self._pushintoreplaybuffer(state)
    def test(self):
        reward = 0
        for _ in range(self.testfreq):
            done = False
            state = self.test_env.reset()
            while done == False:
                action = torch.argmax(self.net(state)).item()
                ns,r,done,_ = self.test_env.step(action)
                state = ns
                reward += r
        return reward/self.testfreq
    
    def _softupdate(self):
        raise NotImplementedError
    def train(self):
        from tqdm import tqdm
        for epoch in tqdm(range(self.EPOCH)):
            reward = self.test()
            self.writer.add_scalar('reward',reward,epoch)
            self.collectnewdata()
            self._trainanepoch()
            # reward = self.test()
            # self.writer.add_scalar("reward",reward,epoch)

    def _trainanepoch(self):
        raise NotImplementedError("You should implement train function for your purpose")

class DQNNaive(AgentBase):
    def __init__(self, train_epoch=400, MAX_SIZE=1024, sample_size=32, tau=0.005, lr=0.0001,gamma = 0.99,logdir = "./log/NaiveDQN",shaping=False) -> None:
        super(DQNNaive,self).__init__(train_epoch, MAX_SIZE, sample_size, tau, lr,gamma,logdir,shaping)
        
    
    def _trainanepoch(self,update_freq=16):
        for _ in range(update_freq):
            states,actions,rewards,nextstates = self.buffer.sample(self.sample_size)
            # stateactionvalues = self.net(states)
            actions = torch.from_numpy(actions).to(torch.long).cuda().unsqueeze(-1)
            stateactionvalues = torch.gather(self.net(states),-1,actions).squeeze()
            nextstateactionvalues = torch.max(self.net(nextstates),-1)[0]
            TDreward = (nextstateactionvalues + torch.mul(self.gamma,torch.from_numpy(rewards).cuda())).to(torch.float32).detach()
            self.optimizer.zero_grad()
            TDerror = self.tderrorfunction(stateactionvalues,TDreward)
            TDerror.backward()
            self.optimizer.step()
    def _softupdate(self):
        pass
        # self.logdir = logdir
        # self.writer = SummaryWriter(self.logdir)
    # def train(self):
        # for epoch in range(self.EPOCH):
class DQNrandomsample(DQNNaive):
    def __init__(self, train_epoch=400, MAX_SIZE=1024, sample_size=32, tau=0.005, lr=0.0001, gamma=0.99, logdir="./log/NaiveDQNrandomsample", shaping=False) -> None:
        super(DQNrandomsample,self).__init__(train_epoch, MAX_SIZE, sample_size, tau, lr, gamma, logdir, shaping)
    
    def interactwithenv(self, state=None):
        if state is None:
            state = self.train_env.reset()
        return self.train_env.action_space.sample(),state
        # return super().interactwithenv(state)


class DQNTarget_softupdate(AgentBase):
    def __init__(self, train_epoch=400, MAX_SIZE=1024, sample_size=32, tau=0.005, lr=0.0001, gamma=0.99, logdir='./log/DQNTarget', shaping=False) -> None:
        super().__init__(train_epoch, MAX_SIZE, sample_size, tau, lr, gamma, logdir, shaping)
    
    def _softupdate(self):
        for targetparam,naiveparm in zip(self.targetnet.parameters(),self.net.parameters()):
            targetparam.copy_(
                self.tau * naiveparm + (1 - self.tau) * targetparam
            )
    def _trainanepoch(self,update_freq=16):
        for _ in range(update_freq):
            states,actions,rewards,nextstates = self.buffer.sample(self.sample_size)
            # stateactionvalues = self.net(states)
            actions = torch.from_numpy(actions).to(torch.long).cuda().unsqueeze(-1)
            stateactionvalues = torch.gather(self.net(states),-1,actions).squeeze()
            nextstateactionvalues = torch.max(self.targetnet(nextstates),-1)[0]
            TDreward = (nextstateactionvalues + torch.mul(self.gamma,torch.from_numpy(rewards).cuda())).to(torch.float32).detach()
            self.optimizer.zero_grad()
            TDerror = self.tderrorfunction(stateactionvalues,TDreward)
            TDerror.backward()
            self.optimizer.step()

class DQNDoubleTarget_softupdate(AgentBase):
    def __init__(self, train_epoch=400, MAX_SIZE=1024, sample_size=32, tau=0.005, lr=0.0001, gamma=0.99, logdir='./log/DQNDoubleTarget', shaping=False) -> None:
        super().__init__(train_epoch, MAX_SIZE, sample_size, tau, lr, gamma, logdir, shaping)
        self.lossindex = 0
    def _softupdate(self):
        for targetparam,naiveparm in zip(self.targetnet.parameters(),self.net.parameters()):
            targetparam.copy_(
                self.tau * naiveparm + (1 - self.tau) * targetparam
            )
    def _trackloss(self,loss):
        self.writer.add_scalar('TDerror',loss,self.lossindex)
        self.lossindex += 1
    def _trainanepoch(self,update_freq=16):
        for _ in range(update_freq):
            states,actions,rewards,nextstates = self.buffer.sample(self.sample_size)
            # stateactionvalues = self.net(states)
            actions = torch.from_numpy(actions).to(torch.long).cuda().unsqueeze(-1)
            nextactions = torch.argmax(self.net(nextstates),dim=-1).unsqueeze(-1).cuda()
            stateactionvalues = torch.gather(self.net(states),-1,actions).squeeze()
            # nextstateactionvalues = torch.max(self.targetnet(nextstates),-1)[0]
            nextstateactionvalues = torch.gather(self.targetnet(nextstates),-1,nextactions).squeeze()
            TDreward = (nextstateactionvalues + torch.mul(self.gamma,torch.from_numpy(rewards).cuda())).to(torch.float32).detach()
            self.optimizer.zero_grad()
            # print(TDreward.shape)
            # print(stateactionvalues.shape)
            # exit()
            TDerror = self.tderrorfunction(stateactionvalues,TDreward)
            self._trackloss(TDerror)
            TDerror.backward()
            self.optimizer.step()
from model.model import Advantagenet

class DuelingDQN(DQNTarget_softupdate):
    def __init__(self, train_epoch=400, MAX_SIZE=1024, sample_size=32, tau=0.005, lr=0.0001, gamma=0.99, logdir='./log/DQNDuelingTarget', shaping=False) -> None:
        super(DuelingDQN,self).__init__(train_epoch, MAX_SIZE, sample_size, tau, lr, gamma, logdir, shaping)
        self.Advantagenet = Advantagenet().cuda()
        self.optimizer = torch.optim.Adam(self.Advantagenet.parameters(),lr = self.lr)
        self.targetAdvantagenet = Advantagenet().cuda()

    def _trainanepoch(self, update_freq=16):
        # return super()._trainanepoch(update_freq)
        for _ in range(update_freq):
            
    # def train(self):
        # return super().train()

        
        
# class DQNAgent:
#     # import datetime
#     def __init__(self,train_epoch=400,MAX_SIZE=512,device = torch.device('cuda:0')) -> None:
#         self.trainenv = make_env()
#         self.testenv = make_env()
#         self.targetnet = Net(device=device)
#         self.net = Net(device=device)
#         self.device = device
#         self.memory = np.random.random((MAX_SIZE,10))
#         self.memoryindex = 0
#         self.epsilon = 0.1
#         self.maxsize = MAX_SIZE
#         self.batchsize = 32
#         self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.01)
#         self.testtimes = 32
#         self.trainepoch = train_epoch
#         self.testlog = 0
#         self.gamma = 0.9
#         self.trainlog = 0
#         self.lossfunc = nn.MSELoss()
#         from torch.utils.tensorboard import SummaryWriter
#         import datetime
#         self.writer = SummaryWriter('./logdir/TargetDQN/'+datetime.datetime.now().strftime('%m%d_%H%M%S'))
    
#     def collectinfo(self,COUNT=128):
#         # COUNT = 128
#         numcollect = 0
#         while True:
#             done = False
#             current_state = self.trainenv.reset()
#             # current_state = np.array(current_state)
#             while done == False:
#                 if random.random() < self.epsilon:
#                     action = random.randint(0,1)
#                 else:
#                     action = torch.argmax(self.net(current_state))
#                     action = int(action)
                
#                 next_state,reward,done,_ = self.trainenv.step(action)
#                 x,x_dot,theta,theta_dot = next_state
#                 r1 = (self.trainenv.x_threshold - abs(x)) / self.trainenv.x_threshold - 0.8
#                 r2 = (self.trainenv.theta_threshold_radians - abs(theta)) / self.trainenv.theta_threshold_radians - 0.5
#                 reward = r1 + r2
#                 self.memory[self.memoryindex,:] = np.hstack((current_state,[action,reward],next_state))
#                 current_state = next_state
#                 self.memoryindex += 1
#                 self.memoryindex %= self.maxsize
#                 numcollect += 1
#                 if numcollect >= COUNT:
#                     return
#     def testresult(self):
#         total_reward = 0
#         for _ in range(self.testtimes):
#             done = False
#             state = self.testenv.reset()
#             while done == False:
#                 action = torch.argmax(self.net(state)).item()
#                 state,reward,done,_ = self.testenv.step(action)
#                 total_reward += reward
#         self.testlog += 1
#         self.writer.add_scalar('reward',total_reward/self.testtimes,self.testlog)
#     def updatenetparameter(self):
#         for _ in range(1):
#             self.optimizer.zero_grad()
#             index = np.random.choice(self.maxsize,self.batchsize)
#             data = self.memory[index,:]
#             current_state = data[:,:4]
#             action = torch.from_numpy(data[:,4]).to(self.device).to(torch.int64)
#             reward = torch.from_numpy(data[:,5]).to(self.device).to(torch.float32)
#             next_state = data[:,6:]
#             TDreward = reward + self.gamma * torch.max(self.targetnet(next_state),dim=-1)[0].detach()
#             # print(self.targetnet(next_state))
#             # print(torch.max(self.targetnet(next_state),dim=-1)[0])
#             # exit()
#             # print(self.net(current_state))
#             # print(action.unsqueeze(-1))
#             # print(torch.gather(self.net(current_state),-1,action.unsqueeze(-1)))
#             # exit()
#             expect_current_action_value = torch.gather(self.net(current_state),-1,action.unsqueeze(-1)).squeeze()
            
#             loss = self.lossfunc(expect_current_action_value,TDreward)
            
#             loss.backward()
#             self.trainlog += 1
#             self.writer.add_scalar('loss',loss,self.trainlog)
#             self.optimizer.step()
    

#     def learn(self):
#         # self.trainlog = 0
#         self.collectinfo(COUNT=self.maxsize)
#         from tqdm import tqdm
#         train_time = 0
#         for _ in tqdm(range(self.trainepoch)):
#             for time in range(self.maxsize//self.batchsize):
#                 # from time import time
#                 # start = time()
#                 self.collectinfo()
#                 # end = time()
#                 # print("Collect time is",end - start)
#                 self.updatenetparameter()
#                 self.testresult()
#                 train_time += 1
#                 # start = time()
#                 # print('train time is',start - end)
#                 if train_time % 64 == 0:
#                     self.targetnet.load_state_dict(self.net.state_dict())
#             # self.testresult()


# if __name__ == "__main__":
    # pass
    # Agent = DQNAgent()
    # Agent.learn()

if __name__ == "__main__":
    # Agent = DQNNaive(logdir='./log/DQNNaive',shaping=True)
    # Agent = DQNTarget_softupdate(shaping=True)
    Agent = DQNDoubleTarget_softupdate(shaping=True)
    # Agent = DQNrandomsample(shaping=True)
    Agent.train()