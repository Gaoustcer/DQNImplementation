import torch
import gym
import torch.nn as nn
import torch
import numpy as np

import random
# class net(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

def initial(layer):
    if isinstance(layer,nn.Linear):
        nn.init.xavier_normal_(layer.weight)
# origin input is N,H,W,C and expected input is N,C,H,W
# np.transpose((0,3,2,1)),(2,1,0) when N = 1
class Net(nn.Module):
    def __init__(self,input_nums=4,output_nums=2,device=torch.device('cuda:0')) -> None:
        super(Net,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,2,4,2),
            nn.ReLU(),
            nn.Conv2d(2,1,4,3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(884,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,6)
        ).to(device)
        # self.net = nn.Sequential(
        #     nn.Linear(input_nums,32),
        #     nn.ReLU(),
        #     nn.Linear(32,16),
        #     nn.ReLU(),
        #     nn.Linear(16,output_nums)
        # ).to(device)
        self.net.apply(initial)
        self.device = device

    def forward(self,input:np.array):
        if input.ndim == 3:
            input = np.transpose(input,(-1,-3,-2))
        elif input.ndim == 4:
            input = np.transpose(input,(0,-1,-3,-2))
        input = torch.from_numpy(input).to(self.device).to(torch.float32)
        return self.net(input)
def make_env():
    return gym.make("Pong-v0")
    # return gym.make('CartPole-v0')
class DQNAgent:
    # import datetime
    def __init__(self,train_epoch=400,MAX_SIZE=512,device = torch.device('cuda:0')) -> None:
        self.trainenv = make_env()
        self.testenv = make_env()
        self.targetnet = Net(device=device)
        self.net = Net(device=device)
        self.device = device
        # self.memory = np.random.random((MAX_SIZE,10))
        self.currentstate = np.random.random((MAX_SIZE,210,160,3))
        self.reward = np.random.random((MAX_SIZE))
        self.action = np.random.random(MAX_SIZE)
        self.nextstate = np.random.random((MAX_SIZE,210,160,3))
        self.memoryindex = 0
        self.epsilon = 0.1
        self.maxsize = MAX_SIZE
        self.batchsize = 64
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.01)
        self.testtimes = 4
        self.trainepoch = train_epoch
        self.testlog = 0
        self.gamma = 0.9
        self.trainlog = 0
        self.lossfunc = nn.MSELoss()
        from torch.utils.tensorboard import SummaryWriter
        import datetime
        self.writer = SummaryWriter('./logdir/TargetDQNPong/'+datetime.datetime.now().strftime('%m%d_%H%M%S'))
    
    def collectinfo(self,COUNT=128):
        # COUNT = 128
        numcollect = 0
        while True:
            done = False
            current_state = self.trainenv.reset()
            # current_state = np.array(current_state)
            while done == False:
                if random.random() < self.epsilon:
                    action = random.randint(0,1)
                else:
                    action = torch.argmax(self.net(current_state))
                    action = int(action)
                
                next_state,reward,done,_ = self.trainenv.step(action)
                # x,x_dot,theta,theta_dot = next_state
                # r1 = (self.trainenv.x_threshold - abs(x)) / self.trainenv.x_threshold - 0.8
                # r2 = (self.trainenv.theta_threshold_radians - abs(theta)) / self.trainenv.theta_threshold_radians - 0.5
                # reward = r1 + r2
                # self.memory[self.memoryindex,:] = np.hstack((current_state,[action,reward],next_state))
                self.currentstate[self.memoryindex] = current_state
                self.nextstate[self.memoryindex] = next_state
                self.reward[self.memoryindex] = reward
                self.action[self.memoryindex] = action
                current_state = next_state
                self.memoryindex += 1
                self.memoryindex %= self.maxsize
                numcollect += 1
                if numcollect >= COUNT:
                    return
    def testresult(self):
        total_reward = 0
        for i in range(self.testtimes):
            done = False
            state = self.testenv.reset()
            while done == False:
                action = torch.argmax(self.net(state)).item()
                state,reward,done,_ = self.testenv.step(action)
                total_reward += reward
            # print("test finish",i)
        self.testlog += 1
        self.writer.add_scalar('reward',total_reward/self.testtimes,self.testlog)
    def updatenetparameter(self):
        for _ in range(1):
            self.optimizer.zero_grad()
            index = np.random.choice(self.maxsize,self.batchsize)
            
            # data = self.memory[index,:]
            # current_state = data[:,:4]
            action = torch.from_numpy(self.action[index]).to(self.device).to(torch.int64)
            reward = torch.from_numpy(self.reward[index]).to(self.device).to(torch.float32)
            next_state = self.nextstate[index]
            current_state = self.currentstate[index]
            TDreward = reward + self.gamma * torch.max(self.targetnet(next_state),dim=-1)[0].detach()
            # print(self.targetnet(next_state))
            # print(torch.max(self.targetnet(next_state),dim=-1)[0])
            # exit()
            # print(self.net(current_state))
            # print(action.unsqueeze(-1))
            # print(torch.gather(self.net(current_state),-1,action.unsqueeze(-1)))
            # exit()
            expect_current_action_value = torch.gather(self.net(current_state),-1,action.unsqueeze(-1)).squeeze()
            
            loss = self.lossfunc(expect_current_action_value,TDreward)
            
            loss.backward()
            self.trainlog += 1
            self.writer.add_scalar('loss',loss,self.trainlog)
            self.optimizer.step()
    

    def learn(self):
        # self.trainlog = 0
        self.collectinfo(COUNT=self.maxsize)
        from tqdm import tqdm
        train_time = 0
        for _ in (range(self.trainepoch)):
            for time in tqdm(range(self.maxsize//self.batchsize)):
                # from time import time
                # start = time()
                self.collectinfo()
                # end = time()
                # print("Collect time is",end - start)
                self.updatenetparameter()
                # print("Update finished")
                self.testresult()
                # print("Test finished")
                train_time += 1
                # start = time()
                # print('train time is',start - end)
                if train_time % 64 == 0:
                    self.targetnet.load_state_dict(self.net.state_dict())
                # print("Train one time finish!")
            # self.testresult()


if __name__ == "__main__":
    Agent = DQNAgent()
    Agent.learn()

