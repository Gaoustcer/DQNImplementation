import torch
import numpy as np
from model.A_C import Actor,Critic

from DQN.dqnDueling import AgentBase
import gym
import torch.nn.functional as F
import d4rl
def make_an_env():
    return gym.make('maze2d-umaze-v1')

class DDPGAgent(AgentBase):
    def __init__(self, train_epoch=400, MAX_SIZE=1024, sample_size=32, tau=0.005, lr=0.0001, gamma=0.99, logdir='./log/base', shaping=False,Env = "maze2d-umaze-v1") -> None:
        # self.train_env = make_env()
        # self.test_env = make_env()
        self.ActorNet = Actor().cuda()
        self.TargetActorNet = Actor().cuda()
        self.CriticNet = Critic().cuda()
        self.TargetCriticNet = Critic().cuda()
        self.rewardshape = False
        self.lr = lr
        self.criticoptimizer = torch.optim.Adam(self.CriticNet.parameters(),lr = self.lr)
        self.actoroptimizer = torch.optim.Adam(self.ActorNet.parameters(),lr = self.lr)
        super(DDPGAgent,self).__init__(train_epoch, MAX_SIZE, sample_size, tau, lr, gamma, logdir, shaping)
        # self.train_env = make_an_env()
        # self.test_env = make_an_env()
        self.rewardshape = False
        self.rewardindex = 0
        self.valueindex = 0
        # print("Initial the env")
        # print("What is action size",self.train_env.action_space.sample())
        # exit()
    
    def _softupdate(self):
        # return super()._softupdate
        for target,param in zip(self.TargetActorNet.parameters(),self.ActorNet.parameters()):
            target.copy_(
                self.tau * param + (1 - self.tau) * target
            )
        for target,param in zip(self.TargetCriticNet.parameters(),self.CriticNet.parameters()):
            target.copy_(
                self.tau * param + (1 - self.tau) * target
            )
        
    def test(self):
        reward = 0
        for _ in range(self.testfreq):
            done = False
            state = self.test_env.reset()
            while done == False:
                action = self.ActorNet(state).cpu().detach().numpy()
                ns,r,done,_ = self.test_env.step(action)
                state = ns
                reward += r
        return reward/self.testfreq
    
    def interactwithenv(self, state=None):
        if state is None:
            state = self.train_env.reset()
        action = np.array(self.ActorNet(state).cpu().detach()).astype(np.float32)
        # print("action is",action)
        # print("action space is",self.train_env.action_space.sample())
        return action,state
        # return super().interactwithenv(state)
    def _trackvalue(self,value):
        self.writer.add_scalar('value',value,self.valueindex)
        self.valueindex += 1
    def _trainanepoch(self):
        # return super()._trainanepoch()
        for i in range(4):
            current_state,action,reward,next_state = self.buffer.sample(self.sample_size)
            Current_value = self.CriticNet(current_state,action).squeeze()
            # print()
            with torch.no_grad():
                next_action = self.TargetActorNet(next_state)
                # print("nextstate shape",next_state.shape)
                # print("nextaction shape",next_action.shape)
                Next_value = self.gamma * self.TargetCriticNet(next_state,next_action).squeeze()+torch.from_numpy(reward).cuda().to(torch.float32)
                # print("forward result",self.TargetCriticNet(next_state,next_action)[0].shape)
            TDloss = F.mse_loss(Current_value,Next_value)
            # print("Current value",Current_value.shape)
            # print("Next Value",Next_value.shape)
            # # print(self.TargetActorNet(next_state,next_action).shape)
            # exit()
            self.criticoptimizer.zero_grad()
            TDloss.backward()
            self._trackloss(TDloss)
            self.criticoptimizer.step()
            values = self.CriticNet(current_state,self.ActorNet(current_state))
            values = values.mean()
            self._trackvalue(values)
            self.actoroptimizer.zero_grad()
            values.backward()
            self.actoroptimizer.step()
        reward = self.test()
        self.writer.add_scalar("reward",reward,self.rewardindex)
        self.rewardindex += 1
        # self._trackloss()
    
if __name__ == "__main__":
    Agent = DDPGAgent(logdir='./logBCQ/DDPG')
    Agent.train()