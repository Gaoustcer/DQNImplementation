from model.A_C import Actor,Critic
from model.Variation_Auto_encoder import VariationAutoEncode
from replaybuffer.static_dataset import Maze2d
import torch
import torch.nn as nn
import gym
import d4rl
from torch.utils.data import DataLoader
import torch.nn.functional as F
class BCQ_trainer(object):
    def __init__(self,state_dim = 4,action_dim = 2,latent_dim =2,tau = 0.02,discount_rate = 0.99,gamma = 0.98,EPOCH = 256,nabla = 0.75) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.tau = tau
        self.gamma = gamma
        self.EPOCH = EPOCH
        self.Actor = Actor().cuda()
        self.TargetActor = Actor().cuda()
        self.Critic = Critic().cuda()
        self.TargetCritic = Critic().cuda()
        self.optimizerActor = torch.optim.Adam(self.Actor.parameters(),lr = 0.0001)
        self.optimizerCritic = torch.optim.Adam(self.Critic.parameters(),lr = 0.0001)
        self.VAE = VariationAutoEncode().cuda()
        self.VAEoptimizer = torch.optim.Adam(self.VAE.parameters(),lr = 0.0001)
        self.testenv = gym.make('maze2d-umaze-v1')
        self.static_data = DataLoader(Maze2d(),batch_size=64)
        self.valfreq = 32
        self.discount = discount_rate
        self.action_sampletime = 10
        self.nabla = nabla
        self.testtime = 0
        self.traintime = 0
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter('./logBCQ')
    
    def validate(self):
        reward = 0
        for _ in range(self.valfreq):
            done = False
            state = self.testenv.reset()
            while done == False:
                action = self.Actor(state).cpu().detach().numpy()
                ns,r,done,_ = self.testenv.step(action)
                state = ns
                reward += r
        return reward/self.valfreq

    # def validate(self):
    #     reward = 0
    #     for _ in range(self.valfreq):
    #         state = self.testenv.reset()
    #         done = False
    #         while done == False:
    #             # action = self.VAE.generateactionfromstate(state)
    #             # print('action is',action)
    #             action = self.Actor(torch.from_numpy(state).cuda(),None).detach().cpu().numpy()
    #             nx,r,done,_ = self.testenv.step(action)
    #             state = nx
    #             reward += r
    #     return reward/self.valfreq

    def _softupdate(self):
        # pass
        for targetparm,parm in zip(self.TargetActor.parameters(),self.Actor.parameters()):
            targetparm.copy_(
                self.tau * parm + (1 - self.tau) * targetparm
            )
        for targetparm,parm in zip(self.TargetCritic.parameters(),self.Critic.parameters()):
            targetparm.copy_(
                self.tau * parm + (1 - self.tau) * targetparm
            )
        

    def train(self):
        from tqdm import tqdm
        for epoch in tqdm(range(self.EPOCH)):
            self._trainepoch()
            self._softupdate()
    
    def _trainepoch(self):
        from tqdm import tqdm
        for index,(current_state,action,reward,next_state,done) in tqdm(enumerate(self.static_data)):
            current_state = current_state.cuda()
            action = action.cuda()
            next_state = next_state.cuda()
            done = done.cuda()
            reward = reward.cuda()
            mu,sigma,vaecurrentactions = self.VAE(current_state)
            '''
            -0.5(1+torch.log(sigma ** 2)-mu ** 2-sigma ** 2)
            '''
            # update VAE net
            KLdivergence = -0.5 * (1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2).mean()
            Actionloss = F.mse_loss(vaecurrentactions,action)
            VAELoss = Actionloss + KLdivergence
            self.VAEoptimizer.zero_grad()
            VAELoss.backward()
            self.VAEoptimizer.step()
            '''
            Train Critic net,input is s,a,r,s^
            TDerror = Critic(s,a) - (TargetCritic(s^,vae(s^)) +  r)
            '''
            CurrentActionvalues = self.Critic(current_state,action)[0].squeeze()
            next_state = torch.repeat_interleave(next_state,self.action_sampletime,0)
            _,_,predactionsfornextstates = self.VAE(next_state)
            # print("predict is",predactionsfornextstates.shape)
            NextActionvalues1,NextActionvalues2 = self.TargetCritic(next_state,predactionsfornextstates)
            Nextvalue = self.nabla * torch.min(NextActionvalues1,NextActionvalues2) + (1 - self.nabla) * torch.max(NextActionvalues1,NextActionvalues2)
            # print("nextvalue is",Nextvalue.shape)
            Nextvalue = Nextvalue.reshape(self.static_data.batch_size,-1).max(1)[0]
            # print("current",CurrentActionvalues1.shape)
            # print("Next value",Nextvalue.shape)
            # exit()
            Target_value = (self.gamma* ~done * Nextvalue + reward)
            TDloss = F.mse_loss(CurrentActionvalues,(self.gamma * Target_value + reward).detach())
            self.optimizerCritic.zero_grad()
            TDloss.backward()
            self.optimizerCritic.step()
            '''
            Train Actor net,input is s Q(s,Actor(a)) is greater
            '''
            actionfromActor = self.Actor(current_state)
            values = -self.Critic(current_state,actionfromActor)[0].mean()
            self.optimizerActor.zero_grad()
            values.backward()
            self.optimizerActor.step()
            if index % 32 == 0:
                reward = self.validate()
                self.writer.add_scalar('reward',reward,self.testtime)
                self.testtime += 1
        

    # def _trainepoch(self):
    #     from tqdm import tqdm
    #     for current_state,action,reward,next_state,done in tqdm(self.static_data):
    #         current_state = current_state.cuda()
    #         action = action.cuda()
    #         next_state = next_state.cuda()
    #         done = done.cuda()
    #         reward = reward.cuda()

    #         constructionactions,mu,sigma = self.VAE(current_state,action)
    #         actionloss = F.mse_loss(constructionactions,action)
    #         actionloss += -0.5 * (1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2).to(torch.float32).mean()
    #         self.VAEoptimizer.zero_grad()
    #         actionloss.backward()
    #         self.VAEoptimizer.step()

    #         next_state = torch.repeat_interleave(next_state,self.action_sampletime,0)
    #         Next_value1,Next_value2 = self.TargetCritic(next_state,self.TargetActor(next_state,self.VAE.generateactionfromstate(next_state)))
    #         Next_value = self.nabla * torch.min(Next_value1,Next_value2) + (1 - self.nabla) * torch.max(Next_value1,Next_value2)
    #         # print(Current_value.shape)
            
    #         Next_value = Next_value.reshape(self.static_data.batch_size,-1).max(1)[0]
    #         Target_value =  (reward + ~done * self.discount * Next_value).to(torch.float32).detach()
    #         # print("Target value shape",Target_value.shape)
    #         Current_value1,Current_value2 = self.Critic(current_state,action)
    #         TDloss = F.mse_loss(Current_value1,Target_value) + F.mse_loss(Current_value2,Target_value)
    #         self.optimizerCritic.zero_grad()
    #         TDloss.backward()
    #         self.optimizerCritic.step()

    #         vaeactions = self.VAE.generateactionfromstate(current_state)
    #         newactions = self.Actor(current_state,vaeactions)
    #         actionvalue,_ = self.Critic(current_state,newactions)
    #         actionvalue = - actionvalue.mean()
    #         self.optimizerActor.zero_grad()
    #         actionvalue.backward()
    #         self.optimizerActor.step()
    #         if self.traintime % 32 == 0:
    #             r = self.validate()
    #             self.writer.add_scalar('reward',r,self.testtime)
    #             self.testtime += 1
                
                
    #         self.traintime += 1
            # exit()
            # (640,1)

            # exit()
        



if __name__ == "__main__":
    Agent = BCQ_trainer()
    # print(Agent.validate())
    # Agent._trainepoch()
    Agent.train()
    # Agent._trainepoch()
    