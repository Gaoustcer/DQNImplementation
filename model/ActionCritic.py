import torch
import torch.nn as nn
import numpy as np
class Actor(nn.Module):
    # generate Noise into Actor
    def __init__(self,input_dim = 4,output_dim = 2,noise = 0.01) -> None:
        super(Actor,self).__init__()
        self.action_encoding = nn.Sequential(
            nn.Linear(input_dim + output_dim,32),
            nn.ReLU(),
            nn.Linear(32,output_dim),
            nn.Tanh()
        )
        self.noise = noise

    def forward(self,state,action):
        if len(action) == 2 and len(action.shape) == 1:
            action = np.expand_dims(action,1)
        # if len(state.shape) == 1:
        embedding = self.noise * torch.from_numpy(np.concatenate([state,action],-1)).cuda().to(torch.float32)
        # elif len(state.shape) == 2:
        #     action = np.reshape(action,(len(action),1))
        #     embedding = self.noise * torch.from_numpy()
        action = torch.from_numpy(action).cuda()
        return (action + self.action_encoding(embedding)).clamp(-1,1)

from copy import deepcopy
class Critic(nn.Module):
    def __init__(self,input_dim = 4,output_dim = 2) -> None:
        super(Critic,self).__init__()
        self.C1 = nn.Sequential(
            nn.Linear(input_dim + output_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.C2 = deepcopy(self.C1)
    
    def forward(self,state,action):
        if len(action) == 2 and len(action.shape) == 1:
            action = np.expand_dims(action,1)
        # if len(state.shape) == 1:
        embedding = self.noise * torch.from_numpy(np.concatenate([state,action],-1)).cuda().to(torch.float32)
        # embedding = torch.from_numpy(np.concatenate())
        return self.C1(embedding),self.C2(embedding)
