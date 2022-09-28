import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Normal
class VariationAutoEncode(nn.Module):
    def __init__(self,input_dim = 4,output_dim = 2,latent_dim = 2) -> None:
        super(VariationAutoEncode,self).__init__()
        self.meanencode = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,latent_dim)
        )
        self.sigmaencode = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,latent_dim)
        )
        self.generate_actions = nn.Sequential(
            nn.Linear(latent_dim,16),
            nn.ReLU(),
            nn.Linear(16,output_dim),
            nn.Tanh()
        )
        self.latent_dim = latent_dim
        self.normal = Normal(0,1)
    # for training purpose,use (states,actions) for output
    def forward(self,states):
        if isinstance(states,np.ndarray):
            states = torch.from_numpy(states)
        states = states.cuda().to(torch.float32)
        sigma = self.sigmaencode(states)
        mu = self.meanencode(states)
        noise = self.normal.sample((sigma.shape)).cuda()
        encoding = noise * sigma + mu
        return mu,sigma,self.generate_actions(encoding)


    # def forward(self,states,actions):
    #     if isinstance(states,np.ndarray):
    #         states = torch.from_numpy(states)
    #     states = states.cuda().to(torch.float32)
    #     if isinstance(actions,np.ndarray):
    #         actions = torch.from_numpy(actions)
    #     actions = actions.cuda().to(torch.float32)
    #     if len(actions.shape) != 2 and len(states.shape)!=1:
    #         actions = torch.unsqueeze(actions,-1)
    #     input_tensor = torch.concat([states,actions],1)
    #     # print(input_tensor.shape)
    #     mean = self.meanencode(input_tensor)
    #     sigma = self.sigmaencode(input_tensor)
    #     noise = sigma * torch.normal(0,1,mean.shape).cuda() + mean
    #     # print("noise",noise.shape)
    #     # print("state",states.shape)
    #     # print('concat',torch.concat([states,noise],-1).shape)
    #     rebultactions = self.generate_actions(torch.concat([states,noise],-1))
    #     return rebultactions,mean,sigma
    
    # def generateactionfromstate(self,states:np.ndarray):
    #     if isinstance(states,np.ndarray):
    #         states = torch.from_numpy(states)
    #     states = states.to(torch.float32).cuda()
    #     if len(states.shape) == 1:
    #         noisesize = 1
    #     else:
    #         noisesize = states.shape[0]
    #     noise = torch.normal(0,1,(noisesize,self.latent_dim)).cuda()
    #     # print("noise is",noise.shape)
    #     # print("state is",states.shape)
    #     if len(states.shape) == 1:
    #         noise = noise.squeeze()
    #         # states = states.unsqueeze(-1)
    #         # noise = noise.squeeze()
    #         # print(noise.shape)
    #         # print(states.shape)
    #         # self.generate_actions(torch.concat([states,noise],0))
    #     return self.generate_actions(torch.concat([states,noise],-1))
        # self.sigmaencode = 

if __name__ == "__main__":
    VAEnet = VariationAutoEncode(input_dim=4,output_dim=1).cuda()
    states = np.random.random((5,4))
    actions = np.random.random(5)
    # states = np.random.random(4)
    # actions = np.random.random(1)
    r_action,mean,sigma = VAEnet(states,actions)
    print(r_action)
    print(mean)
    print(sigma)
    loss = torch.log(sigma**2) - sigma**2 - mean**2
    print("loss is",loss)
    print('average loss is',loss.mean())
    print(VAEnet.generateactionfromstate(np.random.random((5,4))))
