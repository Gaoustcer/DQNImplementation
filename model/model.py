import torch.nn as nn

import torch
import numpy as np

class Advantagenet(nn.Module):
    def __init__(self) -> None:
        super(Advantagenet,self).__init__()
        self.Advantagenet = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.ReLU()
        )
        self.Valuenet = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
    
    def forward(self,state:np.ndarray):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(torch.float32).cuda()
        Advantages = -self.Advantagenet(state)
        Values = self.Valuenet(state)
        StateActionvalues = Advantages + Values
        Maxadvatage = torch.max(Advantages,-1)[0].unsqueeze(-1)
        return StateActionvalues + Maxadvatage
