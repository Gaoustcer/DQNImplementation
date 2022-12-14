import torch.nn as nn

import torch
import numpy as np

class Advantagenet(nn.Module):
    def __init__(self) -> None:
        super(Advantagenet,self).__init__()
        # self.Advantagenet = nn.Sequential(
        #     nn.Linear(4,8),
        #     nn.ReLU(),
        #     nn.Linear(8,4),
        #     nn.ReLU(),
        #     nn.Linear(4,2),
        #     nn.ReLU()
        # )
        # self.Valuenet = nn.Sequential(
        #     nn.Linear(4,8),
        #     nn.ReLU(),
        #     nn.Linear(8,4),
        #     nn.ReLU(),
        #     nn.Linear(4,1)
        # )
        self.featureextract = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,32)
        )
        self.Advantageencoding = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.ReLU()
        )
        self.Valueencoding = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    
    def forward(self,state:np.ndarray):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(torch.float32).cuda()
        features = self.featureextract(state)

        Advantages = -self.Advantageencoding(features)
        Values = self.Valueencoding(features)
        StateActionvalues = Advantages + Values
        Maxadvatage = torch.max(Advantages,-1)[0].unsqueeze(-1)
        return StateActionvalues + Maxadvatage
