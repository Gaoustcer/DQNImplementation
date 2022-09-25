# from dqnPong import Net
import numpy as np
import gym
import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Conv2d(3,2,4,2),
    nn.ReLU(),
    nn.Conv2d(2,1,4,3),
    # nn.ReLU(),
    # nn.Conv2d(1,1,3,2)
)
if __name__ == "__main__":
    env = gym.make('Pong-v0')
    state = env.reset()
    state = np.transpose(state,(-1,-3,-2))
    result = net(torch.from_numpy(state).to(torch.float32))
    print(result.shape)
    exit()
    done = False
    deltastate = 0
    while done == False:
        state,reward,done,info = env.step(env.action_space.sample())
        deltastate += reward
    print(deltastate)
