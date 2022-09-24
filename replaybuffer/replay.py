from collections import namedtuple
from random import choices
import numpy as np
class buffer(object):
    def __init__(self,max_size = 1024) -> None:
        self.TRANSITION = namedtuple('replaybuffer','current_state action reward next_state')
        self.memory = self.TRANSITION([],[],[],[])
        self.index = 0
        self.maxsize = max_size
        # self.size = 0
        self.size = 0
        self.full = False
    def _nextindex(self):
        self.index += 1
        self.index %= self.maxsize
    
    def push_memory(self,currentstate,action,reward,nextstate):
        if self.size < self.maxsize:
            self.memory.current_state.append(currentstate)
            self.memory.action.append(action)
            self.memory.reward.append(reward)
            self.memory.next_state.append(nextstate)
            self.size += 1
        else:
            self.memory.current_state[self.index] = currentstate
            self.memory.action[self.index] = action
            self.memory.reward[self.index] = reward
            self.memory.next_state[self.index] = nextstate
            self._nextindex()
            self.full = True

    def sample(self,n):
        assert n <= self.size
        index_list = choices(range(self.size),k=n)
        current_state = []
        action = []
        next_state = []
        reward = []
        for index in index_list:
            current_state.append(self.memory.current_state[index])
            action.append(self.memory.action[index])
            next_state.append(self.memory.next_state[index])
            reward.append(self.memory.reward[index])
        return np.array(current_state),np.array(action),np.array(reward).astype(float),np.array(next_state)


if __name__ == "__main__":
    memorybuffer = buffer()
    import gym
    import d4rl
    env = gym.make('maze2d-umaze-v1')
    done = False
    while True:
        cs = env.reset()
        done = False
        while done == False:
            a = env.action_space.sample()
            ns,r,done,_ = env.step(a)
            memorybuffer.push_memory(cs,a,r,ns)
        if memorybuffer.full == True:
            break
    current_state,action,reward,next_state = memorybuffer.sample(64)
    import torch
    current_state = torch.from_numpy(current_state)
    actions = torch.from_numpy(action)
    # print(current_state)
    # print(actions.shape)
    # print(actions)
    exit()
    M = 128
    N = 64
    from time import time
    start = time()
    for _ in range(M):
        memorybuffer.sample(N)
    end = time()
    print("Time cost is",end - start)
    

    