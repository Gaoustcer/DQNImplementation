import d4rl
import gym
from torch.utils.data import Dataset
class Maze2d(Dataset):
    def __init__(self) -> None:
        super(Maze2d,self).__init__()
        self.env = gym.make('maze2d-umaze-v1')
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.len = len(self.dataset['actions'])

    def __len__(self):
        return self.len
        # return len(self.dataset['actions'])
    
    def __getitem__(self, index):
        return self.dataset['observations'][index],self.dataset['actions'][index],self.dataset['reward'][index],self.dataset['next_observations'][index]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    loader = DataLoader(Maze2d(),batch_size=32)
    for element in loader:
        for e in element:
            print(e.shape)
        exit()
        # return super().__getitem__(index)