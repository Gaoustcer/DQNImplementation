import torch.nn as nn 
import torch
if __name__ == "__main__":
    LSTM = nn.LSTM(input_size = 3,hidden_size = 5)
    inputtensor = torch.rand((7,3))
    # print(LSTM(inputtensor)[1].shape)
    for index,t in enumerate(LSTM(inputtensor)):
        print("index",index,"tensor is",t)