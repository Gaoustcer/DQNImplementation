import numpy as np

import matplotlib.pyplot as plt
from random import randint
from math import comb
from math import log2

N = 32
target = [randint(0,128)+1 for _ in range(N)]
targetpossibility = [s_/sum(target) for s_ in target]
def learningpossibility(theta):
    return [comb(N,i)*(theta**i)*((1-theta)**(N-i)) for i in range(N)]


def KL_divergence(p:list,q:list):
    KL_value = 0
    for p_,q_ in zip(p,q):
        KL_value += p_ * log2((p_/q_))
    return KL_value

if __name__ == "__main__":
    theta_list = []
    KL_list = [] 
    for theta in np.arange(0.1,0.9,0.01):
        theta_list.append(theta)
        theory_possibility = learningpossibility(theta)
        KL_list.append(KL_divergence(theory_possibility,targetpossibility))
    plt.scatter(x=theta_list,y=KL_list)
    plt.savefig('./KL_reverse.jpg')
