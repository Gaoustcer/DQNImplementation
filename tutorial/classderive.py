class Base:
    def __init__(self) -> None:
        print("init the base")

class Derive(Base):
    def __init__(self) -> None:
        super(Derive,self).__init__()
        print("init the Derive")

import numpy as np
def f(state = None):
    if state == None:
        print("state change")
        state = np.random.random(2)
if __name__ == "__main__":
    state = None
    f(state)
    print(state)