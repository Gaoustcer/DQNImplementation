class Base:
    def __init__(self) -> None:
        print("init the base")
        self.x = 1024
class Derive(Base):
    def __init__(self) -> None:
        super().__init__()
        print("init the derive")
        self.x = 2

if __name__ == "__main__":
    d = Derive()
    print(d.x)