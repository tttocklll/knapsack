class MDSFLA:
    def __init__(self, capacity: int, weight: list[int], value: list[int]):
        self.capacity = capacity
        self.weight = weight
        self.value = value
        # parameters
        self.P = 200 # population size
        self.m = 10 # number of memeplexes
        self.iMax = 100 # number of iterations within eaach memeplex


