from __future__ import annotations
import numpy as np

class Frog:
    def __init__(self, dimension, weight, value, capacity):
        self.dimension = dimension
        self.weight = weight
        self.value = value
        self.capacity = capacity
        self.x = self.generate_x()
        self.fitness = self.calc_fitness()

    def generate_x(self):
        return np.random.randint(0, 2, self.dimension)

    def calc_fitness(self):
        cur_weight = 0
        cur_value = 0
        for i in range(self.dimension):
            if self.x[i]:
                cur_weight += self.weight[i]
                cur_value += self.value[i]
        return cur_value if cur_weight <= self.capacity else 0

    def mutation(self, p_m):
        for i in range(self.dimension):
            if np.random.rand() < p_m:
                self.x[i] = (self.x[i] + 1) % 2


class MDSFLA:
    def __init__(self, capacity: int, weight: list[int], value: list[int]):
        self.capacity = capacity
        self.weight = weight
        self.value = value
        self.dimension = len(weight)
        # parameters
        self.P = 200 # population size
        self.m = 10 # number of memeplexes
        self.iMax = 100 # number of iterations within each memeplex
        self.p_m = 0.06 # genetic mutation probability
        # frogs
        self.frogs = np.array([])

    def solve(self):
        # generate population of P frogs randomly
        self.generate_frogs()

        for _ in range(self.iMax):
            # sort P frogs in descending order
            self.frogs = sorted(self.frogs, key=lambda x: x.fitness, reverse=True)
            # partition P frogs into m memeplexes
            cur_memeplexes = [[] for i in range(self.m)]
            for i, frog in enumerate(self.frogs):
                cur_memeplexes[i % self.m].append((frog))
            cur_memeplexes = np.array(cur_memeplexes)
            # local search
            # local_search(cur_memeplexes) # -> shuffled the m memeplexed
            # apply mutation on the population
            for frog in self.frogs:
                frog.mutation(self.p_m)

        # determine the best solution
        self.frogs = sorted(self.frogs, key=lambda x: x.fitness, reverse=True)
        return self.frogs[0]

    def generate_frogs(self):
        for i in range(self.P):
            # evaluate the fitness of the frog
            self.frogs = np.append(self.frogs, Frog(self.dimension, self.weight, self.value, self.capacity))



def main():
    problem = "p07"
    # read problem files
    f = open(f"../problem/{problem}/{problem}_c.txt", "r")
    data = f.read()
    capacity = int(data)
    f = open(f"../problem/{problem}/{problem}_w.txt", "r")
    data = f.read()
    weight = [int(i) for i in data.split()]
    f = open(f"../problem/{problem}/{problem}_p.txt", "r")
    data = f.read()
    value = [int(i) for i in data.split()]
    f = open(f"../problem/{problem}/{problem}_s.txt", "r")
    data = f.read()
    ans = [int(i) for i in data.split()]

    # solve
    sfla = MDSFLA(capacity, weight, value)
    result = sfla.solve()
    print(result.x, result.fitness)


if __name__ == "__main__":
    main()