from __future__ import annotations
import numpy as np
import copy
import tqdm
method = 3


class Frog:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.x = self.generate_x()
        self.fitness = 0

    def generate_x(self):
        return np.random.randint(0, 2, self.dimension)

    def calc_fitness(self, capacity: int, weight: list[int], value: list[int]):
        cur_weight = 0
        cur_value = 0
        for i in range(self.dimension):
            if self.x[i]:
                cur_weight += weight[i]
                cur_value += value[i]
        self.fitness = cur_value if cur_weight <= capacity else 0

    def mutation(self, p_m: float):
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
        self.P = 200  # population size
        self.m = 10  # number of memeplexes
        self.iMax = 100  # number of iterations within each memeplex
        self.p_m = 0.06  # genetic mutation probability
        # frogs
        self.frogs = np.array([])
        self.best_frog = None

    def generate_frogs(self):
        for i in range(self.P):
            # evaluate the fitness of the frog
            new_frog = Frog(self.dimension)
            new_frog.calc_fitness(self.capacity, self.weight, self.value)
            self.frogs = np.append(self.frogs, new_frog)
        self.best_frog = copy.deepcopy(self.frogs[0])

    def create_x_w_new(self, x_b, x_w):
        x_w_new = copy.deepcopy(x_w)
        for i in range(self.dimension):
            D_i = np.random.rand() * (x_b.x[i] - x_w.x[i])
            if method == 1:
                t = x_w.x[i] + D_i
                x_w_new.x[i] = 0 if t <= 0 else 1 if t >= 1 else round(t)
            elif method == 2:
                t = 1 / (1 + np.exp(-D_i))
                u = np.random.rand()
                x_w_new.x[i] = 0 if t <= u else 1
            else:
                t = 1 / (1 + np.exp(-D_i))
                alpha = 0.4
                x_w_new.x[i] = 0 if t <= alpha else 1 if t >= (1 + alpha) / 2 else x_w.x[i]
        x_w_new.calc_fitness(self.capacity, self.weight, self.value)
        return x_w_new

    def local_search(self, memeplexes):
        x_g = copy.deepcopy(self.best_frog)
        for memeplex in memeplexes:
            for it in range(self.iMax):
                memeplex = sorted(memeplex, key=lambda x: x.fitness, reverse=True)
                x_b = memeplex[0]
                x_w = memeplex[-1]
                # apply eqn. 2, 3 and 6
                x_w_new = self.create_x_w_new(x_b, x_w)
                if x_w_new.fitness > x_w.fitness:
                    memeplex[-1] = x_w_new
                    if x_w_new.fitness > x_g.fitness:
                        x_g = x_w_new
                    continue
                # apply eqn. 2, 3 and 6 with replacing x_b with x_g
                x_w_new = self.create_x_w_new(x_g, x_w)
                if x_w_new.fitness > x_w.fitness:
                    memeplex[-1] = x_w_new
                    if x_w_new.fitness > x_g.fitness:
                        x_g = x_w_new
                    continue
                x_w_new.x = x_w_new.generate_x()
                x_w_new.calc_fitness(self.capacity, self.weight, self.value)
                memeplex[-1] = x_w_new
                if x_w_new.fitness > x_g.fitness:
                    x_g = x_w_new
        if x_g.fitness > self.best_frog.fitness:
            self.best_frog = copy.deepcopy(x_g)
        return memeplexes

    def solve(self) -> Frog:
        # generate population of P frogs randomly
        self.generate_frogs()

        delta = 0  # for termination condition
        for _ in tqdm.tqdm(range(self.iMax)):
            former_best = copy.deepcopy(self.best_frog)
            # sort P frogs in descending order
            self.frogs = sorted(self.frogs, key=lambda x: x.fitness, reverse=True)
            # partition P frogs into m memeplexes
            cur_memeplexes = [[] for i in range(self.m)]
            for i, frog in enumerate(self.frogs):
                cur_memeplexes[i % self.m].append((frog))
            cur_memeplexes = np.array(cur_memeplexes)
            # local search
            cur_memeplexes = self.local_search(cur_memeplexes)  # -> shuffled the m memeplexed
            self.frogs = cur_memeplexes.flatten()
            np.random.shuffle(self.frogs)
            # apply mutation on the population
            for frog in self.frogs:
                frog.mutation(self.p_m)
                frog.calc_fitness(self.capacity, self.weight, self.value)
                if frog.fitness > self.best_frog.fitness:
                    self.best_frog = copy.deepcopy(frog)
            if former_best == self.best_frog:
                delta += 1
            if delta >= self.iMax // 10:
                break

        # determine the best solution
        self.frogs = sorted(self.frogs, key=lambda x: x.fitness, reverse=True)
        return self.best_frog


def main():
    problem = "p08"
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
    print(f"solved: {all(result.x == ans)}")
    print(f"result:   x = {result.x.tolist()}, value = {result.fitness}")
    w = 0
    v = 0
    for j in range(len(ans)):
        if ans[j]:
            w += weight[j]
            v += value[j]
    print(f"expected: x = {ans}, value = {v}")


if __name__ == "__main__":
    main()
