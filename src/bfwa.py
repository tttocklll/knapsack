from __future__ import annotations
import numpy as np
import copy
import tqdm


class Firework:
    def __init__(self, D: int):
        self.D = D
        self.x = self.generate_x()
        self.gorgeous_degree = 0

    def generate_x(self):
        return np.random.randint(0, 2, self.D)

    def calc_gorgeous_degree(self, capacity: int, weight: list[int], value: list[int]):
        cur_weight = 0
        cur_value = 0
        for i in range(self.D):
            if self.x[i]:
                cur_weight += weight[i]
                cur_value += value[i]
        self.gorgeous_degree = cur_value if cur_weight <= capacity else 0


class BFWA:
    def __init__(self, capacity: int, weight: list[int], value: list[int]):
        self.capacity = capacity
        self.weight = weight
        self.value = value
        self.D = len(weight)  # dimension
        # parameters
        self.n = 2 * self.D  # number of fireworks
        self.T_max = 4 * self.D  # max of iterations
        self.A_c = self.D  # max explosion amplitude
        # for result
        self.best_firework = None

    def solve(self) -> Firework:
        # initialize items of fireworks I(x) and its gorgeous degree f(I(x))
        self.generate_fireworks()

        for _ in tqdm.tqdm(range(self.T_max)):
            # determine number of sparks N_i
            self.calc_number_of_sparks()
            # determine fireworks explosion step Step_ie
            self.calc_fireworks_step()
            # carry out fireworks explosion
            self.fireworks_explosion()
            # determine mutation space of fireworks M_im
            self.calc_mutation_space()
            # determine mutation explosion step Step_im
            self.calc_mutation_step()
            # carry out mutation explosion
            self.mutation_explosion()
            # calculate gorgeous degree and select explosion location
            self.select_next_gen()

        return self.best_firework


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
    bfwa = BFWA(capacity, weight, value)
    result = bfwa.solve()
    print(f"solved: {all(result.x == ans)}")
    print(f"result:   x = {result.x.tolist()}, value = {result.gorgeous_degree}")
    w = 0
    v = 0
    for j in range(len(ans)):
        if ans[j]:
            w += weight[j]
            v += value[j]
    print(f"expected: x = {ans}, value = {v}")


if __name__ == "__main__":
    main()