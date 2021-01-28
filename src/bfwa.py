from __future__ import annotations
import numpy as np
import copy
import tqdm
from random import sample
epsilon = 1e-10  # to avoid 0-division error


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
        self.N_m = self.D // 2  # number of mutation explosion
        # firework
        self.fireworks = []
        self.best_firework: Firework = None
        self.worst_firework: Firework = None

    def generate_fireworks(self):
        for _ in range(self.n):
            new_firework = Firework(self.D)
            new_firework.calc_gorgeous_degree(self.capacity, self.weight, self.value)
            self.fireworks.append(new_firework)
            if self.best_firework is None:
                self.best_firework = new_firework
                self.worst_firework = new_firework
            else:
                self.best_firework = self.best_firework if self.best_firework.gorgeous_degree > new_firework.gorgeous_degree else new_firework
                self.worst_firework = self.best_firework if self.best_firework.gorgeous_degree < new_firework.gorgeous_degree else new_firework

    def calc_number_of_sparks(self) -> list[int]:
        def inner(i: int):  # calc f(I_i(x)) - y_min + epsilon
            return self.fireworks[i].gorgeous_degree - self.worst_firework.gorgeous_degree + epsilon

        def calc_each_N_i(i: int, total_inner: float):
            return round(self.n * inner(i) / total_inner)

        total_inner = 0
        for i in range(self.n):
            total_inner += inner(i)

        # calc each N_i
        N_max = 10  # max number of spark
        N_i = []
        for i in range(self.n):
            N_cur = calc_each_N_i(i, total_inner)
            N_i.append(1 if N_cur < 1 else N_max if N_cur > N_max else N_cur)
        return N_i

    def calc_fireworks_step(self) -> list[int]:
        def inner(i: int):
            return self.best_firework.gorgeous_degree - self.fireworks[i].gorgeous_degree + epsilon

        def calc_each_A_i(i: int, total_inner: float):
            A_min = 1
            return A_min + np.floor(self.A_c * inner(i) / total_inner)

        def calc_each_Step_ie(A_i: int):
            return 1 if A_i <= 1 else self.A_c if A_i > self.A_c else np.random.randint(1, A_i + 1)

        total_inner = 0
        for i in range(self.n):
            total_inner += inner(i)

        A_i = [calc_each_A_i(i, total_inner) for i in range(self.n)]
        Step_ie = [calc_each_Step_ie(A_i[i]) for i in range(self.n)]
        return Step_ie

    def explosion_operator(self, I: Firework, S: list[int], r: int) -> Firework:
        spark = copy.deepcopy(I)
        inv_idx = sample(S, r)
        for i in inv_idx:
            spark.x[i] = (spark.x[i] + 1) % 2
        spark.calc_gorgeous_degree(self.capacity, self.weight, self.value)
        return spark

    def calc_mutation_space(self) -> list:
        M_im_list = []
        for firework in self.fireworks:
            M_im_list.append(np.where(firework.x == self.best_firework.x)[0].tolist())
        return M_im_list

    def calc_mutation_step(self) -> list[int]:
        def inner(i: int):
            return self.fireworks[i].gorgeous_degree - self.worst_firework.gorgeous_degree + epsilon

        def calc_each_A_i(i: int, total_inner: float):
            A_min = 1
            return A_min + np.floor(self.A_c * inner(i) / total_inner)

        def calc_each_Step_im(A_i: int):
            return 1 if A_i <= 1 else self.A_c if A_i > self.A_c else np.random.randint(1, A_i + 1)

        total_inner = 0
        for i in range(self.n):
            total_inner += inner(i)

        A_i = [calc_each_A_i(i, total_inner) for i in range(self.n)]
        Step_im = [calc_each_Step_im(A_i[i]) for i in range(self.n)]
        return Step_im

    def distance(self, I_i: Firework, I_j: Firework):
        return np.sum(I_i.x != I_j.x)

    def select_next_gen(self):
        def calc_p(i: int, sum_dist):
            cur_sum = 0
            for j in range(len(self.fireworks)):
                cur_sum += self.distance(self.fireworks[i], self.fireworks[j])
            return cur_sum / sum_dist

        # keep best firework spot
        self.fireworks.sort(key=lambda x: x.gorgeous_degree)
        self.best_firework = self.fireworks[-1]
        self.fireworks.pop()

        # calculate probability of whether I_i should be in next gen
        sum_dist = 0
        for i in range(len(self.fireworks)):
            for j in range(len(self.fireworks)):
                sum_dist += self.distance(self.fireworks[i], self.fireworks[j])
        p_list = []
        for i in range(len(self.fireworks)):
            p_list.append([calc_p(i, sum_dist), self.fireworks[i]])
        p_list.sort(key=lambda x: x[0])
        p_list = np.array(p_list)
        self.fireworks = p_list[-self.n+1:][:, 1].tolist() + [self.best_firework]
        self.fireworks.sort(key=lambda x: x.gorgeous_degree)
        self.worst_firework = self.fireworks[0]

    def solve(self) -> Firework:
        # initialize items of fireworks I(x) and its gorgeous degree f(I(x))
        self.generate_fireworks()

        for _ in tqdm.tqdm(range(self.T_max)):
            # determine number of sparks N_i
            N_i_list = self.calc_number_of_sparks()
            # determine fireworks explosion step Step_ie
            Step_ie_list = self.calc_fireworks_step()
            # carry out fireworks explosion
            M_0 = [i for i in range(self.D)]
            sparks = []
            for i, firework in enumerate(self.fireworks):
                for p in range(N_i_list[i]):
                    sparks.append(self.explosion_operator(firework, M_0, Step_ie_list[i]))
            # determine mutation space of fireworks M_im
            M_im_list = self.calc_mutation_space()
            # determine mutation explosion step Step_im
            Step_im_list = self.calc_mutation_step()
            # carry out mutation explosion
            mutation_sparks = []
            for i, firework in enumerate(self.fireworks):
                for j in range(self.N_m):
                    mutation_sparks.append(self.explosion_operator(firework, M_im_list, Step_im_list[i]))
            self.fireworks += sparks + mutation_sparks
            # calculate gorgeous degree and select explosion location
            self.select_next_gen()

        return self.best_firework


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
