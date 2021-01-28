from bfwa import BFWA
from sfla import MDSFLA
from statistics import mean, stdev
import time


def main():
    n = 30
    now = int(time.time())
    with open(f"../log/log_{now}.csv", "a") as log:
        log.write("problem, best, worst, average, std, data\n")
    for i in range(1, 9):
        problem = f"p0{i}"
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

        # carry out test
        result_bfwa = []
        result_sfla = []
        for _ in range(n):
            cur_bfwa = BFWA(capacity, weight, value)
            result = cur_bfwa.solve()
            result_bfwa.append(result.gorgeous_degree)
        with open(f"../log/log_{now}.csv", "a") as log:
            log.write(f"{problem}, {max(result_bfwa)}, {min(result_bfwa)}, {mean(result_bfwa)}, {stdev(result_bfwa)}, {', '.join(map(str, result_bfwa))}\n")
        for _ in range(n):
            cur_sfla = MDSFLA(capacity, weight, value)
            result = cur_sfla.solve()
            result_sfla.append(result.fitness)
        with open(f"../log/log_{now}.csv", "a") as log:
            log.write(f"{problem}, {max(result_sfla)}, {min(result_sfla)}, {mean(result_sfla)}, {stdev(result_sfla)}, {', '.join(map(str, result_sfla))}\n")


if __name__ == "__main__":
    main()
