import random
from copy import deepcopy

from numpy.random import randint
import numpy as np


def fitness(x): return sum(x)


def makeWheel(population, scores):
    scores = np.array(scores)
    pos_scores = max(scores) - scores + 1
    wheel = []
    total = sum(pos_scores)
    top = 0
    for i, p in enumerate(population):
        f = pos_scores[i] / total
        wheel.append((top, top + f, p))
        top += f
    return wheel


def binSearch(wheel, num):
    mid = len(wheel) // 2
    low, high, answer = wheel[mid]
    if low <= num <= high:
        return deepcopy(answer)
    elif low < num:
        return binSearch(wheel[mid + 1:], num)
    else:
        return binSearch(wheel[:mid], num)


def select_wheel_sus(wheel, N):
    stepSize = 1.0 / N
    answer = []
    r = random.random()
    answer.append(binSearch(wheel, r))
    while len(answer) < N:
        r += stepSize
        if r > 1:
            r %= 1
        answer.append(binSearch(wheel, r))
    return answer


def select_wheel_roullete(wheel, N):
    answer = []
    while len(answer) <= N:
        r = random.random()
        answer.append(binSearch(wheel, r))
    return answer


def tournament(pop, scores, k=5):
    # selección aleatoria
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # realizar un torneo
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return deepcopy(pop[selection_ix])


def roullete(pop, scores):
    scores = np.array(scores)
    pos_scores = max(scores) - scores + 1  # 1 / scores + 1
    p = np.random.uniform(0, sum(pos_scores))
    for i, f in enumerate(pos_scores):
        if p <= 0:
            break
        p -= f
    return deepcopy(pop[i])


def selection_sus(pop, scores):
    wheel = makeWheel(pop, scores)
    return select_wheel_sus(wheel, len(pop))


def selection_tournament(pop, scores, tour_size=5):
    return [tournament(pop, scores, tour_size) for _ in range(len(pop))]


def selection_roullete(pop, scores):
    wheel = makeWheel(pop, scores)
    return select_wheel_roullete(wheel, len(pop))


if __name__ == '__main__':
    pop = [(random.random(), random.random()) for _ in range(10)]
    scores = [fitness(p) for p in pop]

    print('-- sus --')
    print(sum(scores))
    selection = selection_sus(pop, scores)
    print(sum([fitness(p) for p in selection]))

    print('-- tournament --')
    print(sum(scores))
    selection = selection_tournament(pop, scores)
    print(sum([fitness(p) for p in selection]))

    print('-- roullete --')
    print(sum(scores))
    selection = selection_roullete(pop, scores)
    print(sum([fitness(p) for p in selection]))