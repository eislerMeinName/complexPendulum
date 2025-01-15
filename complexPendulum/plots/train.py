import os
import time
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt

import csv


def smooth(scalars: List[float], weight) -> List[float]:
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

def read(name: str, max: float):
    step: List[float] = []
    rew: List[float] = []
    factor = 1
    if 'Baselined' in name:
        factor = 100
    with open(name, newline='') as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:
            #debug(bcolors.OKBLUE, str(row['Step']) + str(row['Value']))
            if float(row['Step']) > max:
                break
            step.append(float(row['Step']))
            rew.append(float(row['Value']) / factor)

    return step, rew


if __name__ == "__main__":
    plt.rc('font', size=20)
    #plt.grid()
    name: List[str] = ['../data/training/setup1/Direct.csv', '../data/training/setup1/Gain.csv', '../data/training/setup1/Baselined.csv']
    step, rew = read(name[0], 200000)
    smoothed = smooth(rew, 0.5)
    plt.plot(step, rew, color='tab:blue', alpha=0.25, label='Direct')
    plt.plot(step, smoothed, color='tab:blue', label='Direct Smoothed')

    step, rew = read(name[1], 3e7)
    smoothed = smooth(rew, 0.5)
    plt.plot(step, rew, color='tab:orange', alpha=0.25, label='Gain')
    plt.plot(step, smoothed, color='tab:orange', label='Gain Smoothed')

    step, rew = read(name[2], 3e7)
    smoothed = smooth(rew, 0.5)
    plt.plot(step, rew, color='tab:purple', alpha=0.25, label='Baselined')
    plt.plot(step, smoothed, color='tab:purple', label='Baselined Smoothed')

    plt.yscale('symlog')
    #plt.ylim(-1100, 0)
    plt.xlim(0, 200000)
    plt.xlabel('training steps')
    plt.ylabel('return')
    plt.legend()
    plt.xticks([50000, 100000, 150000, 200000])
    plt.show()
