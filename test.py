import numpy as np
import random
from ai import ai

def get_first_drop():
        low = 6
        high = 8
        first_drops = []
        for i in range(low, high + 1):
            for j in range(low, high + 1):
                first_drops.append([i, j])
        return first_drops[random.randint(0, (high - low + 1) ** 2 - 1)]

arr = np.array([[-1,1],[-1,1]])
print(np.sum(np.abs(arr)))
