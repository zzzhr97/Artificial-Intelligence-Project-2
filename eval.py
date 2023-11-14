import numpy as np
from ai import State, INF_VALUE

def evaluate_func(state, **kwargs):
    # TODO: implement evaluate function
    return INF_VALUE[-state.color] // 2