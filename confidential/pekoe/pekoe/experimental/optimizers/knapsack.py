from typing import Any, Dict

import numpy as np


def knapsack_approx(capacity: int, weights: Dict[Any, int]) -> Dict[Any, int]:
    # values == weights
    res = {}
    for i, w in weights.items():
        if w <= capacity:
            res[i] = w
            capacity -= w
        elif res:
            wdiff = w - np.array(list(res.values()))
            wdiff[wdiff > capacity] = -1  # not ok
            j = wdiff.argmax()
            wdiff = wdiff[j]
            if wdiff > 0:
                capacity -= wdiff
                j = list(res.keys())[j]
                del res[j]
                res[i] = w
    return res
