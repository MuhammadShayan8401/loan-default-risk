import numpy as np

def find_best_threshold(y_test, y_probs, fp_cost, fn_cost):

    best_threshold = 0
    best_cost = float("inf")

    for t in np.arange(0, 1, 0.01):

        y_pred = (y_probs >= t).astype(int)

        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        cost = (fp * fp_cost) + (fn * fn_cost)

        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return best_threshold, best_cost