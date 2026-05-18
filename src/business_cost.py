from sklearn.metrics import confusion_matrix


def calculate_cost(y_test, y_probs, threshold, fp_cost, fn_cost):

    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_cost = (fp * fp_cost) + (fn * fn_cost)

    return tn, fp, fn, tp, total_cost