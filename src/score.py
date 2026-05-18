def probability_to_score(prob):

    # Convert probability to credit score (0–1000)
    score = (1 - prob) * 1000

    return round(score, 0)