def get_risk_band(score):

    if score >= 800:
        return "AAA (Excellent)"
    elif score >= 700:
        return "AA (Very Good)"
    elif score >= 600:
        return "A (Good)"
    elif score >= 500:
        return "B (Moderate)"
    elif score >= 400:
        return "C (High Risk)"
    else:
        return "D (Very High Risk)"