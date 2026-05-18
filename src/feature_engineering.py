def create_features(df):

    df = df.copy()

    # Example engineered features (safe placeholders)
    if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
        df["INCOME_TO_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1)

    if "DAYS_BIRTH" in df.columns:
        df["AGE"] = df["DAYS_BIRTH"] / -365

    return df