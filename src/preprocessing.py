import pandas as pd

def clean_data(df):
    df = df.dropna()
    return df


def select_numeric_features(df):
    X = df.drop("TARGET", axis=1)
    X = X.select_dtypes(include=["int64", "float64"])
    return X