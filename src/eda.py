import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def load_data(path=None):

    if path is None:
        path = BASE_DIR / "data" / "application_train.csv"

    return pd.read_csv(path)


def get_missing_values(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    return pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_percent
    }).sort_values(by="Missing %", ascending=False)


def plot_target_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x="TARGET", data=df, ax=ax)
    ax.set_title("Loan Default Distribution")
    return fig


def get_correlations(df):
    return df.corr(numeric_only=True)["TARGET"].sort_values()