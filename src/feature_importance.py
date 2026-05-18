import pandas as pd

def get_feature_importance(model, X):

    importance = model.get_feature_importance()

    return pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)