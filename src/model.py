from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


def train_models(df):

    data = df.dropna()

    X = data.drop("TARGET", axis=1)
    y = data["TARGET"]

    X = X.select_dtypes(include=["int64", "float64"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # LOGISTIC REGRESSION
    # -------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    # -------------------------
    # CATBOOST (MAIN MODEL)
    # -------------------------
    cb = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0
    )

    cb.fit(X_train, y_train)
    cb_probs = cb.predict_proba(X_test)[:, 1]

    return {
        "X_test": X_test,
        "y_test": y_test,
        "lr_probs": lr_probs,
        "cb_probs": cb_probs,
        "lr_model": lr,
        "cb_model": cb
    }