import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

from src.eda import (
    load_data,
    get_missing_values,
    plot_target_distribution,
    get_correlations
)

from src.model import train_models
from src.business_cost import calculate_cost
from src.score import probability_to_score

# NEW IMPORTS
import numpy as np

# -------------------------
# CONFIG
# -------------------------

st.set_page_config(page_title="Loan Risk System", layout="wide")

# -------------------------
# LOAD DATA
# -------------------------

df = load_data()

# -------------------------
# TRAIN MODELS ONCE
# -------------------------

models = train_models(df)

y_test = models["y_test"]
X_test = models["X_test"]

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def find_best_threshold(y_test, y_probs, fp_cost, fn_cost):
    best_t = 0
    best_cost = float("inf")

    for t in np.arange(0, 1, 0.01):
        y_pred = (y_probs >= t).astype(int)

        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        cost = (fp * fp_cost) + (fn * fn_cost)

        if cost < best_cost:
            best_cost = cost
            best_t = t

    return best_t, best_cost


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


def get_feature_importance(model, X):
    importance = model.get_feature_importance()

    return sorted(
        zip(X.columns, importance),
        key=lambda x: x[1],
        reverse=True
    )

# -------------------------
# SIDEBAR
# -------------------------

page = st.sidebar.radio(
    "Navigation",
    ["EDA", "Data Overview", "Model", "Business Cost", "Risk Scoring", "Feature Importance"]
)

# =========================================================
# EDA
# =========================================================

if page == "EDA":

    st.title("Exploratory Data Analysis")

    st.subheader("Target Distribution")
    st.pyplot(plot_target_distribution(df))

    st.subheader("Missing Values")
    st.dataframe(get_missing_values(df).head(20))

    st.subheader("Correlation")
    st.dataframe(get_correlations(df).tail(10))

# =========================================================
# DATA OVERVIEW
# =========================================================

elif page == "Data Overview":

    st.title("Dataset Overview")

    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

    st.dataframe(df.head())

# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model":

    st.title("Model Comparison")

    lr_acc = accuracy_score(y_test, (models["lr_probs"] > 0.5).astype(int))
    cb_acc = accuracy_score(y_test, (models["cb_probs"] > 0.5).astype(int))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")
        st.metric("Accuracy", round(lr_acc, 4))

    with col2:
        st.subheader("CatBoost")
        st.metric("Accuracy", round(cb_acc, 4))

# =========================================================
# BUSINESS COST (WITH AUTO OPTIMIZATION)
# =========================================================

elif page == "Business Cost":

    st.title("Cost Optimization Engine")

    fp_cost = st.number_input("False Positive Cost", value=1000)
    fn_cost = st.number_input("False Negative Cost", value=10000)

    y_probs = models["cb_probs"]

    # AUTO BEST THRESHOLD
    best_t, best_cost = find_best_threshold(
        y_test,
        y_probs,
        fp_cost,
        fn_cost
    )

    st.success(f"Best Threshold Found: {round(best_t, 2)}")
    st.success(f"Minimum Cost: ${best_cost:,.0f}")

    threshold = st.slider(
        "Manual Threshold Override",
        0.0, 1.0, float(best_t), 0.01
    )

    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_cost = (fp * fp_cost) + (fn * fn_cost)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("TN", tn)
    col2.metric("FP", fp)
    col3.metric("FN", fn)
    col4.metric("Cost", f"${total_cost:,}")

# =========================================================
# RISK SCORING
# =========================================================

elif page == "Risk Scoring":

    st.title("Credit Risk Scoring System")

    idx = st.slider("Select Customer", 0, len(X_test)-1, 0)

    prob = models["cb_probs"][idx]
    score = probability_to_score(prob)

    band = get_risk_band(score)

    decision = "APPROVE" if score > 650 else "REJECT"

    st.metric("Default Probability", round(prob, 4))
    st.metric("Credit Score", score)

    st.success(f"Risk Band: {band}")

    if decision == "APPROVE":
        st.success("Decision: APPROVE")
    else:
        st.error("Decision: REJECT")

    st.subheader("Customer Data")
    st.dataframe(X_test.iloc[[idx]])

# =========================================================
# FEATURE IMPORTANCE (NEW)
# =========================================================

elif page == "Feature Importance":

    st.title("Model Explainability")

    model = models["cb_model"]

    importance = get_feature_importance(model, X_test)

    st.subheader("Top Features Driving Loan Default")

    st.dataframe(importance[:20])

    fig, ax = plt.subplots()
    sns.barplot(
        x=[x[1] for x in importance[:15]],
        y=[x[0] for x in importance[:15]],
        ax=ax
    )

    st.pyplot(fig)