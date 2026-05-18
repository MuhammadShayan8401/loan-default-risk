# 📊 Loan Default Risk with Business Cost Optimization

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-GradientBoosting-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-lightgrey)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet)

---

## 🧠 Project Overview

A real-world **credit risk scoring system** that predicts loan default probability and optimizes lending decisions using **business cost-sensitive machine learning**.

Instead of focusing only on accuracy, this system minimizes **financial loss** by balancing false approvals and false rejections.

---

## 🎯 Objective

* Predict loan default probability
* Optimize decision threshold using cost-benefit analysis
* Reduce total financial risk for lending institutions

---

## ⚙️ Key Features

* 🔍 Exploratory Data Analysis (EDA)
* 🤖 Machine Learning Models (Logistic Regression + CatBoost)
* 💰 Business Cost Optimization (FP vs FN tradeoff)
* 📉 Automatic Threshold Optimization
* 💳 Credit Score Generation (0–1000 scale)
* ⚖️ Risk Band Classification (AAA → D)
* 📊 Feature Importance Analysis
* 🖥️ Interactive Streamlit Dashboard

---

## 💰 Business Logic

```text
Total Cost = (False Positives × FP Cost) + (False Negatives × FN Cost)
```

✔ The system finds the optimal threshold that minimizes this cost.

---

## 🏦 Credit Risk System

| Score Range | Risk Level         |
| ----------- | ------------------ |
| 800+        | AAA (Excellent)    |
| 700–799     | AA (Very Good)     |
| 600–699     | A (Good)           |
| 500–599     | B (Moderate)       |
| 400–499     | C (High Risk)      |
| <400        | D (Very High Risk) |

---

## 📊 Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* CatBoost
* Matplotlib & Seaborn
* Streamlit

---

## 🚀 Outcome

* End-to-end ML pipeline
* Business-driven decision system
* Real-world credit scoring simulation
* Interactive risk dashboard

---

## 📌 Key Learning

* Cost-sensitive machine learning
* Binary classification modeling
* Risk scoring systems
* Model interpretability
* Production-style ML app design

---

## 👨‍💻 About Me

I am a Machine Learning and Data Science enthusiast focused on building real-world AI systems, especially in the FinTech domain.

I enjoy working on end-to-end projects that go beyond notebooks and focus on deployment, business impact, and decision systems.

My interests include:

* Machine Learning & Deep Learning
* Credit Risk Modeling
* Data Science Applications in Finance
* Building interactive AI dashboards

I aim to develop production-ready AI systems that solve real business problems.
