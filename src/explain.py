import shap


def get_shap_values(model, X_sample):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values