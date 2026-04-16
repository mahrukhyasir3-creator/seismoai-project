
import shap
import matplotlib.pyplot as plt
import numpy as np


def compute_shap_values(model, X_train, X_sample):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_sample)
    return shap_values


def plot_feature_importance(shap_values):
    """
    Plot SHAP feature importance safely.
    """
    try:
        shap.plots.bar(shap_values[:, :, 1], max_display=10)
    except:
        shap.plots.bar(shap_values, max_display=10)

    plt.show()


def explain_prediction(prediction):
    if prediction == 1:
        return "Predicted as NOISY trace due to abnormal seismic feature patterns."
    else:
        return "Predicted as CLEAN trace due to normal seismic feature behavior."
