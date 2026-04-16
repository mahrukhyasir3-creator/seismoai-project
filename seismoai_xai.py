
import shap
import matplotlib.pyplot as plt


def compute_shap_values(model, X_train, X_sample):
    """
    Compute SHAP values.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_sample)
    return shap_values


def plot_feature_importance(shap_values):
    """
    Plot SHAP feature importance.
    """
    try:
        shap.plots.bar(shap_values[:, :, 1], max_display=10)
    except:
        shap.plots.bar(shap_values, max_display=10)

    plt.show()


def explain_prediction(prediction):
    """
    Explain model prediction.
    """
    if prediction == 'noisy':
        return "Trace predicted as NOISY due to abnormal seismic statistics."
    else:
        return "Trace predicted as GOOD due to normal seismic behavior."
