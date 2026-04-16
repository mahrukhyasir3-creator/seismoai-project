
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def extract_features(traces):
    """
    Extract statistical features from seismic traces.
    """
    mean_amp = np.mean(traces, axis=1)
    std_amp = np.std(traces, axis=1)
    max_amp = np.max(traces, axis=1)
    min_amp = np.min(traces, axis=1)
    energy = np.sum(traces**2, axis=1)

    features = np.column_stack([
        mean_amp,
        std_amp,
        max_amp,
        min_amp,
        energy
    ])

    return features


def train_classifier(X, y):
    """
    Train Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


def predict_traces(model, X):
    """
    Predict noisy/clean traces.
    """
    return model.predict(X)
