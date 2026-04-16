
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def extract_features(traces):
    """
    Extract 6 statistical features from each seismic trace.
    """
    if traces.ndim != 2:
        raise ValueError(f"Expected 2D array, got {traces.ndim}D")

    out = []

    for tr in traces:
        tr = tr.astype(float)

        std = float(tr.std())

        kurtosis = float(
            (((tr - tr.mean()) / std) ** 4).mean()
        ) if std > 0 else 0.0

        out.append([
            float(abs(tr).mean()),
            std,
            float(abs(tr).max()),
            float((tr ** 2).sum()),
            float((np.diff(np.sign(tr)) != 0).sum()),
            kurtosis
        ])

    return np.array(out)


def train_classifier(traces, labels):
    """
    Train Random Forest classifier.
    """
    if len(traces) != len(labels):
        raise ValueError("traces and labels length mismatch")

    features = extract_features(traces)

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    clf.fit(features, labels)

    report = classification_report(
        labels,
        clf.predict(features),
        zero_division=0
    )

    return {
        "model": clf,
        "classes": list(clf.classes_),
        "report": report
    }


def predict_traces(traces, model_dict):
    """
    Predict labels for traces.
    """
    clf = model_dict["model"]

    features = extract_features(traces)

    preds = clf.predict(features)
    probs = clf.predict_proba(features)

    return preds, probs
