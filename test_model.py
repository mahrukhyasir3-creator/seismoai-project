
import numpy as np
import seismoai_model

traces = np.random.randn(50, 4001)

X = seismoai_model.extract_features(traces)

assert X.shape[1] == 5

y = np.random.randint(0, 2, 50)

model = seismoai_model.train_classifier(X, y)

preds = seismoai_model.predict_traces(model, X[:5])

assert len(preds) == 5

print("All Tests Passed")
