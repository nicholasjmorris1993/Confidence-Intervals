import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def confidence_interval(actual, predict):
    model = ConfidenceInterval()
    model.bootstrap(actual, predict)
    model.score()

    return model


class ConfidenceInterval:
    def bootstrap(self, actual, predict):
        df = pd.DataFrame({
            "Actual": actual,
            "Predict": predict,
        })

        self.accuracy = []
        np.random.seed(0)
        seeds = np.random.random_integers(low=0, high=1e6, size=1000)

        for i in range(1000):
            sample = df.sample(frac=0.5, replace=True, random_state=seeds[i])
            self.accuracy.append(accuracy_score(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
            ))
    
    def score(self):
        alpha = 0.95
        p = ((1 - alpha) / 2) * 100
        lower = np.percentile(self.accuracy, p)
        p = (alpha + ((1 - alpha) / 2)) * 100
        upper = np.percentile(self.accuracy, p)
        mean = np.mean(self.accuracy)

        self.interval = pd.DataFrame({
            "95 Percent Lower Confidence": [lower],
            "Expected Accuracy": [mean],
            "95 Percent Upper Confidence": [upper],
        })

