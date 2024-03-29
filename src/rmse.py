import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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

        self.rmse = []
        np.random.seed(0)
        seeds = np.random.random_integers(low=0, high=1e6, size=1000)

        for i in range(1000):
            sample = df.sample(frac=0.5, replace=True, random_state=seeds[i])
            self.rmse.append(mean_squared_error(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
                squared=False,
            ))
    
    def score(self):
        alpha = 0.95
        p = ((1 - alpha) / 2) * 100
        lower = np.percentile(self.rmse, p)
        p = (alpha + ((1 - alpha) / 2)) * 100
        upper = np.percentile(self.rmse, p)
        mean = np.mean(self.rmse)

        self.interval = pd.DataFrame({
            "95 Percent Lower Confidence": [lower],
            "Expected RMSE": [mean],
            "95 Percent Upper Confidence": [upper],
        })

