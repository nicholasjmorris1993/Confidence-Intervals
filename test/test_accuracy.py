import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("/home/nick/Confidence-Intervals/src")
from accuracy import confidence_interval


data = pd.read_csv("/home/nick/Confidence-Intervals/test/LungCap.csv")
data["LungCap"] = pd.cut(data["LungCap"], bins=3)  # convert output to categorical

# encode categories into integer labels
labeler = LabelEncoder()
data["LungCap"] = labeler.fit_transform(data["LungCap"])

data = data.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
train = data.copy().head(int(len(data)*0.5))
test = data.copy().tail(int(len(data)*0.5))

X = train.copy().drop(columns="LungCap")
Y = train.copy()[["LungCap"]]

model = XGBClassifier(
    booster="gbtree",
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=7,
    min_child_weight=1,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
)
model.fit(X, Y)

X = test.copy().drop(columns="LungCap")
Y = test.copy()[["LungCap"]]

y_pred = model.predict(X)
y_true = Y.to_numpy().ravel()
predictions = pd.DataFrame({
    "Predicted": y_pred,
    "Actual": y_true,
})

interval = confidence_interval(
    actual=predictions["Actual"], 
    predict=predictions["Predicted"],
)

print(interval.interval)
