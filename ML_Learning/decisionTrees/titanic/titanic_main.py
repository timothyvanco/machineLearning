import numpy as np
import pandas as pd
from titanic_decision_tree import DecisionTree

train = pd.read_csv("train_preprocessed.csv")
test = pd.read.csv("test_preprocessed.csv")

model = DecisionTree()

model.fit(data = train, target = "Survived")

predictions = model.predict(test)
predictions[:5]