import numpy as np
import pandas as pd

class DecisionTree:

    def fit(self, data, target):
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist()
        self.independent.remove(target)

    def predict(self, data):
        return np.array([self.__flow_data_tru_tree(row) for row in data.values])

    def __flow_data_tru_tree(self, row):
        return self.data[self.target].value_counts().apply(lambda x: x/len(self.data)).tolist()




