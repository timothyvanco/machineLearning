from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# create a matrix including the data
data = [[8,8,'dog'], [50,40,'dog'], [8,9,'cat'], [15,12,'dog'], [9,9.8,'cat']]

# generating a dataframe from the matrix
df = pd.DataFrame(data, columns = ['weight', 'height', 'label'])

# defining predictors
X = df[['weight', 'height']]

# definig the target variable and mapping it to 1 for dog and 0 for cat
y = df['label'].replace({'dog':1, 'cat':0})

# instantiating the model
tree = DecisionTreeClassifier()

# fitting the model
model = tree.fit(X, y)

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image


dot_data = StringIO()

export_graphviz(
    model,
    out_file = dot_data,
    filled=True, rounded=True, proportion=False,
    special_characters=True,
    feature_names=X.columns,
    class_names=["cat", "dog"]
)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
