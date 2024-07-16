import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data/wine_data_clean.csv')
counts = df[['country', 'variety']].groupby('variety').count()
varieties = counts[counts.country >= 500].index
df = df[df.variety.isin(varieties)]

# Encode labels
le = LabelEncoder()
df['variety'] = le.fit_transform(df['variety'])

df.drop(['points', 'price', 'country', 'neg', 'neu', 'pos', 'compound'],
        axis=1, inplace=True)

# Set classifier
sgdc = SGDClassifier()

# Prepare parameters
SGDParamGrid = {"max_iter": [100, 1000],
                "loss": ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
                "penalty": ['l2', 'l1']}

# Create GridSearch Object
SGDGrid = GridSearchCV(sgdc, SGDParamGrid)

# Prepare data
X, Y = df.clean_description, df['variety']
vectorizer = TfidfVectorizer()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Adjust and find best parameters
SGDGrid.fit(X_train, Y_train)

print("Best parameters combination for SGD found:")
best_parameters = SGDGrid.best_estimator_.get_params()
for param_name in sorted(SGDParamGrid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

test_accuracy = SGDGrid.score(X_test, Y_test)
print(
    "Accuracy of the best parameters for SGD using the inner CV of "
    f"the grid search: {SGDGrid.best_score_:.3f}"
)
print(f"Accuracy on test set: {test_accuracy:.3f}")


