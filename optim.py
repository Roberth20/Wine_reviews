from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Prepare data
df = pd.read_csv('data/wine_data_clean.csv')
counts = df[['country', 'variety']].groupby('variety').count()
varieties = counts[counts.country >= 1000].index
df = df[df.variety.isin(varieties)]

df.drop(['points', 'price', 'country', 'neg', 'neu', 'pos', 'compound'],
        axis=1, inplace=True)

# Encode labels
le = LabelEncoder()
df['variety'] = le.fit_transform(df['variety'])

X, Y = df.clean_description, df['variety']

# Explore all classifiers
classifiers = {
'SGD': SGDClassifier(),
'Perceptron': Perceptron(),
'Passive-Aggressive': PassiveAggressiveClassifier(),
'RidgeClassifier': RidgeClassifier(),
'RadomForestClassifier': RandomForestClassifier()
}
vectorizer = TfidfVectorizer()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
for cls_name, cls in classifiers.items():
    cls.fit(X_train, Y_train)
    print(f'{cls_name}')
    print(f'Accuracy: {cls.score(X_test, Y_test)}\n')
    
