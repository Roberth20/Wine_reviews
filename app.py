import streamlit as st
import pandas as pd
import pickle
import nltk
import string

df = pd.read_csv('data/vectorized_wine_reviews.csv')

st.title('Wine Prediction App')

st.write('''This app try to predict **wine varities**. \n The data used was *Wine reviews* from *ZACKTHOUTT* after data cleaning, transforming and analysis, I build this AI model. \n \n You only need to type a desription for a wine profile of flavours and smells, and the system will process and mak a prediction for you.''')
        
st.header('Type a description for a wine.')

sample_description = df.clean_description[0]
description = st.text_area("Wine description", sample_description, height=250)
sp = nltk.corpus.stopwords.words('english')

def clean_description(text):
    # Lower text
    text = text.lower()
    # tokenize and remove punctuation
    text = [word.strip(string.punctuation) for word in text.split(' ')]
    # Remove words with numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # Remove stopwords
    text = [word for word in text if word not in sp]
    # Remove empty tokens
    text = [w for w in text if len(w) > 0]
    # Remove words with one letter\n",
    text = [w for w in text if len(w) > 1]
    # Reconstruct the description\n",
    text = " ".join(text)
    return text

description = clean_description(description)

st.write("***")

# Print input description
st.header("Input description")
description

# Prepare data
df = df['clean_description']
df = pd.concat([pd.DataFrame({'clean_description': description}, index=[0]), df], axis=0)
df.reset_index(drop=True, inplace=True)

# Load models
le = pickle.load(open('models/le_variety.pkl', 'rb'))
model = pickle.load(open('models/sgd.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

X = vectorizer.transform(df.iloc[0])

# Make predictions
prediction = model.predict(X)
prediction_proba = model.predict_proba(X)

st.header('Prediction')
st.write(f'We recommend you: {le.inverse_transform(prediction)[0]}')

st.header('Prediction Probabilities')
data_labels = pd.DataFrame(columns=le.classes_, data=prediction_proba)
st.write(data_labels[data_labels > 0].dropna(axis=1))