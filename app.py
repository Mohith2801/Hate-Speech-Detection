# app.py
import streamlit as st
import pickle
import re
import string
import nltk

# Download stopwords if not already present
nltk.download('stopwords')
from nltk.corpus import stopwords

stopword = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Load vectorizer and classifier
with open("cv.pkl", "rb") as f:
    cv = pickle.load(f)
with open("clf.pkl", "rb") as f:
    clf = pickle.load(f)

st.title("Hate Speech Detection")

user_input = st.text_area("Enter text to analyze:")

if st.button("Detect"):
    cleaned = clean_data(user_input)
    vectorized = cv.transform([cleaned]).toarray()
    prediction = clf.predict(vectorized)[0]
    st.write(f"Prediction: **{prediction}**")