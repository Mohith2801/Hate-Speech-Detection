import streamlit as st
import pickle
import re
import string
import nltk

# Download NLTK stopwords data. This needs to be done on every run
# on temporary file systems like Streamlit Cloud.
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
try:
    with open("cv.pkl", "rb") as f:
        cv = pickle.load(f)
    with open("clf.pkl", "rb") as f:
        clf = pickle.load(f)
except FileNotFoundError:
    st.error("Model files (cv.pkl or clf.pkl) not found. Please ensure they are in the same directory as app.py.")
    st.stop()

st.title("Hate Speech Detection")

user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if user_input:
        cleaned_input = clean_data(user_input)
        vectorized_input = cv.transform([cleaned_input])
        prediction = clf.predict(vectorized_input)[0]
        st.write("Prediction:")
        st.write(prediction)
    else:
        st.write("Please enter some text to analyze.")