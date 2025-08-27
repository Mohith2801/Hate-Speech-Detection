import streamlit as st
import pickle
import re
import string
import nltk
import os
import tempfile

# Function to safely download NLTK data to a temporary directory
def download_nltk_data_if_not_present(data_package):
    try:
        # Try to find the data package
        nltk.data.find(f"corpora/{data_package}")
    except nltk.downloader.DownloadError:
        # If not found, download to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            nltk.data.path.append(tmpdir)
            nltk.download(data_package, download_dir=tmpdir)

# Download the 'stopwords' data package
download_nltk_data_if_not_present('stopwords')
from nltk.corpus import stopwords

# The rest of your code remains the same
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

if st.button("Analyze"):
    if user_input:
        cleaned_input = clean_data(user_input)
        vectorized_input = cv.transform([cleaned_input])
        prediction = clf.predict(vectorized_input)[0]
        st.write("Prediction:")
        st.write(prediction)
    else:
        st.write("Please enter some text to analyze.")