# Hate Speech Detection

This project uses machine learning to detect hate speech and offensive language in tweets. It includes a Jupyter Notebook for training and a Streamlit app for interactive prediction.

## Features

- Data cleaning and preprocessing (stopwords removal, stemming)
- Model training using Decision Tree Classifier
- Real-time prediction via Streamlit web app
- Easy deployment with pickle files

## Files

- `HateSpeechDetection (1).ipynb` — Jupyter Notebook for data analysis and model training
- `app.py` — Streamlit app for prediction
- `cv.pkl` — Saved CountVectorizer
- `clf.pkl` — Saved trained classifier
- `twitter_data.csv` — Dataset
- `requirements.txt` — Python dependencies

## How to Run Locally

1. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

2. **Train the model (optional):**
    - Open the notebook and run all cells to retrain and save `cv.pkl` and `clf.pkl`.

3. **Start the Streamlit app:**
    ```
    streamlit run app.py
    ```

## How to Deploy on Streamlit Cloud

- Upload all files (`app.py`, `cv.pkl`, `clf.pkl`, `requirements.txt`) to your Streamlit Cloud project.
- The app will be available online for predictions.

## Usage

- Enter any text in the Streamlit app to check if it contains hate speech or offensive language.

## Live Demo

Try the app online: [Hate Speech Detection Streamlit App](https://hate-speech-detection-9idg8m3au5tdta8n4oedjk.streamlit.app/)

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- nltk
- streamlit

## License

This project is for educational purpose
