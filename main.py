import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# load the pre trained model
model = load_model("simple_rnn_model_imdb.h5")


# helper functions
# function to decode reviews
def process_input(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# function to predict sentiment
def predict_sentiment(review):
    processed_review = process_input(review)
    prediction = model.predict(processed_review)

    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment, prediction[0][0]


## streamlit app

import streamlit as st

st.title("IMDB Sentiment Analysis")
st.write(
    "This is a simple example of a Streamlit app that uses a pre-trained model to predict the sentiment of an IMDB review"
)

# user input
user_input = st.text_area("Enter your review")

if st.button("Predict"):
    preprocess_input = process_input(user_input)
    # make predict
    prediction = model.predict(preprocess_input)
    sentiment = "positive review" if prediction[0][0] > 0.5 else "negative review"

    # display the result
    st.write(f"The sentiment of the review is: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")
else:
    st.write("Please enter a review for prediction")
