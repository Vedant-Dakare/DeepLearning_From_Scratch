import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


## loading the word index and reverse it
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')


## preprocessing the user input 
## function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3 , '?') for i in encoded_review])   ## convert integer index back to word

## function to preprocess the user input
def preprocess_text(text):
    words = text.lower().split()        ## lowering the user i/p
    encoded_review = [word_index.get(word,2) + 3 for word in words]  ## Convert words into integer indices
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)     ## add padding 
    return padded_review


## prediction function 
def prediction_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment,prediction[0][0]

import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")


user_input = st.text_area("Movie Review", "Type your review here...")
if st.button("Predict Sentiment"):

    preprocessed_input = preprocess_text(user_input)

    prediction = model.predict(preprocessed_input)

    sentiment  = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
else :
    st.write("Please enter a movie review and click the 'Predict Sentiment' button to see the result.") 