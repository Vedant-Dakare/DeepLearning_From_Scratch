import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_pred_LSTM.h5')


with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


## predict next word function 
## function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        ## this make sure that token list must be less than max sequence len otherwise it just
        #  keep the latest word context and ignore the rest of the context
        token_list = token_list[-(max_sequence_len-1):]     

    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')    ## pre padding is imp
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)[0]   ## we take the word with the highest prob as to be pred word index
    for word,index in tokenizer.word_index.items(): ## converting back to word from index
        if index == predicted_word_index:
            return word
    return None
        

## streamlit app

st.title('Predict Next Word with LSTM')
input_text = st.text_input("Enter the sequence of words","To be or not to be")
if st.button("Predict next word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next word : {next_word}')