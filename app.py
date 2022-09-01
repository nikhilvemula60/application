from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import utils as ut

app = Flask(__name__) 
model = ut.create_model(24775, 20)
model.load_weights('models/epochs_100_to_200.h5')

@route('/') # Homepage
def home():
    return render_template('index.html')

@route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    texts = ut.load_text_data('data/preprocessed_data.csv')
    _ = ut.tokenize_texts(texts)
    seed_text = request.form['seed_text']
    seed_text_copy = seed_text
    poetry_length = int(request.form['poem_length'])
    text = []
    for _ in range(poetry_length):
        encoded = ut.tokenizer.texts_to_sequences([seed_text])
        encoded = pad_sequences(encoded, maxlen=20, padding='pre')

        y_pred = np.argmax(model.predict(encoded), axis=-1)

        predicted_word = ""
        for word, index in ut.tokenizer.word_index.items():
            if index == y_pred:
                predicted_word = word
                break

        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)

    seed_text = text[-1]
    text = ' '.join(text)

    return render_template('index.html', prediction_text=f'Completed Poem: {seed_text_copy}... {text}') # rendering the predicted result

#var port = process.env.PORT || 3000;

if __name__ == "__main__":
    run(debug=True)
