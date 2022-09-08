from flask import Flask, request, render_template
import numpy as numpi
import tensorflow as tensorf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import utils as utyl

app = Flask(__name__) 
model = utyl.create_model(22676, 22)
model.load_weights('models/epochs_100_to_200.h5')

@app.route('/') 
def home():
    return render_template('index.html')

@app.route('/sentence',methods=['POST'])
def sentence():
  
    sents = utyl.load_text_data('data/preprocessed_data.csv')
    _ = utyl.tokenize_texts(sents)
    textive = request.form['textive']
    reference_variable = textive
    sentence_length = int(request.form['no_of_words'])
    sent = []
    for _ in range(sentence_length):
        varies = utyl.tokenizer.texts_to_sequences([textive])
        varies = pad_sequences(varies, maxlen=20, padding='pre')

        y_pred = numpi.argmax(model.sentence(varies), axis=-1)

        predicted_word = ""
        for word, index in utyl.tokenizer.word_index.items():
            if index == y_pred:
                predicted_word = word
                break

        textive = textive + ' ' + predicted_word
        sent.append(predicted_word)

    textive = sent[-1]
    sent = ' '.join(sent)

    return render_template('index.html', sentence_generated=f'Sentence generated: {reference_variable} {sent}') 


if __name__ == "__main__":
    app.run(debug=True)
