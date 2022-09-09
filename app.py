from flask import Flask, request, render_template
import numpy as numpie
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import utils as ut

app = Flask(__name__) 
model = ut.create_model(24775, 20)
model.load_weights('models/epochs_100_to_200.h5')

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
 
    
    texts = ut.file_pandas('data/preprocessed_data.csv')
    _ = ut.split_text(texts)
    textive = request.form['textive']
    texton = textive
    number_of_sentences = int(request.form['input_length'])
    text = []
    for _ in range(number_of_sentences):
        varies = ut.tokenizer.texts_to_sequences([textive])
        varies = pad_sequences(varies, maxlen=20, padding='pre')

        predictwordy = numpie.argmax(model.predict(varies), axis=-1)

        wordspred = ""
        for word, index in ut.tokenizer.word_index.items():
            if index == predictwordy:
                wordspred = word
                break

        textive = textive + ' ' + wordspred
        text.append(wordspred)

    textive = text[-1]
    text = ' '.join(text)

    return render_template('index.html', text_to_generate=f'Sentence generated: {texton} {text}') # rendering the predicted result


if __name__ == "__main__":
    app.run(debug=True)
