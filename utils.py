import pandas as pandy
import numpy as numpie
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense, Embedding , Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import CategoricalAccuracy 

inputto = 120
tokenizer = Tokenizer()

def load_text_data(filepath):
    data = pandy.read_csv(filepath)
    text = data['text']
    return text

def tokenize_texts(texts):
    tokenizer.fit_on_texts(texts)
    wrapper = tokenizer.texts_to_sequences(texts)
    return wrapper

def encode_characters(texts):
    texts = texts.apply(lambda x: [ele for ele in x])
    chars = sorted(set(texts.explode()))
    char_indices = dict((char, chars.index(char)+1) for char in chars)
    wrapper = []
    for text in texts:
        varies = numpie.array([char_indices[char] for char in text])
        wrapper.append(varies)
    return wrapper, char_indices

def _windowize_data(data, inputto):
    data = numpie.array(data)
    nooflstm = len(data) - inputto
    y = data[inputto:]
    indices = numpie.arange(inputto) + numpie.arange(nooflstm)[:, None]
    x = data[indices]
    return x, y


def create_model(calcword, inputto):
    optimizer = keras.optimizers.Adam(learning_rate=.0001)
    model = Sequential()
    model.add(Embedding(calcword, 128,input_length=inputto))
    model.add(LSTM(128, input_shape=(inputto,1), return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(128))
    model.add(Dropout(.2))
    model.add(Dense(calcword, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
    return model



def generate_poetry_words(textive, number_of_sentences, n_lines, model):
  for i in range(n_lines):
    text = []
    for _ in range(number_of_sentences):
      varies = tokenizer.texts_to_sequences([textive])
      varies = pad_sequences(varies, maxlen=20, padding='pre')

      predictwordy = numpie.argmax(model.predict(varies), axis=-1)

      wordspred = ""
      for word, index in tokenizer.word_index.items():
        if index == predictwordy:
          wordspred = word
          break

      textive = textive + ' ' + wordspred
      text.append(wordspred)

    textive = text[-1]
    text = ' '.join(text)
    print(text)
    
def generate_poetry_characters(textive, number_of_sentences, n_lines, model, char_indices):
  for i in range(n_lines):
    text = []
    for _ in range(number_of_sentences):
      seed = [char for char in textive]
      seqgenrandom = [char_indices[char] for char in seed]
      if len(seqgenrandom) < 100:
        for j in range(99-len(seqgenrandom)):
          seqgenrandom.insert(0,0)

      predictwordy = numpie.argmax(model.predict([seqgenrandom]), axis=-1)

      gen_words = ""
      for character, index in char_indices.items():
        if index == predictwordy:
          gen_words = character
          break

      textive = textive + gen_words
      text.append(gen_words)

    textive = text[-1]
    text = ''.join(text)
    print(text)

if __name__ == '__main__':
    data = load_text_data()
    wrapper = tokenize_texts(data)
    X,Y = _windowize_data(wrapper[0], inputto)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)
    calcword = len(tokenizer.word_index) + 1
    model = create_model(calcword, inputto)

