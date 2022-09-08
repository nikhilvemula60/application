import pandas as pd
import numpy as np
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

n_prev = 100
tokenizer = Tokenizer()

def load_text_data(filepath):
    data = pd.read_csv(filepath)
    sent = data['sent']
    return sent

def tokenize_texts(sents):
    tokenizer.fit_on_texts(sents)
    encoded_texts = tokenizer.texts_to_sequences(sents)
    return encoded_texts

def encode_characters(sents):
    sents = sents.apply(lambda x: [ele for ele in x])
    chars = sorted(set(sents.explode()))
    char_indices = dict((char, chars.index(char)+1) for char in chars)
    encoded_texts = []
    for sent in sents:
        varies = np.array([char_indices[char] for char in sent])
        encoded_texts.append(varies)
    return encoded_texts, char_indices

def _windowize_data(data, n_prev):
    data = np.array(data)
    n_predictions = len(data) - n_prev
    y = data[n_prev:]
    indices = np.arange(n_prev) + np.arange(n_predictions)[:, None]
    x = data[indices]
    return x, y


def create_model(num_words, n_prev):
    optimizer = keras.optimizers.Adam(learning_rate=.0001)
    model = Sequential()
    model.add(Embedding(num_words, 128, input_length=n_prev))
    model.add(LSTM(128, input_shape=(n_prev,1), return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(128))
    model.add(Dropout(.2))
    model.add(Dense(num_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
    return model



def generate_poetry_words(textive, sentence_length, n_lines, model):
  for i in range(n_lines):
    sent = []
    for _ in range(sentence_length):
      varies = tokenizer.texts_to_sequences([textive])
      varies = pad_sequences(varies, maxlen=20, padding='pre')

      y_pred = np.argmax(model.predict(varies), axis=-1)

      predicted_word = ""
      for word, index in tokenizer.word_index.items():
        if index == y_pred:
          predicted_word = word
          break

      textive = textive + ' ' + predicted_word
      sent.append(predicted_word)

    textive = sent[-1]
    sent = ' '.join(sent)
    print(sent)
    
def generate_poetry_characters(textive, sentence_length, n_lines, model, char_indices):
  for i in range(n_lines):
    sent = []
    for _ in range(sentence_length):
      seed = [char for char in textive]
      encoded_seed = [char_indices[char] for char in seed]
      if len(encoded_seed) < 100:
        for j in range(99-len(encoded_seed)):
          encoded_seed.insert(0,0)

      y_pred = np.argmax(model.predict([encoded_seed]), axis=-1)

      predicted_character = ""
      for character, index in char_indices.items():
        if index == y_pred:
          predicted_character = character
          break

      textive = textive + predicted_character
      sent.append(predicted_character)

    textive = sent[-1]
    sent = ''.join(sent)
    print(sent)

if __name__ == '__main__':
    data = load_text_data()
    encoded_texts = tokenize_texts(data)
    X,Y = _windowize_data(encoded_texts[0], n_prev)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)
    num_words = len(tokenizer.word_index) + 1
    model = create_model(num_words, n_prev)

