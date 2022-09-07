import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense, Embedding , Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

n_prev = 100
tokenizer = Tokenizer()

def load_text_data(filepath):
    data = pd.read_csv(filepath)
    text = data['text']
    return text

def tokenize_texts(texts):
    tokenizer.fit_on_texts(texts)
    encoded_texts = tokenizer.texts_to_sequences(texts)
    return encoded_texts

def encode_characters(texts):
    texts = texts.apply(lambda x: [ele for ele in x])
    chars = sorted(set(texts.explode()))
    char_indices = dict((char, chars.index(char)+1) for char in chars)
    encoded_texts = []
    for text in texts:
        encoded = np.array([char_indices[char] for char in text])
        encoded_texts.append(encoded)
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
    model.add(Bidirectional(LSTM(150,input_shape=(n_prev,1), return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(96)))
    model.add(Dense(num_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
    return model

def generate_poetry_words(seed_text, poetry_length, n_lines, model):
  for i in range(n_lines):
    text = []
    for _ in range(poetry_length):
      encoded = tokenizer.texts_to_sequences([seed_text])
      encoded = pad_sequences(encoded, maxlen=20, padding='pre')

      y_pred = np.argmax(model.predict(encoded), axis=-1)

      predicted_word = ""
      for word, index in tokenizer.word_index.items():
        if index == y_pred:
          predicted_word = word
          break

      seed_text = seed_text + ' ' + predicted_word
      text.append(predicted_word)

    seed_text = text[-1]
    text = ' '.join(text)
    print(text)
    
def generate_poetry_characters(seed_text, poetry_length, n_lines, model, char_indices):
  for i in range(n_lines):
    text = []
    for _ in range(poetry_length):
      seed = [char for char in seed_text]
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

      seed_text = seed_text + predicted_character
      text.append(predicted_character)

    seed_text = text[-1]
    text = ''.join(text)
    print(text)

if __name__ == '__main__':
    data = load_text_data()
    encoded_texts = tokenize_texts(data)
    X,Y = _windowize_data(encoded_texts[0], n_prev)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)
    num_words = len(tokenizer.word_index) + 1
    model = create_model(num_words, n_prev)

