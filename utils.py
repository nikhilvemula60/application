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

def char_gen(textive, number_of_sentences, n_lines, model, char_indices):
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

def file_pandas(filepath):
    data = pandy.read_csv(filepath)
    text = data['text']
    return text

def complete_data(data, inputto):
    data = numpie.array(data)
    nooflstm = len(data) - inputto
    y = data[inputto:]
    indices = numpie.arange(inputto) + numpie.arange(nooflstm)[:, None]
    x = data[indices]
    return x, y


def modelfile(calcword, inputto):
    optimizer = keras.optimizers.Adam(learning_rate=.0001)
    model = Sequential()
    model.add(Embedding(calcword, 128, input_length=inputto))
    model.add(LSTM(128, input_shape=(inputto,1), return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(128))
    model.add(Dropout(.2))
    model.add(Dense(calcword, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
    return model



def sent_gen(textive, number_of_sentences, n_lines, model):
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

def split_text(pre_processed):
    tokenizer.fit_on_texts(pre_processed)
    wrapper = tokenizer.texts_to_sequences(pre_processed)
    return wrapper
    
def text_in_char(pre_processed):
    pre_processed = pre_processed.apply(lambda x: [ele for ele in x])
    chars = sorted(set(pre_processed.explode()))
    char_indices = dict((char, chars.index(char)+1) for char in chars)
    wrapper = []
    for text in pre_processed:
        varies = numpie.array([char_indices[char] for char in text])
        wrapper.append(varies)
    return wrapper, char_indices

if __name__ == '__main__':
    data = file_pandas()
    wrapper = split_text(data)
    X,Y = complete_data(wrapper[0], inputto)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3)
    calcword = len(tokenizer.word_index) + 1
    model = modelfile(calcword, inputto)

