# load data

import pandas as pd
import numpy as np
#import tokeizer
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from Models.LSTM import lstm_model, train_lstm_model, predict_lstm_model
from Models.GRU import gru_model, train_gru_model, predict_gru_model
from Models.BiLSTM import bilstm_model, train_bilstm_model, predict_bilstm_model
from Models.BiGRU import bigru_model, train_bigru_model, predict_bigru_model

# load data
df = pd.read_csv('Data/Is_troll_body.csv')
df2 = pd.read_csv('Data/Is_not_troll_body.csv')

#split data into train and test 70/30
train = df.sample(frac=0.7, random_state=200)
test = df.drop(train.index)

train2 = df2.sample(frac=0.7, random_state=200)
test2 = df2.drop(train2.index)

#concatenate train and test data
train = pd.concat([train, train2])
test = pd.concat([test, test2])

#shuffle data
train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)


# split data into X and y
X_train = train['body']
y_train = train['is_troll']
X_test = test['body']
y_test = test['is_troll']

y_train = y_train.replace({'1': 1, '0': 0})
y_test = y_test.replace({'1': 1, '0': 0})

optymizers = ['adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']

def preprocess_sentences(X_train, X_test):
    # Convert sentences to sequences
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Pad sequences with zeros
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=100)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=100)

    return X_train, X_test

X_train, X_test = preprocess_sentences(X_train, X_test)

for opt in optymizers:
    print("Optymizer: ", opt)
    lstm = lstm_model(opt)
    lstm = train_lstm_model(lstm, X_train, y_train, X_test, y_test)
    lstm_scores = predict_lstm_model(lstm, X_test, y_test)
    gru = gru_model(opt)
    gru = train_gru_model(gru, X_train, y_train, X_test, y_test)
    gru_scores = predict_gru_model(gru, X_test, y_test)
    bilstm = bilstm_model(opt)
    bilstm = train_bilstm_model(bilstm, X_train, y_train, X_test, y_test)
    bilstm_scores = predict_bilstm_model(bilstm, X_test, y_test)
    bigru = bigru_model(opt)
    bigru = train_bigru_model(bigru, X_train, y_train, X_test, y_test)
    bigru_scores = predict_bigru_model(bigru, X_test, y_test)
    print("LSTM: %.2f%%" % (lstm_scores[1]*100))
    print("GRU: %.2f%%" % (gru_scores[1]*100))
    print("BiLSTM: %.2f%%" % (bilstm_scores[1]*100))
    print("BiGRU: %.2f%%" % (bigru_scores[1]*100))
    print("_"*50)








