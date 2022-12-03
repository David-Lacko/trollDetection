import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb




#LSTM Model
def lstm_model(optymizer = "adam"):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(10000, 128))
    model.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(keras.layers.Dense(46, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer= optymizer, metrics=['accuracy'])
    return model


def train_lstm_model(model, X_train, y_train, X_test, y_test,epochs=3, batch_size=64):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return model

def predict_lstm_model(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return scores




