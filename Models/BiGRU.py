import keras

def bigru_model(optymizer = "adam"):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(10000, 128))
    model.add(keras.layers.Bidirectional(keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(keras.layers.Dense(46, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optymizer, metrics=['accuracy'])
    return model

def train_bigru_model(model, X_train, y_train, X_test, y_test,epochs=3, batch_size=64):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return model

def predict_bigru_model(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return scores
