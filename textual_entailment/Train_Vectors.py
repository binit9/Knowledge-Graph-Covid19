from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
import numpy as np

# https://keras.io/getting-started/sequential-model-guide/

def trainSequential(model_name, trainDataVecs, train_label):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=trainDataVecs.shape[1]))
    model.add(Dense(train_label.shape[1], activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(trainDataVecs, train_label, epochs=100, batch_size=32)
    # loss_and_metrics = model.evaluate(testDataVecs, test_label, batch_size=128)

    model.save(model_name)
    del model

def trainMLP(model_name, trainDataVecs, train_label):
    model = Sequential()
    model.add(Dense(64, input_dim=trainDataVecs.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(train_label.shape[1], activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(trainDataVecs, train_label, epochs=500, batch_size=32)
    # loss_and_metrics = model.evaluate(testDataVecs, test_label, batch_size=128)

    model.save(model_name)
    del model

def trainLSTM(model_name, trainDataVecs, train_label):
    model = Sequential()
    model.add(LSTM(128, input_shape=trainDataVecs.shape[1:]))
    model.add(Dense(train_label.shape[1]))
    model.add(Activation('softmax'))

    model.summary()

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(trainDataVecs, train_label, epochs=100, batch_size=32)
    # loss_and_metrics = model.evaluate(testDataVecs, test_label, batch_size=128)

    model.save(model_name)
    del model

def trainLSTM1(model_name, trainDataVecs, train_label):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(trainDataVecs, train_label, epochs=100, batch_size=32)
    # loss_and_metrics = model.evaluate(testDataVecs, test_label, batch_size=128)

    model.save(model_name)
    del model






def trainMaxCosLSTM(model_name, trainDataVecs, train_label):
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, 100)))
    model.add(Dense(train_label.shape[1]))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(trainDataVecs, train_label, epochs=100, batch_size=32)
    # loss_and_metrics = model.evaluate(testDataVecs, test_label, batch_size=128)

    model.save(model_name)
    del model

# https://arxiv.org/pdf/1705.09054.pdf


def trainCNN(model_name, trainDataVecs, train_label):
    # Embedding
    max_features = 20000
    maxlen = 200
    embedding_size = 128

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 30
    epochs = 2

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Train...')
    model.fit(trainDataVecs, train_label, batch_size=batch_size, epochs=epochs)

    model.save(model_name)
    del model
