import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
# from keras.models import load_model

from tensorflow.keras.optimizers import Adam


import pandas as pd
import numpy as np
import  seaborn as sns
import matplotlib.pyplot as plt


from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


cancer = datasets.load_breast_cancer()

X = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)

y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(455,30,1)
X_test = X_test.reshape(114, 30, 1)


epochs = 50
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape = (30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer=Adam(lr=0.00005), loss = 'binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)


# print(np.exp(model.predict(X_test[0:8])))



history.model.save('my_model.h5')