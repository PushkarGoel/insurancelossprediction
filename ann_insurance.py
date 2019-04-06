#Part 1-Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.metrics import mean_absolute_error 
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense


# Importing the dataset
dataset = pd.read_csv('Insurance_Data.csv')
X = dataset.iloc[:, 2:7].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
x1=x1.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_x1 = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
x_test=sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test=sc_y.transform(y_test)
x1 = sc_x1.fit_transform(x1)

# Part 2 - Now let's make the ANN!

# Initialising the ANN
ann = Sequential()

# Adding the input layer and the first hidden layer
ann.add(Dense(256, kernel_initializer = 'normal', activation = 'relu', input_dim = 5))

# Adding the second adn third hidden layer
ann.add(Dense(256, kernel_initializer = 'normal', activation = 'relu'))
ann.add(Dense(128, kernel_initializer = 'normal', activation = 'relu'))

# Adding the output layer
ann.add(Dense(1, kernel_initializer = 'normal', activation = 'linear'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])
ann.summary()


#added checkpoints for finding the required iteration of model to avoid over or underfitting

# simple early stopping 
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=200)

#saving required checkpoint of model
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)

# fit model
history = ann.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0, callbacks=[es, mc])

# load the saved model
saved_model = load_model('best_model.h5')

# evaluate the model
_, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
_, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

y_pred=saved_model.predict(x_test)

#finding rmse for y_pred and y_test
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(y_pred, y_test)
print("rms error is: " + str(rmse_val))


