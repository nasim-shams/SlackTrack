# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:51:34 2018

@author: shams
"""

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Dense, Input, LSTM
from keras.models import Model
import h5py


# loading input file
user= 'U0AB5K6HY'
filename = 'C:/Users/shams/OneDrive/Desktop/Insight/datasets/'+user+'.json'
df_data_1 = pd.read_json(filename)
# defining the batch size and number of epochs 
# per day
batch_size = 50
epochs = 100
timesteps = 7

#per week
batch_size = 10
epochs = 100
timesteps = 1

def get_train_length(dataset, batch_size, test_percent):
    # substract test_percent to be excluded from training, reserved for testset
    length = len(dataset)
    length *= 1 - test_percent
    train_length_values = []
    for x in range(int(length) - 100,int(length)): 
        modulo=x%batch_size
        if (modulo == 0):
            train_length_values.append(x)
    return (max(train_length_values))

length = get_train_length(df_data_1, batch_size, 0.1)
upper_train = length + timesteps*2
df_data_1_train = df_data_1[0:upper_train]
training_set = np.nan_to_num(df_data_1_train.loc[:,:].values)
training_set.shape

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc.fit_transform(np.float64(training_set.reshape(-1,1)))
training_set_scaled = sc.fit_transform(np.float64(training_set))
training_set_scaled.shape

x_train = []
y_train = []

# Creating a data structure with n timesteps
"""
for i in range(timesteps, length + timesteps): 
    x_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i:i+timesteps,0])
    #y_train.append(training_set_scaled[i:i+1,0])
"""

# Creating a data structure with n timesteps: MULTIVARIATE
for i in range(timesteps, length + timesteps): 
    x_train.append(training_set_scaled[i-timesteps:i,:])
    y_train.append(training_set_scaled[i:i+timesteps,0])
    #y_train.append(training_set_scaled[i:i+1,0])


print (length + timesteps)
print (len(x_train))
print (len (y_train))
print (np.array(x_train).shape)
print (np.array(y_train).shape)
"""
# Reshaping
X_train, y_train = np.array(x_train), np.array(y_train)
X_train = np.reshape(x_train, (X_train.shape[0], X_train.shape[1], ))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
"""
# Reshaping
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

from keras.layers import Dense, Input, LSTM
from keras.models import Model
import h5py

# Initialising the LSTM Model with MAE Loss-Function
# Using Functional API

inputs_layer = Input(batch_shape=(batch_size,timesteps,x_train.shape[2]))
lstm_1 = LSTM(10, stateful=True, return_sequences=True)(inputs_layer)
lstm_2 = LSTM(10, stateful=True, return_sequences=True)(lstm_1)
output_layer = Dense(units = 1)(lstm_2)

regressor_mae = Model(inputs=inputs_layer, outputs = output_layer)

regressor_mae.compile(optimizer='adam', loss = 'mae')
regressor_mae.summary()
regressor_mae.save(filepath='C:/Users/shams/OneDrive/Desktop/Insight/models/test_model.h5')


for i in range(epochs):
    print("Epoch: " + str(i))
    regressor_mae.fit(X_train, y_train, shuffle=False, epochs = 1, batch_size = batch_size)
    regressor_mae.reset_states()









