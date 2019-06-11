# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:50:13 2018

@author: shams
"""


# importing libraries

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Dense, Input, LSTM,  GRU, BatchNormalization
from keras.models import Model
import h5py
from sklearn.preprocessing import RobustScaler

Info = 'B5'
loss_func = 'mse'
#~~~~~~~~~~~~~~~~ loading data~~~~~~~~~~~~~~ 
df_data_1 = pd.read_json('/scratch/nshams/data/adminData.json')
df_data_1 = df_data_1.drop('usr_tag',axis =1)
#~~~~~~~~~~~~~~preparing train and test data~~~~~~~~~~~~~~~~~~~~~~
# defining the batch size and number of epochs 
# per day
batch_size = 20
epochs = 100
timesteps = 2

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

length = get_train_length(df_data_1, batch_size, 0.2)
upper_train = length + timesteps*2
df_data_1_train = df_data_1[0:upper_train]
training_set_y = np.nan_to_num(df_data_1_train['user_ts'].values)
training_set = np.nan_to_num(df_data_1_train.loc[:,:].values)

# Feature Scaling
#from sklearn.preprocessing import MinMaxScaler RobustScaler
#sc = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc.fit_transform(np.float64(training_set.reshape(-1,1)))
#training_set_scaled = sc.fit_transform(np.float64(training_set))
#training_set_y_scaled = sc.fit_transform(np.float64(training_set_y.reshape(-1,1)))
sc = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
training_set_scaled = sc.fit_transform(training_set)
training_set_y_scaled = np.sign(training_set_y.reshape(-1,1))
#training_set_y_scaled = sc.fit_transform(training_set_y.reshape(-1,1))



x_train = []
y_train = []

for i in range(timesteps, length + timesteps): 
    x_train.append(training_set_scaled[i-timesteps:i,:])
    y_train.append(training_set_y_scaled[i:i+timesteps])
    
# Reshaping
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

print ('training set x :', np.array(x_train).shape)
print ('training set y:', np.array(y_train).shape)

#~~~~~~~~~~~~~~~~~~~ building the model ~~~~~~~~~~~~~~~~~~~~~~~~~~~
inputs_layer = Input(batch_shape=(batch_size,timesteps,x_train.shape[2]))
lstm_1 = GRU(8, stateful=True, return_sequences=True)(inputs_layer)
#lstm_2 = LSTM(8, stateful=True, return_sequences=True)(lstm_1)
batchN = BatchNormalization()(lstm_1)
dense_1 = Dense(units = 8)(batchN)
output_layer = Dense(units = 1,activation = 'elu')(dense_1)

regressor = Model(inputs=inputs_layer, outputs = output_layer)

regressor.compile(optimizer='adam', loss = loss_func)
regressor.summary()


#~~~~~~~~~~~~~~~training~~~~~~~~~~~~~~~~~~~~~~~~~~
for i in range(epochs):
    print("Epoch: " + str(i))
    regressor.fit(x_train, y_train, shuffle=False, epochs = 1, batch_size = batch_size)
    regressor.reset_states()

#~~~~~~~~~~~~~~~~saving the trained model~~~~~~~~~~~~~~~~~~~~~~~~~
    
regressor.save(filepath='/scratch/nshams/models/'+ Info +'_'+loss_func+'_trained.h5')
