# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:58:13 2018

@author: shams
"""
# importing libraries
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Dense, Input, LSTM
from keras.models import Model
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler



Info = 'B5'
loss_func = 'mse'
#~~~~~~~~~~~~~~~~ loading data and model ~~~~~~~~~~~~~~ 
df_data_1 = pd.read_json('/scratch/nshams/data/adminData.json')
df_data_1 = df_data_1.drop('usr_tag',axis =1)
regressor = load_model(filepath='/scratch/nshams/models/'+ Info +'_'+loss_func+'_trained.h5')

#~~~~~~~~~~~~~~~ create test set ~~~~~~~~~~~~~~~~~~~~~~~
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

def get_test_length(dataset, batch_size):
    
    test_length_values = []
    for x in range(len(dataset) - 200, len(dataset) - timesteps*2): 
        modulo=(x-upper_train)%batch_size
        if (modulo == 0):
            test_length_values.append(x)
            print (x)
    return (max(test_length_values))

test_length = get_test_length(df_data_1, batch_size)
print(test_length)
upper_test = test_length + timesteps*2
testset_length = test_length - upper_train
print (testset_length)
print (upper_train, upper_test, len(df_data_1))

# construct test set

#subsetting
df_data_1_test = df_data_1[upper_train:upper_test] 
test_set_y = np.nan_to_num(df_data_1_test['user_ts'].values)
test_set = np.nan_to_num(df_data_1_test.loc[:,:].values)

#scaling
sc = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
scaled_test_values = sc.fit_transform(np.float64(test_set))
scaled_test_values_y = np.sign(test_set_y.reshape(-1,1))
#scaled_test_values_y = sc.fit_transform(np.float64(test_set_y.reshape(-1,1)))

#scaled_test_values = np.tanh(np.float64(test_set))
#scaled_test_values_y = np.tanh(np.float64(test_set_y.reshape(-1,1)))



#creating input data
x_test = []
y_test = []
for i in range(timesteps, testset_length + timesteps):
    x_test.append(scaled_test_values[i-timesteps:i, :])
    y_test.append(scaled_test_values_y[i:timesteps+i])# this is for the last timestep (7th)
    #y_test.append(scaled_test_values_y[i:1+i])
x_test = np.array(x_test)
y_test = np.array(y_test)


#reshaping
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

predicted_values = regressor.predict(x_test, batch_size=batch_size)

np.save('/scratch/nshams/models/predvals.npy',predicted_values)



#regressor_mse.reset_states()
