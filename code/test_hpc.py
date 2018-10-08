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


ii = 0
loss_func = 'mse'
#~~~~~~~~~~~~~~~~ loading data~~~~~~~~~~~~~~ 
userTotal = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/userTotal.json')
userTotal = userTotal.sort_values(by=['totalPosts'],ascending = False)
user = userTotal.iloc[ii]['user']
filename = 'C:/Users/shams/OneDrive/Desktop/Insight/datasets/'+user+'.json'
df_data_1 = pd.read_json(filename)
#~~~~~~~~~~~~~~~ create test set ~~~~~~~~~~~~~~~~~~~~~~~
batch_size = 20
epochs = 100
timesteps = 2


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
test_set_y = np.nan_to_num(df_data_1_train['date'].values)
test_set = np.nan_to_num(df_data_1_test.loc[:,:].values)

#scaling
scaled_test_values = sc.fit_transform(np.float64(test_set))
scaled_test_values_y = sc.fit_transform(np.float64(test_set_y.reshape(-1,1)))



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
#regressor = load_model(filepath='C:/Users/shams/OneDrive/Desktop/Insight/models/'+user+'_'+loss_func+'_trained.h5')
regressor = load_model(filepath='C:/Users/shams/OneDrive/Desktop/Insight/models/U03AZ5GP5_mae_trained_3lb204hrts2.h5')

predicted_values = regressor.predict(x_test, batch_size=batch_size)

#regressor_mse.reset_states()

predicted_values = np.reshape(predicted_values, 
                             (predicted_values.shape[0], 
                              predicted_values.shape[1]))

predicted_values = sc.inverse_transform(predicted_values)

pred_mse = []

for j in range(0, testset_length - timesteps):
    pred_mse = np.append(pred_mse, predicted_values[j, timesteps-1]) # this is for terget = last time step
    #pred_mse = np.append(pred_mse, predicted_values[j, 0])
pred_mse = np.reshape(pred_mse, (pred_mse.shape[0], 1))
predicted_values = np.reshape(predicted_values[:,1],100,1)



# Visualising the results
plt.figure()
#plt.plot(test_set[timesteps:len(pred_mse),0], color = 'red', label = 'Real')
plt.plot(scaled_test_values[timesteps:len(pred_mse),0], color = 'red', label = 'Real')
#plt.figure()
#plt.plot(pred_mse[0:len(pred_mse) - timesteps], color = 'green', label = 'Predicted')
plt.plot(predicted_values[0:len(predicted_values) - timesteps], color = 'green', label = 'Predicted')
plt.title('7th day prediction')
plt.xlabel('Time')
plt.ylabel('posts per day')
plt.legend()
plt.show()



from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_set[timesteps:len(pred_mse)], pred_mse[0:len(pred_mse) - timesteps]))
print(rmse)


mean = np.mean(np.float64(test_set[timesteps:len(pred_mse)]))
print (mean)


rmse/mean * 100

