# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:05:20 2018

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
from statsmodels.tsa.arima_model import ARIMA
import math


ii = 0
#~~~~~~~~~~~~~~~~ loading data~~~~~~~~~~~~~~ 
userTotal = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/userTotal.json')
userTotal = userTotal.sort_values(by=['totalPosts'],ascending = False)
user = userTotal.iloc[ii]['user']
filename = 'C:/Users/shams/OneDrive/Desktop/Insight/datasets/'+user+'.json'
df_data_1 = pd.read_json(filename)['date']
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X= np.nan_to_num(df_data_1.values)
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(7,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.figure()
plt.plot(test)
plt.plot(predictions, color='red')
plt.title('ARIMA')
plt.show()