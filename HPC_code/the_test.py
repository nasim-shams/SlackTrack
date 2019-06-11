# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:51:32 2018

@author: shams
"""

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
test_set = df_data_1_test.loc[:,'daily'].values

#scaling
scaled_real_bcg_values_test = sc.fit_transform(np.float64(test_set.reshape(-1,1)))

#creating input data
X_test = []
for i in range(timesteps, testset_length + timesteps):
    X_test.append(scaled_real_bcg_values_test[i-timesteps:i, 0])
X_test = np.array(X_test)


#reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#prediction
predicted_bcg_values_test_mae = regressor_mae.predict(X_test, batch_size=batch_size)
regressor_mae.reset_states()

#print(predicted_bcg_values_test_mae.shape)

#reshaping
predicted_bcg_values_test_mae = np.reshape(predicted_bcg_values_test_mae, 
                                       (predicted_bcg_values_test_mae.shape[0], 
                                        predicted_bcg_values_test_mae.shape[1]))

#print(predicted_bcg_values_test_mae.shape)
#inverse transform
predicted_bcg_values_test_mae = sc.inverse_transform(predicted_bcg_values_test_mae)


#creating y_test data
y_test = []
for j in range(0, testset_length - timesteps):
    y_test = np.append(y_test, predicted_bcg_values_test_mae[j, timesteps-1])

# reshaping
y_test = np.reshape(y_test, (y_test.shape[0], 1))

#print(y_test.shape)

# Visualising the results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(test_set[timesteps:len(y_test)], color = 'red', label = 'Real values')
plt.plot(y_test[0:len(y_test) - timesteps]*200, color = 'blue', label = 'Predicted values')
plt.title('Prediction - MAE')
plt.xlabel('Time')
plt.ylabel('values')
plt.legend()
plt.show()

