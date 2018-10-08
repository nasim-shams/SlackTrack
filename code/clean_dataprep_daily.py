# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:57:19 2018

@author: shams
"""


import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Dense, Input, LSTM
from keras.models import Model
import h5py


#loading main dataset
allChannelData = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/allData.json')
allChannelData['time'] = pd.to_datetime(allChannelData['ts'],unit='s')
userTotal = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/userTotal.json')
userTotal = userTotal.sort_values(by=['totalPosts'],ascending = False)
"""
allChannelData['time'] = pd.to_datetime(allChannelData['ts'],unit='s')
user_full_df = pd.DataFrame()

userIDs = list()
userPosts = list()
for user in allChannelData['user'].unique():
    
    try:

        #ts = allChannelData[allChannelData['user']==user]['ts']/(24*60*60)
        usrLog = allChannelData[allChannelData['user']==user].sort_values(by=['ts'])
        usrLog['date']=usrLog['time'].apply(lambda x : x.date())   
        usr_daily = usrLog['date'].value_counts().sort_index()   
        usr_daily= usr_daily.reindex(fill_value=0)
        usr_daily.index = pd.DatetimeIndex(usr_daily.index)
        usr_weekly = usr_daily.resample('W').sum()
        usr_daily = usr_daily.resample('D').sum()
       # userTS['daily'] = usr_daily
        #userTS['weekly'] = usr_weekly
        #userTS['user'] = user
        #userTS.to_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/byUser/'+user+'.json')
        
        userIDs.append(user)
        userPosts.append(usr_weekly.sum())
    except:
        print('i sense distrubance in the force')
userTotal = list(zip(userIDs,userPosts))
userTotal = pd.DataFrame(userTotal, columns=['user','totalPosts'])
userTotal.to_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/userTotal.json')

"""
#for channel prediction
 
    # randome stuff:
    #index = pd.date_range(start=usrLog['date'].min(),end=usrLog['date'].max(), freq='D')
    #Tseries = pd.Series(np.zeros(len(index)), index=index)
for ii in range(10):
    
    user = userTotal.iloc[ii]['user']
    activeUserChans = pd.DataFrame(allChannelData[allChannelData['user']==user]['channel'])
    activeUserChans= activeUserChans['channel'].unique()
    
    usrLog = allChannelData[allChannelData['user']==user].sort_values(by=['ts'])
    
    usrLog['date']=usrLog['time'].apply(lambda x : x.date())   
    usr_daily = usrLog['date'].value_counts().sort_index()   
    usr_daily= usr_daily.reindex(fill_value=0)
    usr_daily.index = pd.DatetimeIndex(usr_daily.index)
    usr_weekly = usr_daily.resample('W').sum()
    usr_daily = usr_daily.resample('D').sum()
       #
    user_full_df=pd.DataFrame(usr_daily)

      
#for user prediction
    start_day = usr_daily.index[0]
    isweekend = np.array(usr_daily.index.weekday>4)*1
    
    for jj in activeUserChans:
            print(jj)
            #channelData = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/channels_alldays/'+jj+'.json')
            channel_log = allChannelData[allChannelData['channel']==jj].sort_values(by='ts')
            channel_log['date']=channel_log['time'].apply(lambda x : x.date())
            channel_daily = channel_log['date'].value_counts().sort_index()
            channel_daily = channel_daily.reindex(fill_value=0)
            channel_daily.index = pd.DatetimeIndex(channel_daily.index)
            channel_daily = channel_daily.resample('D').sum()
            channel_weekly = channel_daily.resample('W').sum()
            user_full_df=user_full_df.join(pd.DataFrame(channel_daily).rename(columns={'date':jj}))
            #index = pd.date_range(start=usrLog['date'].min(),end=usrLog['date'].max(), freq='D')
            
    user_full_df=user_full_df.join(pd.DataFrame(isweekend,columns={'isweekend'},index=usr_daily.index))
    user_full_df.to_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/'+user+'.json')   

            
