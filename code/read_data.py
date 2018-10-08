# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:35:12 2018

@author: shams
"""

import pandas as pd
import numpy
import os 
import matplotlib.pyplot as pyplot

#--------------------- load data and save per channel --------------------
channel_names = os.listdir('C:/Users/shams/OneDrive/Desktop/Insight/datasets/Insight_data/')
for channel_name in channel_names:
    
    channel_folder= 'C:/Users/shams/OneDrive/Desktop/Insight/datasets/Insight_data/'+channel_name+'/'
    channel_days = os.listdir(channel_folder)
    #channel_days.index(channel_name+'.json')
    channel_days.remove(channel_name+'.json')
    channelData = pd.DataFrame()
    
    for day in channel_days:
        filename = channel_folder + day
        dayData = pd.read_json(filename).loc[:,['user','ts','type','subtype','text']]
        dayData['time'] = pd.to_datetime(dayData['ts'],unit='s')
        channelData = channelData.append(dayData)
    
    channelData = channelData.reset_index(drop=True)
    if os.path.isfile(channel_folder+channel_name+'.json'): os.remove(channel_folder+channel_name+'.json')
    channelData.to_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/channels_alldays/'+channel_name+'.json',orient = 'records')
    #test_read = pd.read_json(filename).loc[:,['user','ts','type','subtype','text']]
    
#----------------- load data from previous section, convert to time series and save as big fat files..
channel_names = os.listdir('C:/Users/shams/OneDrive/Desktop/Insight/datasets/Insight_data/')
allChannelData = pd.DataFrame()
for channel_name in channel_names:
    channel_folder= 'C:/Users/shams/OneDrive/Desktop/Insight/datasets/Insight_data/'+channel_name+'/'
    tmp =  pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/channels_alldays/'+channel_name +'.json')
    tmp['channel'] = channel_name
    allChannelData = allChannelData.append(tmp)
    
allChannelData = allChannelData.reset_index(drop=True)
allChannelData.to_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/allData.json')

#~~~~~~~~~~~~~~~~~~~~~~~~~ the actual preprocessing~~~~~~~~~~~~~~~~~~~~~~~~~~~
allChannelData = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/allData.json')

allChannelData['time'] = pd.to_datetime(allChannelData['ts'],unit='s')

userIDs = list()
userPosts = list()
userTS = pd.DataFrame()
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
        userTS['daily'] = usr_daily
        userTS['weekly'] = usr_weekly
        userTS['user'] = user
        #userTS.to_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/byUser/'+user+'.json')
        
        userIDs.append(user)
        userPosts.append(usr_weekly.sum())
    except:
        print('something is fucked up')
userTotal = list(zip(userIDs,userPosts))
userTotal = pd.DataFrame(userTotal, columns=['user','totalPosts'])


      
#for user prediction
start_day = usr_daily.index[0]
            
#for channel prediction
 
    # randome stuff:
    #index = pd.date_range(start=usrLog['date'].min(),end=usrLog['date'].max(), freq='D')
    #Tseries = pd.Series(np.zeros(len(index)), index=index)
for ii in range(10):
    user = userTotal.loc[ii]['user']
    activeUserChans = pd.DataFrame(allChannelData[allChannelData['user']==user]['channel'])
    activeUserChans= activeUserChans['channel'].unique()
    
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
            isweekend = np.array(channel_daily.index.weekday>4)*1
            #index = pd.date_range(start=usrLog['date'].min(),end=usrLog['date'].max(), freq='D')

            
#for user prediction
start_day = min
            
#for channel prediction
  


   
    