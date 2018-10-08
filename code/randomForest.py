# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:40:08 2018

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

#~~~~~~~~~~~~~~~~~~~~~~~~~loading data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

allChannelData = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/allData.json')
allChannelData['time'] = pd.to_datetime(allChannelData['ts'],unit='s')
userTotal = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/userTotal.json')
userTotal = userTotal.sort_values(by=['totalPosts'],ascending = False)

#user= 'U0AB5K6HY'
userID = list()
userw2=list()
userw4=list()
userw6=list()
userw8=list()
userw14=list()
usrNchan = list()
AvgChScore = list()
Nconnected = list()
activeUser = list()

a = allChannelData['user'].unique()
a = [x for x in a if x is not None]

for user in a:
    #print(user)
    userID.append(user)
    #   for user in userTotal.iloc[9:,]['user']:
    usrLog = allChannelData[allChannelData['user']==user].sort_values(by=['ts']) 
    usrLog['date']=usrLog['time'].apply(lambda x : x.date())   
    usr_daily = usrLog['date'].value_counts().sort_index()   
    usr_daily= usr_daily.reindex(fill_value=0)
    usr_daily.index = pd.DatetimeIndex(usr_daily.index)
    usr_daily = usr_daily.resample('D').sum()
    usr_weekly = usr_daily.resample('W').sum()
    usr_weekly.index = pd.DatetimeIndex(usr_weekly.index)

    usr_weekly.fillna(0)   
    
    usr0 =usr_weekly.index[0]
    try:
        usr8 =usr_weekly.index[7]
    except:
        usr8 = usr_weekly.index.max()


    #user_init.append(usr_weekly.iloc[0:4].sum())
    #user_final.append(usr_weekly.iloc[-8:-4].sum())
    userw2.append(usr_weekly.iloc[0:2].mean()) 
    userw4.append(usr_weekly.iloc[2:4].mean()) 
    userw6.append(usr_weekly.iloc[4:6].mean()) 
    try:
        userw8.append(usr_weekly.iloc[6:8].mean()) 
    except:
        userw8.append(0)
    try :
        userw14.append(usr_weekly.iloc[12:14].mean()) 
    except:
        userw14.append(0)
     

    
    
    #number of channels user is a member of @ wk8    
    
    #for ch in usrLog['channel'].unique():
        
   
    usr_joined = usrLog[usrLog['subtype']=='channel_join']
    usr_joined.index = usr_joined['time']
    usr_joined= usr_joined[usr0:usr8]
        
    usr_left = usrLog[usrLog['subtype']=='channel_left']
    usr_left.index = usr_left['time']
    usr_left= usr_left[usr0:usr8]
        
    usrChans = usr_joined['channel'].nunique() - usr_left['channel'].nunique()
    usrNchan.append(usrChans)

   #channel activity score
    ch_score_list = list()
    nusers =list()
    userList = pd.Series()
    nposts =list()

    for channel in usr_joined['channel'].unique() :
        
        channelData = allChannelData[allChannelData['channel']==channel]
        channelLog = channelData.sort_values(by=['ts']) 
        channelLog['time'] = pd.to_datetime(channelLog['ts'],unit='s')
        channelLog['date']=channelLog['time'].apply(lambda x : x.date())  
        ch_daily = channelLog['date'].value_counts().sort_index()   
        ch_daily= ch_daily.reindex(fill_value=0)
        ch_daily.index = pd.DatetimeIndex(ch_daily.index)
        ch_daily = ch_daily.resample('D').sum()
        ch_weekly = ch_daily.resample('W').sum()
        ch_weekly=ch_weekly.fillna(0)
        ch_score = ch_weekly[usr0:usr8].mean()
        ch_score_list.append(ch_score)
        
        nposts.append(channelData.index.max())
        userList = userList.append(channelLog['user'])

        
    ch_score_list = np.array(ch_score_list)
    avg_ch_score = ch_score_list.mean()
    AvgChScore.append(avg_ch_score)
    
    Nusers = userList.nunique()
    Nconnected.append(Nusers)
    
    if userw14[-1] > 0 :
        isactive = 1
    else:
        isactive = 0
        
    activeUser.append(isactive)
    
    
    
UserData = pd.DataFrame({'wk2': userw2,
                         'wk4':userw4,
                         'wk6':userw6,
                         'wk8':userw8,
                         'wk14':userw14,
                         'Nusers':Nconnected,
                         'Nchans':usrNchan,
                         'chanScore':AvgChScore,
                         'isActvie':activeUser},index=userID)      
    
UserData.to_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/RFData.json')
        
    
        
        
"""     

tmp = list(zip(user_init,user_final))
user_trend = pd.DataFrame(tmp, columns=['user_start','user_final'])

user_trend.plot.scatter(x='user_start',y='user_final')


channel_name = list()
nposts =list()
nusers =list()
userList = list()
for channel in allChannelData['channel'].unique():
    channelData = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/channels_alldays/'+channel+'.json')
    channel_name.append(channel)
    nposts.append(channelData.index.max())
    nusers.append(channelData['user'].nunique())
    userList.append(list(channelData['user'].unique()))
    
    channelLog = channelData.sort_values(by=['ts']) 
    channelLog['time'] = pd.to_datetime(channelLog['ts'],unit='s')
    channelLog['date']=channelLog['time'].apply(lambda x : x.date())  
    #channelLog2= channelLog[(channelLog['subtype']!='channel_join') &(channelLog['subtype']!='channel_leave') ]
     tmp = people[(people['building']==ii) & (people['time']==people['time'].max())]
    ch_daily = channelLog['date'].value_counts().sort_index()   
    ch_daily= ch_daily.reindex(fill_value=0)
    ch_daily.index = pd.DatetimeIndex(ch_daily.index)
    ch_daily = ch_daily.resample('D').sum()
    ch_weekly = ch_daily.resample('W').sum()
    ch_weekly=ch_weekly.fillna(0)
    
    
chanStats = pd.DataFrame({'Nusers':nusers,'Nposts':nposts},index=channel_name)
chanStats = chanStats.sort_values(by=['Nposts'],ascending = False)

"""
