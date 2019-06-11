# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:49:49 2018

@author: shams
"""

import numpy as np
import pandas as pd
import networkx as nx

#from keras.preprocessing import sequence
#from keras.models import load_model
#from keras.layers import Dense, Input, LSTM,  GRU
#from keras.models import Model
#import h5py



def usr_top_chans(usr, netWindow , nchans = 5):
    
    
     chanList = list(netWindow.loc[netWindow['user']==usr]['channel'].unique())
     b=netWindow.groupby(['user','channel']).count().reset_index()
     b['weight'] = b['text']
     b=b.drop(['subtype','type','ts','time','date','text'],axis =1)
     G =nx.DiGraph()
     networkG=nx.from_pandas_edgelist(b,source ='user',target='channel', create_using = G )
     networkG.add_weighted_edges_from(list(b.itertuples(index=False, name=None)))
     try:
        h,a=nx.hits(networkG)
        bib = dict((k, a[k]) for k in chanList if k in a)
        chScore = pd.DataFrame.from_dict(bib,orient='index')
        chScore.columns=['hScore']
        chScore= chScore.sort_values(by='hScore',ascending=False)
     except:
        h,a=nx.hits(networkG,tol=1e-01)
        bib = dict((k, a[k]) for k in chanList if k in a)
        chScore = pd.DataFrame.from_dict(bib,orient='index')
        chScore.columns=['hScore']
        chScore= chScore.sort_values(by='hScore',ascending=False)
        
     return(chScore.iloc[0:nchans])

# loading data
user_file = pd.read_json('C:/Users/shams/OneDrive/Documents/Projects/Insight/datasets/users.json')

#channel_file = pd.read_json('/scratch/nshams/data/channels.json')

#allData = pd.read_json('/scratch/nshams/data/allData.json')
allData = pd.read_json('C:/Users/shams/OneDrive/Documents/Projects/Insight/datasets/allData.json')


# prepare training data
freq = 'D'
winSize = 10

#freq = 'W'
allData.drop(allData['user']==None)
allData['time'] = pd.to_datetime(allData['ts'],unit='s')
networkLog = allData.sort_values(by=['ts'] )
networkLog['date']=networkLog['time'].apply(lambda x : x.date())  
networkLog['date'] = pd.to_datetime(networkLog['date'])
#usr_list = [x for x in allData['user'].unique() if x is not None]
usr_list = user_file[user_file['is_admin']==1]['id'].tolist()
bigData = pd.DataFrame()
for usr in usr_list:
    
    # ----------------------build user's time series 
    
    startDate = networkLog.loc[networkLog['user']==usr]['date'].min()   
    endDate = networkLog.loc[networkLog['user']==usr]['date'].max()
    usrWindow = networkLog.loc[(networkLog['date']>= startDate )& (networkLog['date']<= endDate) ]
    usrLog = usrWindow.loc[usrWindow['user']== usr]
    usr_daily = usrLog['date'].value_counts().sort_index()   
    usr_daily= usr_daily.reindex(fill_value=0)
    usr_daily.index = pd.DatetimeIndex(usr_daily.index)
    #usr_weekly = usr_daily.resample('W').sum()
    usr_ts = usr_daily.resample(freq).sum()
    input_ts = pd.DataFrame(usr_ts,index = usr_ts.index)
    input_ts = input_ts.rename(columns={'date':'user_ts'})
    input_ts.fillna(0, inplace=True)
    input_ts['usr_ma'] = input_ts['user_ts'].rolling(window=winSize).mean()
    
    
    # ----------------------find corresponding high score channels
    topChans = usr_top_chans(usr, usrWindow , nchans = 3)
    topChans_list = topChans.index.values.tolist()
    
    ch_counter = list(enumerate(topChans_list, 1))
    for counter , ch in ch_counter:
       
        channel_log = usrWindow[usrWindow['channel']==ch].sort_values(by='ts')
        channel_log['date']=channel_log['time'].apply(lambda x : x.date())
        channel_daily = channel_log['date'].value_counts().sort_index()
        channel_daily = channel_daily.reindex(fill_value=0)
        channel_daily.index = pd.DatetimeIndex(channel_daily.index)
        channel_ts = channel_daily.resample(freq).sum()
        input_ts['ch'+str(counter)] = channel_ts
        input_ts['ch'+str(counter)].fillna(0,inplace = True)
        input_ts['ch'+str(counter)+'_ma'] = input_ts['ch'+str(counter)].rolling(window=winSize).mean()
        
    input_ts = input_ts.iloc[winSize:,:]
    input_ts['usr_tag'] = [usr for x in range(len(input_ts))]
    bigData = bigData.append(input_ts)
    #input_ts.to_json('/scratch/nshams/data/byUser/'+usr+'.json')

bigData.to_json('C:/Users/shams/OneDrive/Documents/Projects/Insight/datasets/adminData.json')
    
    
    # append to the training data
    
    


