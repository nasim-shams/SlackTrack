# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:27:22 2018

@author: shams
"""


#~~~~~~~~~~~~~~~loading packages~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import itertools
import sklearn.tree as skt
import pydotplus
from sklearn.externals.six import StringIO
import graphviz
from imblearn.over_sampling import SMOTE
import networkx as nx
from node2vec import Node2Vec

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


allChannelData = pd.read_json('C:/Users/shams/OneDrive/Documents/Projects/Insight/datasets/allData.json')
allChannelData['time'] = pd.to_datetime(allChannelData['ts'],unit='s')

networkLog = allChannelData.sort_values(by=['ts'] )
networkLog['date']=networkLog['time'].apply(lambda x : x.date())  
networkLog['date'] = pd.to_datetime(networkLog['date'])
networkLog.resample('W',on='date')


usrLog = networkLog.loc[networkLog['user']==usr]


def usr_top_chans(usrLog , nchans = 5):
    
    #endDate = startDate + pd.Timedelta(7, unit='W')
    
    chScore = pd.DataFrame()
    chanList = list(usrLog['channel'].unique())
    
    #for w in  networkLog['date'].unique():  
    for w in pd.date_range(start=startDate,end=endDate,freq='W'):

        b=usrLog.groupby(['user','channel']).count().reset_index()
        b['weight'] = b['text']
        b=b.drop(['subtype','type','ts','time','date','text'],axis =1)
        G =nx.DiGraph()
        networkG=nx.from_pandas_edgelist(b,source ='user',target='channel', create_using = G )
        networkG.add_weighted_edges_from(list(b.itertuples(index=False, name=None)))
        try:
            h,a=nx.hits(networkG)
            bib = pd.DataFrame(dict((k, a[k]) for k in chanList if k in a), index=[w])
            chScore =chScore.append(bib)
        except:
            h,a=nx.hits(networkG,tol=1e-01)
            bib = pd.DataFrame(dict((k, a[k]) for k in chanList if k in a), index=[w])
            chScore =chScore.append(bib)
                
        
        
        return (usr_chans)

#for usr in networkLog['user'].unique():
#    usrLog = networkLog.loc[networkLog['user']==usr]
#    usrLog = usrLog.sort_values(by=['ts'] )
#    startDate = usrLog['date'].min()
#    endDate = startDate + pd.Timedelta(7, unit='W')
#    usrLog['date'].max()
chScore = pd.DataFrame()
chanList = list(networkLog['channel'].unique())
for w in  networkLog['date'].unique():  
    weekLog = networkLog.loc[networkLog['date']==w]
    b=weekLog.groupby(['user','channel']).count().reset_index()
    b['weight'] = b['text']
    b=b.drop(['subtype','type','ts','time','date','text'],axis =1)
    G =nx.DiGraph()
    networkG=nx.from_pandas_edgelist(b,source ='user',target='channel', create_using = G )
    networkG.add_weighted_edges_from(list(b.itertuples(index=False, name=None)))
    try:
        h,a=nx.hits(networkG)
        bib = pd.DataFrame(dict((k, a[k]) for k in chanList if k in a), index=[w])
        chScore =chScore.append(bib)
    except:
        h,a=nx.hits(networkG,tol=1e-01)
        bib = pd.DataFrame(dict((k, a[k]) for k in chanList if k in a), index=[w])
        chScore =chScore.append(bib)

    #h,a=nx.hits(networkG,normalized=True)
    
    