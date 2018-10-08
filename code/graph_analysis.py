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


allChannelData = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/allData.json')
allChannelData['time'] = pd.to_datetime(allChannelData['ts'],unit='s')

networkLog = allChannelData.sort_values(by=['ts'] )
networkLog['date']=networkLog['time'].apply(lambda x : x.date())  
networkLog['date'] = pd.to_datetime(networkLog['date'])
networkLog.resample('W',on='date')


usr_daily = networkLog['date'].value_counts().sort_index()   
usr_daily= usr_daily.reindex(fill_value=0)
usr_daily.index = pd.DatetimeIndex(usr_daily.index)
usr_daily = usr_daily.resample('D').sum()
usr_weekly = usr_daily.resample('W').sum()
usr_weekly.index = pd.DatetimeIndex(usr_weekly.index)
