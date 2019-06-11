# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:43:45 2018

@author: shams
"""
import pandas as pd
import networkx as nx

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
        
    #chScore.iloc[0:nchans].index.values.tolist()
    
    
    
    