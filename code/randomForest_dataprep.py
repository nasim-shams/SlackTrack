# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:06:37 2018

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
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

import networkx as nx

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UserData = pd.read_json('C:/Users/shams/OneDrive/Desktop/Insight/datasets/RFData.json')

y = np.array(UserData.loc[:,'isActvie'])
x = np.array(UserData.loc[:,['wk2','wk4','wk6','wk8','Nchans','Nusers','chanScore']])
x = np.nan_to_num(x)
y = np.nan_to_num(y)

#~~~~~~~~~~~~~~over sampling ~~~~~~~~~~~~~~~~~~~
sm = SMOTE(kind='svm')
tm = TomekLinks()
renn = RepeatedEditedNearestNeighbours()

x_res, y_res = sm.fit_sample(x, y)
x_res, y_res = tm.fit_sample(x, y)
x_res, y_res = renn.fit_sample(x, y)





train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

train_x, test_x, train_y, test_y = train_test_split(x_res, y_res, test_size=0.2)


RF = RandomForestClassifier(min_samples_leaf =5)
RF= RF.fit(train_x, train_y)

y_pred = RF.predict(test_x)

y_score = RF.predict_proba(test_x)

fpr, tpr, threshs = roc_curve(test_y, y_score[:,1])

cnf_matrix = confusion_matrix(test_y, y_pred)
print(cnf_matrix)

"""

dot_data = skt.export_graphviz(RF.estimators_[1],out_file = None,
                               class_names =['0','1'],
                               feature_names = ['wk2','wk4','wk6','wk8','Nchans','Nusers','chanScore'],
                               impurity = False, 
                               filled = True,label = 'none')


g = pydotplus.graph_from_dot_data(dot_data)

"""




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['inactive','active']
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')




