#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 02:11:03 2018

@author: top
"""

import numpy as np
from random import *
import graphviz #if you use conda, need to install inadvance

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
column=['gender','age','blood type','height','weight','bmi','salary']
data=np.zeros((100,7))
for i in range(len(data)):
    #generate gender data
    data[i][0]=randint(0, 1)
    #generate age data
    data[i][1]=randint(1, 50)
    #generate blood type data
    data[i][2]=randint(0,3)
    #generate height data
    if (data[i][0]==0):
        if(data[i][1]>18):
            data[i][3]='%.2f' %gauss(175, 5)
        else:
            data[i][3]='%.2f' %(gauss(175, 5)-(18-data[i][1])*2)
    else:
        if(data[i][1]>18):
            data[i][3]='%.2f' %gauss(163, 4)
        else:
            data[i][3]='%.2f' %(gauss(163, 4)-(18-data[i][1])*1.2)
    #generate weight data        
    if (data[i][0]==0):
        if(data[i][1]>18):
            data[i][4]='%.2f' %gauss(69, 5)
        else:
            data[i][4]='%.2f' %(gauss(69, 5)-(18-data[i][1])*2)
    else:
        if(data[i][1]>18):
            data[i][4]='%.2f' %gauss(57, 4)
        else:
            data[i][4]='%.2f' %(gauss(57, 4)-(18-data[i][1])*1.2)
    #generate bmi data        
    data[i][5]='%.1f'% (data[i][4]/(data[i][3]/100)**2) 
    #generate salary data
    if (data[i][0]==0):
        if(data[i][1]>18):
            data[i][6]='%.2f' %gauss(50, 5)
        else:
            data[i][6]='%.2f' %0
    else:
        if(data[i][1]>18):
            data[i][6]='%.2f' %(gauss(40, 4))
        else:
            data[i][6]='%.2f' %0           
   
    
        
#    data[i][5]='%.2f' %uniform(1, 10)
#    data[i][6]='%.2f' %gauss(163, 5)
print('Data random Generate ')            
print(column)
print('===========================================================')
print(data)
print('===========================================================')
print('===========================================================')
print('===========================================================')
print('')
print("Accuracy of the result:")
print('==========')
x=data[:, 1:7]
y=data[:,0]
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 100)


clf_gini =  DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')

clf_gini.fit(X_train, y_train)

dot_data = tree.export_graphviz(clf_gini, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("By_Gini") 

y_pred_gini = clf_gini.predict(X_test)
print ("Accuracy by Gini is ", accuracy_score(y_test,y_pred_gini)*100)


clf_entropy =  DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')
clf_entropy.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf_entropy, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("By_Entropy")
y_pred_en = clf_entropy.predict(X_test)
print ("Accuracy by Entropy is ", accuracy_score(y_test,y_pred_en)*100)