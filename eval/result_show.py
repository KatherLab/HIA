# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:51:50 2021

@author: Narmin Ghaffari Laleh
"""

import pickle
import numpy as np
from sklearn import metrics


infile = open(r'D:\PT1PT2-CRC-DX\DUMP_Py\RESULTS\split_5_results.pkl','rb')
new_dict = pickle.load(infile)

tp = 0
fp = 0
fn = 0
tn = 0

keys = list(new_dict.keys())


for key in keys:
    temp = new_dict[key]
    if temp['prob'][0][0] > temp['prob'][0][1] and temp['label'] == 0:
        tn += 1
    elif temp['prob'][0][1] > temp['prob'][0][0] and temp['label'] == 1:
        tp += 1
    elif temp['prob'][0][1] > temp['prob'][0][0] and temp['label'] == 0:
        fp += 1
    else:
        fn += 1


msi_prob = []
y_true = []

for key in keys:
    temp = new_dict[key]
    msi_prob.append(temp['prob'][0][1])       
    y_true.append(temp['label'])


fpr, tpr, thresholds = metrics.roc_curve(y_true, msi_prob, pos_label=1)
metrics.auc(fpr, tpr)


############################################################################

import glob, os
import pickle
from sklearn import metrics

result = []
os.chdir("D:\PT1PT2-CRC-DX\DUMP_Py\RESULTS_CLAM")
for file in glob.glob("*.pkl"):
    infile = open(file,'rb')
    new_dict = pickle.load(infile)
    keys = list(new_dict.keys())

    msi_prob = []
    y_true = []
    
    for key in keys:
        temp = new_dict[key]
        msi_prob.append(temp['prob'][0][1])       
        y_true.append(temp['label'])
    
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, msi_prob, pos_label = 1)
    result.append(metrics.auc(fpr, tpr))
    
np.mean(result)    
    
    
    
    