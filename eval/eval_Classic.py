# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 08:37:11 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from PIL import Image
#import openslide as ops
import numpy as np
import utils.utils as utils
from scipy.stats import ttest_ind
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
##############################################################################

def CalculatePatientWiseAUC(resultCSVPath, uniquePatients, target_labelDict, resultFolder, counter, clamMil = False):
    
    returnList = []
    data = pd.read_csv(resultCSVPath)

    keys = list(target_labelDict.keys())
    y_pred_dict = {}
    
    for index, key in enumerate(keys):
        patients = []
        y_true = []
        y_true_label = []
        y_pred = [] 
        
        keys_temp = keys.copy()
        keys_temp.remove(key)
        for pi in uniquePatients:
            patients.append(pi)
            data_temp = data.loc[data['patientID'] == pi]                        
            data_temp = data_temp.reset_index()
            
            y_true.append(data_temp['y'][0])
            y_true_label.append(utils.get_key_from_value(target_labelDict, data_temp['y'][0]))
                        
            if not clamMil:
                dl_pred = np.where(data_temp[keys_temp].lt(data_temp[key], axis=0).all(axis=1), True, False)
                dl_pred = list(dl_pred)
                true_count = dl_pred.count(True)            
                y_pred.append(true_count / len(dl_pred)) 
            else:
                y_pred.append(np.mean(data_temp[key])) 
               
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label = target_labelDict[key])
        print('AUC FOR TARGET {} IN THIS DATA SET IN FOLD {} IS: {} '.format(key, counter, metrics.auc(fpr, tpr)))
        returnList.append('AUC FOR TARGET {} IN THIS DATA SET IN FOLD {} IS: {} '.format(key, counter, metrics.auc(fpr, tpr)))
        
        y_pred_dict[key] = y_pred

    y_pred_dict = pd.DataFrame.from_dict(y_pred_dict)

    df = pd.DataFrame(list(zip(patients, y_true, y_true_label)), columns =['patients', 'y_true', 'y_true_label'])
    df = pd.concat([df, y_pred_dict], axis=1)    
    df.to_csv(os.path.join(resultFolder, 'TEST_RESULT_PATIENT_SCORES_' + str(counter) + '.csv'), index = False)
    return returnList
    
    
                
##############################################################################

def PlotTrainingLossAcc(train_loss_history, train_acc_history):
    
    plt.figure()
    plt.plot(range(len(train_loss_history)), train_loss_history)
    plt.xlabel('Epochs', fontsize = 30)  
    plt.xlabel('Train_Loss', fontsize = 30)

    plt.figure()
    plt.plot(range(len(train_acc_history)), train_acc_history)
    plt.xlabel('Epochs', fontsize = 30)  
    plt.xlabel('Train_Accuracy', fontsize = 30)

##############################################################################

def PlotBoxPlot(y_true, y_pred):
    fig, ax = plt.subplots()
    sns.boxplot(x = y_true, y = y_pred)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xlabel('CLASSES', fontsize = 30)  
    plt.ylabel('SCORES', fontsize = 30)  
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)

##############################################################################

def PlotROCCurve(y_true, y_pred):

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    fig, ax = plt.subplots()
    plt.title('Receiver Operating Characteristic', fontsize = 30)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate', fontsize = 30)
    plt.xlabel('False Positive Rate', fontsize = 30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

##############################################################################

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

##############################################################################

def CalculateTotalROC(resultsPath, results, target_labelDict):
    
    totalData = []
    returnList = []
    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)
    totalData = pd.concat(totalData)
    y_true = list(totalData['y_true'])
    keys = list(target_labelDict.keys())
    
    for key in keys:
        y_pred = totalData[key]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label = target_labelDict[key])
        print('TOTAL AUC FOR target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
        returnList.append('TOTAL AUC For target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
        auc_values = []
        nsamples = 1000
        rng = np.random.RandomState(666)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i in range(nsamples):
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_pred[indices])) < 2 or np.sum(y_true[indices]) == 0:
                continue    
            fpr, tpr, thresholds = metrics.roc_curve(y_true[indices], y_pred[indices], pos_label = target_labelDict[key])
            auc_values.append(metrics.auc(fpr, tpr))
        
        auc_values = np.array(auc_values)
        auc_values.sort()
        
        
        returnList.append('Lower Confidebnce Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)))
        returnList.append('Higher Confidebnce Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)))        
        print('Lower Confidebnce Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)))        
        print('Higher Confidebnce Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)))
        
    totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULTS_PATIENT_SCORES_TOTAL.csv'), index = False)
    return returnList
    

##############################################################################

def find_closes(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]

##############################################################################

def MergeResultCSV(resultsPath, results):
    
    totalData = []    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)
    totalData = pd.concat(totalData)
    totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_TOTAL.csv'))

##############################################################################

def GenerateHighScoreTiles(totalPatientResultPath, totalResultPath, numHighScorePetients, numHighScoreBlocks, targetColName):
    
    subplot_x = (numHighScoreBlocks)/5
    
    data = pd.read_csv(totalPatientResultPath)
    dataTemp = data.loc[data['y_true'] == 1]
    dataTemp = dataTemp.sort_values(by = [targetColName])
    highScorePosPatients = list(dataTemp['patients'][-numHighScorePetients:])
    highScorePos = list(dataTemp[targetColName][-numHighScorePetients:])
    
    results = pd.read_csv(totalResultPath)
    ex = totalPatientResultPath.split('\\')[3]
    
    temp = totalResultPath
    outputFolder = temp.replace(totalResultPath.split('\\')[-1], 'HIGH_SCORE_TILES' + '_' + ex)
    os.makedirs(outputFolder, exist_ok = True)        
        
    
    for index, patient in enumerate(highScorePosPatients):
        dataTemp = results.loc[results['patientID'] == patient]
        dataTemp = dataTemp.sort_values(by = [targetColName])
        highScorePosTiles = list(dataTemp['X'][-numHighScoreBlocks:])
        fig = plt.figure(figsize=(20,20))     
        fig.suptitle(patient + '_' + ex, fontsize=20)

        i = 1
        for tile in highScorePosTiles:
            img = Image.open(tile)
            ax = plt.subplot(subplot_x, 5, i)
            ax.set_axis_off()
            plt.imshow(img)
            i += 1        
        plt.savefig(os.path.join(outputFolder, patient + '_' + str(round(highScorePos[index], 3)) + '.png'))
        
##############################################################################

def GenerateHeatmaps(wsiImgPath, thumbImgPath, tileScoreCSVPath):
    
    slide = ops.OpenSlide(wsiImgPath)   
    shape_slide = slide.dimensions
    
    thumbImg = Image.open(thumbImgPath)
    thumbImg = np.array(thumbImg)
    shape_thumb = thumbImg.shape
    
    x_ratio = shape_thumb[1] / shape_slide[0]
    y_ratio = shape_thumb[0] / shape_slide[1]
   
    data = pd.read_csv(tileScoreCSVPath)
    sl = thumbImgPath.split('\\')[-1]
    sl = sl.replace('_thumb.jpg', '')
    
    temp = list(data['X'])
    indices = [temp.index(i) for i in temp if i.split('\\')[-2] == sl]
    data_temp = data.iloc[indices]
    
    coords = list(data_temp['X'])
    scores = list(data_temp['score_1'])
    coords_x = []
    coords_y = []
    for index , item in enumerate(coords):
        
        c = item[item.find("(")+1:item.find(")")]
        x = int(c.split(',')[0])
        y = int(c.split(',')[1])
        
        coords_x.append(int(x * x_ratio))
        coords_y.append(int(y * y_ratio))
        
        x = coords_x[index]
        y = coords_y[index]
        
        l = int(512 * x_ratio)
        thumbImg[y : y+l, x : x+l, :] = round(scores[index], 5) * 10000
    plt.figure()
    plt.imshow(thumbImg)












































