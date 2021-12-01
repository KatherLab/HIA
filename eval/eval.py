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
import os
from PIL import Image
import numpy as np
import utils.utils as utils
from shutil import copyfile
import glob

##############################################################################

def CalculatePatientWiseAUC(resultCSVPath, args, reportFile, foldcounter = None, clamMil = False):
    
    data = pd.read_csv(resultCSVPath)
    patients = list(set(data['PATIENT']))
    keys = list(args.target_labelDict.keys())
    yProbDict = {}    
    for index, key in enumerate(keys):
        patientsList = []
        yTrueList = []
        yTrueLabelList = []
        yProbList = []         
        keys_temp = keys.copy()
        keys_temp.remove(key)
        for patient in patients:
            patientsList.append(patient)
            data_temp = data.loc[data['PATIENT'] == patient]                        
            data_temp = data_temp.reset_index()            
            yTrueList.append(data_temp['yTrue'][0])
            yTrueLabelList.append(utils.get_key_from_value(args.target_labelDict, data_temp['yTrue'][0]))                        
            if not clamMil:
                dl_pred = np.where(data_temp[keys_temp].lt(data_temp[key], axis=0).all(axis=1), True, False)
                dl_pred = list(dl_pred)
                true_count = dl_pred.count(True)            
                yProbList.append(true_count / len(dl_pred)) 
            else:
                yProbList.append(np.mean(data_temp[key])) 
               
        fpr, tpr, thresholds = metrics.roc_curve(yTrueList, yProbList, pos_label = args.target_labelDict[key])
        if foldcounter:            
            print('\nAUC FOR TARGET {} IN THIS DATA SET IN FOLD {} IS: {} '.format(key, foldcounter, np.round(metrics.auc(fpr, tpr), 3)))
            reportFile.write('AUC FOR TARGET {} IN THIS DATA SET IN FOLD {} IS: {} '.format(key, foldcounter, np.round(metrics.auc(fpr, tpr), 3)) + '\n')
            path = os.path.join(args.result_dir, 'TEST_RESULT_PATIENT_BASED_FOLD_' + str(foldcounter) + '.csv')
        else:
            print('\nAUC FOR TARGET {} IN THIS DATA SET IS: {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
            reportFile.write('AUC FOR TARGET {} IN THIS DATA SET IS: {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)) + '\n')
            path = os.path.join(args.result_dir, 'TEST_RESULT_PATIENT_BASED_FULL.csv')
        
        yProbDict[key] = yProbList
    yProbDict = pd.DataFrame.from_dict(yProbDict)
    df = pd.DataFrame(list(zip(patientsList, yTrueList, yTrueLabelList)), columns =['PATIENT', 'yTrue', 'yTrueLabel'])
    df = pd.concat([df, yProbDict], axis=1)    
    df.to_csv(path, index = False)
    return path
                        
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

def CalculateTotalROC(resultsPath, results, target_labelDict, reportFile):
    
    totalData = []    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)        
    totalData = pd.concat(totalData)
    y_true = list(totalData['yTrue'])
    keys = list(target_labelDict.keys())
    
    for key in keys:
        y_pred = totalData[key]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label = target_labelDict[key])
        print('-' * 30)        
        print('TOTAL AUC FOR target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
        reportFile.write('-' * 30 + '\n')
        reportFile.write('TOTAL AUC FOR target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)) + '\n')
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
        reportFile.write('Lower Confidebnce Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)) + '\n')
        reportFile.write('Higher Confidebnce Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)) + '\n')     
        print('Lower Confidebnce Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)))        
        print('Higher Confidebnce Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)))
        
    totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_PATIENT_BASED_TOTAL.csv'), index = False)    

##############################################################################

def find_closes(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]

##############################################################################

def MergeResultCSV(resultsPath, results, milClam = False):
    
    totalData = []    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)
    totalData = pd.concat(totalData)
    if milClam:
        totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_SLIDE_BASED_TOTAL.csv'), index = False)
    else:            
        totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_TILE_BASED_TOTAL.csv'), index = False)

##############################################################################

def GenerateHighScoreTiles_Classic(totalPatientResultPath, totalResultPath, numHighScorePetients, numHighScoreTiles, target_labelDict, savePath):
                       
    patientData = pd.read_csv(totalPatientResultPath)
    tileData = pd.read_csv(totalResultPath)
    
    keys = list(target_labelDict.keys())
    for key in keys:
        dataTemp = patientData.loc[patientData['yTrueLabel'] == key]
        dataTemp = dataTemp.sort_values(by = [key], ascending = False)
        
        highScorePosPatients = list(dataTemp['PATIENT'][0 : numHighScorePetients])
                 
        fig = plt.figure(figsize=(10,10))
        i = 1
        
        path = os.path.join(savePath, key)
        os.makedirs(path, exist_ok = True)
        for index, patient in enumerate(highScorePosPatients):            
            dataTemp = tileData.loc[tileData['PATIENT'] == patient]
            dataTemp = dataTemp.sort_values(by = [key], ascending = False)
            highScorePosTiles = list(dataTemp['TilePath'][0:numHighScoreTiles])                        
            for tile in highScorePosTiles:            
                img = Image.open(tile)
                copyfile(tile, os.path.join(path, tile.split('\\')[-1]))        
                ax = plt.subplot(numHighScorePetients, numHighScoreTiles, i)
                ax.set_axis_off()
                plt.imshow(img)
                i += 1 
                
        plt.savefig(os.path.join(path,  key + '.png'))
        plt.close()

##############################################################################

def GenerateHighScoreTiles(imgsPath, totalPatientResultPath, totalResultPath, tileScorePath, coordsPath, numHighScorePetients, numHighScoreTiles,
                           target_labelDict, savePath):
                       
    patientData = pd.read_csv(totalPatientResultPath)
    
    tileScoresData = pd.read_csv(tileScorePath, low_memory = False)
    coordsData = pd.read_csv(coordsPath, low_memory = False)      
    tileScoresData = tileScoresData.rename(columns = { 'Unnamed: 0' :'FILENAME'})
    coordsData = coordsData.rename(columns = { 'Unnamed: 0' :'FILENAME'})
    
    keys = list(target_labelDict.keys())
    for key in keys:
        
        dataTemp = patientData.loc[patientData['yTrueLabel'] == key]
        dataTemp = dataTemp.sort_values(by = [key], ascending = False)        
        highScorePosPatients = list(dataTemp['PATIENT'][0 : numHighScorePetients])
                 
        fig = plt.figure(figsize=(10,10))
        sub_i = 1        
        path = os.path.join(savePath, key)
        os.makedirs(path, exist_ok = True)
        for patient in highScorePosPatients: 
            temp_scores = tileScoresData[tileScoresData['FILENAME'].str.contains(patient)]    
            temp_scores = temp_scores.dropna(axis= 'columns')
            temp_scores.reset_index(inplace = True, drop = True)
            temp_coords = coordsData[coordsData['FILENAME'].str.contains(patient)] 
            temp_coords = temp_coords.dropna(axis = 'columns')    
            temp_coords.reset_index(inplace = True, drop = True)                    
            coord_x = []
            coord_y = []
            scores = []           
            for i in range(len(temp_coords.columns) - 1):                
                j = i + 1
                sc = temp_scores.iloc[0,j].strip('][').split(', ')  
                scores.append(float(sc[target_labelDict[key]]))                
                sc = temp_coords.iloc[0,j].strip('][').split(', ')                 
                coord_x.append(int(sc[0]))
                coord_y.append(int(sc[1]))            
            
            temp = pd.DataFrame(list(zip(scores, coord_x, coord_y)), columns = [key, 'coord_x', 'coord_y'])  
            temp = temp.sort_values(by = [key], ascending = False)

            coordx = list(temp['coord_x'][0: numHighScoreTiles])
            coordy = list(temp['coord_y'][0 : numHighScoreTiles])
            tilePaths = glob.glob(os.path.join(imgsPath[0], temp_scores.iloc[0,0].split('*')[-1]))[0]            
            imgs = [os.path.join(tilePaths, i) for i in os.listdir(tilePaths)]
                             
            for item in range(len(coordx)):
                try:                        
                    doFind = [v for v in imgs if '('+ str(coordx[item]) + ',' + str(coordy[item]) + ')' in v]                         
                    img = Image.open(doFind[0])
                    copyfile(doFind[0], os.path.join(os.path.join(savePath, key), doFind[0].split('\\')[-1]))                
                    ax = plt.subplot(numHighScorePetients, numHighScorePetients, sub_i)
                    ax.set_axis_off()
                    plt.imshow(img)
                    sub_i += 1  
                except:
                    print('('+ str(coordx[item]) + ',' + str(coordy[item]) + ')')
                    
        plt.savefig(os.path.join(os.path.join(savePath, key),  key + '.png'))
        plt.close()            
            






































