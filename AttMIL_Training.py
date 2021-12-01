# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:06:38 2021

@author: nghaffarilal
"""

##############################################################################

import utils.utils as utils
from extractFeatures import ExtractFeatures
from utils.data_utils import ConcatCohorts_Classic
from eval.eval import CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import pandas as pd
import random
from sklearn import preprocessing
import torch
from pathlib import Path
from fastai.vision.all import *
from models.model_Attmil import MILModel, MILBagTransform
from utils.core_utils import Train_model_AttMIL, Validate_model_AttMIL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def AttMIL_Training(args):

    targetLabels = args.target_labels
    args.feat_dir = args.feat_dir[0]
    
    for targetLabel in targetLabels:
        for repeat in range(args.repeatExperiment): 
            
            args.target_label = targetLabel        
            random.seed(args.seed)
            args.projectFolder = utils.CreateProjectFolder(args.project_name, args.adressExp, targetLabel, args.model_name, repeat+1)
            print(args.projectFolder)
            if os.path.exists(args.projectFolder):
                continue
            else:
                os.mkdir(args.projectFolder) 
                
            args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
            os.makedirs(args.result_dir, exist_ok = True)
            args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
            os.makedirs(args.split_dir, exist_ok = True)
                    
            reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
            reportFile.write('-' * 30 + '\n')
            reportFile.write(str(args))
            reportFile.write('-' * 30 + '\n')            
            if args.extractFeature:                                      
                imgs = os.listdir(args.datadir_train[0])
                imgs = [os.path.join(args.datadir_train[0], i) for i in imgs]                            
                ExtractFeatures(data_dir = imgs, feat_dir = args.feat_dir, batch_size = args.batch_size, target_patch_size = -1, filterData = True)
            print('\nLOAD THE DATASET FOR TRAINING...\n')     
            patientsList, labelsList, args.csvFile = ConcatCohorts_Classic(imagesPath = args.datadir_train, 
                                                                          cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                                          label = targetLabel, minNumberOfTiles = args.minNumBlocks,
                                                                          outputPath = args.projectFolder, reportFile = reportFile, csvName = args.csv_name,
                                                                          patientNumber = args.numPatientToUse)  


            yTrueLabel = utils.CheckForTargetType(labelsList)            
            le = preprocessing.LabelEncoder()
            yTrue = le.fit_transform(yTrueLabel)            
            args.num_classes = len(set(yTrue))
            args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))                  
            utils.Summarize(args, list(yTrue), reportFile)                                  
            if len(patientsList) < 20:
                continue
            if args.train_full:                
                print('-' * 30)
                print('IT IS A FULL TRAINING FOR ' + targetLabel + '!')            
                train_data = pd.read_csv(args.csvFile)                             
                val_data = train_data.groupby(args.target_label, group_keys = False).apply(lambda x: x.sample(frac = 0.1)) 
                train_data['is_valid'] = train_data.PATIENT.isin(val_data['PATIENT'])
                train_data['SlideAdr'] = [i.replace('BLOCKS_NORM_MACENKO', 'FEATURES') for i in train_data['SlideAdr']]
                train_data['SlideAdr'] = [Path(i + '.pt') for i in train_data['SlideAdr']]
                train_data.to_csv(os.path.join(args.split_dir, 'TrainValSplit.csv'), index = False)
                
                dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                                   get_x = ColReader('SlideAdr'),
                                   get_y = ColReader(args.target_label),
                                   splitter = ColSplitter('is_valid'),
                                   item_tfms = MILBagTransform(train_data[train_data.is_valid].SlideAdr, 4096))
                dls = dblock.dataloaders(train_data, bs = args.batch_size)
                weight = train_data[args.target_label].value_counts().sum() / train_data[args.target_label].value_counts()
                weight /= weight.sum()
                weight = torch.tensor(list(map(weight.get, dls.vocab)))
                criterion = CrossEntropyLossFlat(weight = weight.to(torch.float32))
                model = MILModel(1024, args.num_classes)
                model = model.to(device)
                criterion.to(device)
                optimizer = utils.get_optim(model, args, params = False)
                model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_AttMIL(model = model, trainLoaders = dls.train, 
                                                 valLoaders = dls.valid, criterion = criterion, optimizer = optimizer, args = args, fold = 'FULL')                            
                torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModel')) 
                history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_acc_history, val_loss_history)), 
                                  columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])                
                history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FULL' + '.csv'), index = False)
                print()
                print('-' * 30)                 
            else:
                        
                print('IT IS A ' + str(args.k) + 'FOLD CROSS VALIDATION TRAINING FOR ' + targetLabel + '!')
                patientID = np.array(patientsList)
                yTrue = np.array(yTrue)
                yTrueLabel = np.array(yTrueLabel)
            
                folds = args.k
                kf = StratifiedKFold(n_splits = folds, random_state = args.seed, shuffle = True)
                kf.get_n_splits(patientID, yTrue)
                
                foldcounter = 1
            
                for train_index, test_index in kf.split(patientID, yTrue):

                    data = pd.read_csv(args.csvFile)  
                    test_patients = patientID[test_index]                   
                    train_patients = patientID[train_index]     
                    train_data = data[data['PATIENT'].isin(train_patients)]
                    train_data.reset_index(inplace = True, drop = True)
                    test_data = data[data['PATIENT'].isin(test_patients)]
                    test_data.reset_index(inplace = True, drop = True)
                       
                    val_data = train_data.groupby(args.target_label, group_keys = False).apply(lambda x: x.sample(frac = 0.1)) 
                    train_data['is_valid'] = train_data.PATIENT.isin(val_data['PATIENT'])
                    train_data['SlideAdr'] = [i.replace('BLOCKS_NORM_MACENKO', 'FEATURES') for i in train_data['SlideAdr']]
                    train_data['SlideAdr'] = [Path(i + '.pt') for i in train_data['SlideAdr']]
                    train_data.to_csv(os.path.join(args.split_dir, 'TrainValSplit.csv'), index = False)                   
                    test_data['SlideAdr'] = [i.replace('BLOCKS_NORM_MACENKO', 'FEATURES') for i in test_data['SlideAdr']]
                    test_data['SlideAdr'] = [Path(i + '.pt') for i in test_data['SlideAdr']]
                    test_data.to_csv(os.path.join(args.split_dir, 'TestSplit.csv'), index = False)  
                    
                    print('-' * 30)
                    print("K FOLD VALIDATION STEP => {}".format(foldcounter))  
                    print('-' * 30) 
                    
                    dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                                       get_x = ColReader('SlideAdr'),
                                       get_y = ColReader(args.target_label),
                                       splitter = ColSplitter('is_valid'),
                                       item_tfms = MILBagTransform(train_data[train_data.is_valid].SlideAdr, 4096))
                    dls = dblock.dataloaders(train_data, bs = args.batch_size)
                    weight = train_data[args.target_label].value_counts().sum() / train_data[args.target_label].value_counts()
                    weight /= weight.sum()
                    weight = torch.tensor(list(map(weight.get, dls.vocab)))
                    criterion = CrossEntropyLossFlat(weight = weight.to(torch.float32))
                    model = MILModel(1024, args.num_classes)
                    model = model.to(device)
                    criterion.to(device)
                    optimizer = utils.get_optim(model, args, params = False)
                
                    print('\n')
                    print('START TRAINING ...')
                    model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_AttMIL(model = model, trainLoaders = dls.train, 
                                                     valLoaders = dls.valid, criterion = criterion, optimizer = optimizer, args = args, fold = str(foldcounter))            
                    print('-' * 30)
                
                    torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModelFold' + str(foldcounter)))
                    history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)), 
                                  columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])
                
                    history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FOLD_' + str(foldcounter) + '.csv'), index = False)
                    print('\nSTART EVALUATION ON TEST DATA SET ...', end = ' ')
                    
                    model.load_state_dict(torch.load(os.path.join(args.projectFolder, 'RESULTS', 'bestModelFold' + str(foldcounter))))
                    model = model.to(device)
                    test_dl = dls.test_dl(test_Data)
                    probsList  = Validate_model_AttMIL(model = model, dataloaders = test_dl)

                    probs = {}
                    for key in list(args.target_labelDict.keys()):
                        probs[key] = []
                        for item in probsList:
                            probs[key].append(item[utils.get_value_from_key(args.target_labelDict, key)])
                
                    probs = pd.DataFrame.from_dict(probs)
                    test_data = test_data.rename(columns = {args.target_label: 'yTrueLabel'})
                    test_data['yTrue'] = [utils.get_value_from_key(args.target_labelDict, i) for i in test_data['yTrueLabel']] 
                    
                    testResults = pd.concat([test_data, probs], axis = 1)                    
                    testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_SLIDE_BASED_FOLD_' + str(foldcounter) + '.csv')
                    testResults.to_csv(testResultsPath, index = False)
                    CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = foldcounter , clamMil = False, reportFile = reportFile)                         
                    reportFile.write('-' * 30 + '\n')                
                    foldcounter +=  1               
                patientScoreFiles = []
                tileScoreFiles = []                
                for i in range(args.k):
                    patientScoreFiles.append('TEST_RESULT_PATIENT_BASED_FOLD_' + str(i+1) + '.csv')
                    tileScoreFiles.append('TEST_RESULT_TILE_BASED_FOLD_' + str(i+1) + '.csv')      
                CalculateTotalROC(resultsPath = args.result_dir, results = patientScoreFiles, target_labelDict =  args.target_labelDict, reportFile = reportFile) 
                reportFile.write('-' * 30 + '\n')
                MergeResultCSV(args.result_dir, tileScoreFiles)
                reportFile.close()
             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        