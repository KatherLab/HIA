# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:14:47 2021
@author: Narmin Ghaffari Laleh
"""

#%% 
from utils.data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, GetTiles
from utils.core_utils import Train_model_Classic, Validate_model_Classic
from eval.eval import CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV
import utils.utils as utils

from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import torch
import os
import random
from pathlib import Path
from sklearn import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
def Classic_Training(args):
        
    targetLabels = args.target_labels
    for targetLabel in targetLabels:
        for repeat in range(args.repeatExperiment): 
            args.target_label = targetLabel        
            random.seed(args.seed)
            args.projectFolder = utils.CreateProjectFolder(ExName = args.project_name, ExAdr = args.adressExp, targetLabel = args.target_label,
                                                           model_name = args.model_name, repeat = repeat + 1)
            print('-' * 30 + '\n')
            print(args.projectFolder)
            if args.projectFolder.exists():
                continue
            else:
                os.mkdir(args.projectFolder) 
                
            args.result_dir = Path(args.projectFolder, 'RESULTS')
            os.makedirs(args.result_dir, exist_ok = True)
            args.split_dir = Path(args.projectFolder, 'SPLITS')
            os.makedirs(args.split_dir, exist_ok = True)
               
            reportFile  = open(Path(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
            
            reportFile.write('-' * 30 + '\n')
            reportFile.write(str(args))
            reportFile.write('-' * 30 + '\n')
            
            print('\nLOAD THE DATASET FOR TRAINING...\n')     
            args.csvPath = ConcatCohorts_Classic(imagesPath = args.datadir_train, 
                                                                          cliniTablePath = args.clini_dir,
                                                                          slideTablePath = args.slide_dir,
                                                                          label = targetLabel, 
                                                                          minNumberOfTiles = args.minNumBlocks,
                                                                          outputPath = args.projectFolder,
                                                                          reportFile = reportFile,
                                                                          csvName = args.csv_name,
                                                                          patientNumber = args.numPatientToUse)                        
            
            dataset = pd.read_csv(args.csvPath)
            labels = utils.CheckForTargetType(dataset[args.target_label])
            
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)
            
            args.num_classes = len(set(labels))
            args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
            utils.Summarize(args, list(labels), reportFile)
        
        
            if len(patientsList) < 20:
                continue
            if args.train_full:
                print('IT IS A FULL TRAINING FOR ' + targetLabel + '!')            
                print('GENERATE NEW TILES...')                            
                train_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = args.target_labelDict, maxBlockNum = args.maxBlockNum, test = False)                
                train_x = list(train_data['TilePath'])
                train_y = list(train_data['yTrue'])  
                if args.early_stopping:
                    val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.1))                
                    val_x = list(val_data['TilePath']) 
                    val_y = list(val_data['yTrue'])                  
                    train_data = train_data[~train_data['TilePath'].isin(val_x)]
                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue']) 
                    val_data.to_csv(os.path.join(args.split_dir, 'ValSplit.csv'), index = False)
                else:
                    valGenerator = []
                    
                train_data.to_csv(os.path.join(args.split_dir, 'TrainSplit.csv'), index = False)                                              
                print()
                print('-' * 30)
                
                model, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes, feature_extract = False, use_pretrained = True)
                model.to(device) 
                
                params = {'batch_size': args.batch_size,
                          'shuffle': True,
                          'num_workers': 0,
                          'pin_memory' : False}
            
                train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                trainGenerator = torch.utils.data.DataLoader(train_set, **params)
                if args.early_stopping:
                    val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                    valGenerator = torch.utils.data.DataLoader(val_set, **params)
                            
                noOfLayers = 0
                for name, child in model.named_children():
                     noOfLayers += 1            
                cut = int (args.freeze_Ratio * noOfLayers)                
                ct = 0
                for name, child in model.named_children():
                    ct += 1
                    if ct < cut:
                        for name2, params in child.named_parameters():
                            params.requires_grad = False
            
                optimizer = utils.get_optim(model, args, params = False)            
                criterion = nn.CrossEntropyLoss()
                print('\nSTART TRAINING ...', end = ' ')
                model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_Classic(model = model,
                                                 trainLoaders = trainGenerator, valLoaders = valGenerator,
                                                 criterion = criterion, optimizer = optimizer, args = args, fold = 'FULL')            
                print('-' * 30)
                        
                torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModel'))                
                history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_acc_history, val_loss_history)), 
                                  columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])                
                history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FULL' + '.csv'), index = False)
                reportFile.close()
            
            else:
                
                print('IT IS A ' + str(args.k) + 'FOLD CROSS VALIDATION TRAINING FOR ' + targetLabel + '!')
                patientID = np.array(patientsList)
                labels = np.array(labelsList)
            
                folds = args.k
                kf = StratifiedKFold(n_splits = folds, random_state = args.seed, shuffle = True)
                kf.get_n_splits(patientID, labels)
                
                foldcounter = 1
            
                for train_index, test_index in kf.split(patientID, labels):
                  
                    testPatients = patientID[test_index]   
                    trainPatients = patientID[train_index] 
                    
                    print('GENERATE NEW TILES...\n')    
                    print('FOR TRAIN SET...\n')                         
                    train_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = args.target_labelDict, maxBlockNum = args.maxBlockNum, test = False, filterPatients = trainPatients)                
                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue']) 
                    print('FOR VALIDATION SET...\n')  
                    val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.1))                
                    val_x = list(val_data['TilePath']) 
                    val_y = list(val_data['yTrue'])                  
                    train_data = train_data[~train_data['TilePath'].isin(val_x)]
                    train_x = list(train_data['TilePath'])
                    train_y = list(train_data['yTrue']) 
                    print('FOR TEST SET...\n')
                    test_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = args.target_labelDict, maxBlockNum = args.maxBlockNum, test = True, filterPatients = testPatients)                                    
                    test_x = list(test_data['TilePath'])
                    test_y = list(test_data['yTrue']) 
                    
                    test_data.to_csv(os.path.join(args.split_dir, 'TestSplit_' + str(foldcounter) + '.csv'), index = False)
                    train_data.to_csv(os.path.join(args.split_dir, 'TrainSplit_' + str(foldcounter) + '.csv'), index = False)
                    val_data.to_csv(os.path.join(args.split_dir, 'ValSplit_' + str(foldcounter) + '.csv'), index = False)                       

                    print('-' * 30)
                    print("K FOLD VALIDATION STEP => {}".format(foldcounter))  
                    print('-' * 30)  
                    
                    model, input_size = utils.Initialize_model(args.model_name, args.num_classes, feature_extract = False, use_pretrained = True)
                    model.to(device)
                    params = {'batch_size': args.batch_size,
                              'shuffle': True,
                              'num_workers': 0,
                              'pin_memory' : False}
                
                    train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                    trainGenerator = torch.utils.data.DataLoader(train_set, **params)
                
                    val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                    valGenerator = torch.utils.data.DataLoader(val_set, **params)   

                    params = {'batch_size': args.batch_size,
                              'shuffle': False,
                              'num_workers': 0, 
                              'pin_memory' : False}
                
                    test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
                    testGenerator = torch.utils.data.DataLoader(test_set, **params)
                                      
                    noOfLayers = 0
                    for name, child in model.named_children():
                        noOfLayers += 1
                    
                    cut = int (args.freeze_Ratio * noOfLayers)
                    
                    ct = 0
                    for name, child in model.named_children():
                        ct += 1
                        if ct < cut:
                            for name2, params in child.named_parameters():
                                params.requires_grad = False
                            
                    optimizer = utils.get_optim(model, args, params = False)                
                    criterion = nn.CrossEntropyLoss()
                
                    print('\n')
                    print('START TRAINING ...')
                    model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_Classic(model = model, trainLoaders = trainGenerator, valLoaders = valGenerator,
                                                     criterion = criterion, optimizer = optimizer, args = args, fold = str(foldcounter))            
                    print('-' * 30)
                
                    torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModelFold' + str(foldcounter)))
                    history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)), 
                                  columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])
                
                    history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FOLD_' + str(foldcounter) + '.csv'), index = False)
                    print('\nSTART EVALUATION ON TEST DATA SET ...', end = ' ')
                    
                    model.load_state_dict(torch.load(os.path.join(args.projectFolder, 'RESULTS', 'bestModelFold' + str(foldcounter))))
                    probsList  = Validate_model_Classic(model = model, dataloaders = testGenerator)

                    probs = {}
                    for key in list(args.target_labelDict.keys()):
                        probs[key] = []
                        for item in probsList:
                            probs[key].append(item[utils.get_value_from_key(args.target_labelDict, key)])
                
                    probs = pd.DataFrame.from_dict(probs)
                    testResults = pd.concat([test_data, probs], axis = 1)                    
                    testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_TILE_BASED_FOLD_' + str(foldcounter) + '.csv')
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
                    
##############################################################################





















