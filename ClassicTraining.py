"""
Created on Mon Mar  8 10:14:47 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

from utils.data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, LoadTrainTestFromFolders, GetTiles
from utils.core_utils import Train_model_Classic, Validate_model_Classic
from eval.eval_Classic import PlotTrainingLossAcc, CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV, GenerateHighScoreTiles
import utils.utils as utils

from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from tqdm import tqdm
import torchvision
import numpy as np
import pandas as pd
import argparse
import torch
import os
import random
from sklearn import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def ClassicTraining(args):
    
    stats_total = {}
    stats_df = pd.DataFrame()
    
    targetLabels = args.target_labels
    
    for targetLabel in targetLabels:
        
        args.target_label = targetLabel        
        random.seed(args.seed)
        args.projectFolder = utils.CreateProjectFolder(args.project_name, args.adressExp, targetLabel, args.model_name)
        
        if os.path.exists(args.projectFolder):
            continue
        else:
            os.mkdir(args.projectFolder) 
            
        reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
        reportFile.write('**********************************************************************'+ '\n')
        reportFile.write(str(args))
        reportFile.write('\n' + '**********************************************************************'+ '\n')
        
           
        patientsList, labelsList, slidesList, clinicalTableList, slideTableList = ConcatCohorts_Classic(imagesPath = args.datadir_train,
                                                            cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,label = targetLabel, reportFile = reportFile)
        
        print('\n*** LOAD THE DATASET FOR TRAINING ***\n')  
        
        labelsList = utils.CheckForTargetType(labelsList)
        
        le = preprocessing.LabelEncoder()
        labelsList = le.fit_transform(labelsList)
        
        args.num_classes = len(set(labelsList))
        args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
      
        utils.Summarize_Classic(args, list(labelsList), reportFile)
        
        if len(patientsList) < 20:
            continue
        
        if args.train_full:
            print('\n***IT IS A FULL TRAINING FOR ' + targetLabel + ' !***\n')
            
            train_x = []
            train_y = []
            
            args.result = os.path.join(args.projectFolder, 'RESULTS')
            os.makedirs(args.result, exist_ok = True)
            args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
            os.makedirs(args.split_dir, exist_ok = True)
            
            patientID = np.array(patientsList)
            train_data = GetTiles(patients = patientID, labels = labelsList, imgsList = slidesList, label= targetLabel, 
                                  slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = False)
    
            train_x = list(train_data['tilePath'])
            train_y = list(train_data[targetLabel])
                
            df = pd.DataFrame(list(zip(train_x, train_y)), columns =['tilePath', 'label'])
            df.to_csv(os.path.join(args.split_dir, 'FULL_TRAIN' + '.csv'), index = False)
            print()  
             
            
            model_ft, input_size = utils.Initialize_model(args.model_name, args.num_classes, use_pretrained = True)
                
            params = {'batch_size': args.batch_size,
                      'shuffle': True,
                      'num_workers': 0,
                      'pin_memory' : False}
            
            train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
            traingenerator = torch.utils.data.DataLoader(train_set, **params)
            
            model_ft = model_ft.to(device)   
            
            noOfLayers = 0
            for name, child in model_ft.named_children():
                 noOfLayers += 1
            
            cut = int (args.freeze_Ratio * noOfLayers)
            
            ct = 0
            for name, child in model_ft.named_children():
                ct += 1
                if ct < cut:
                    for name2, params in child.named_parameters():
                        params.requires_grad = False
            
            print('\n*** INITIALIZE THE  OPTIMIZER***\n', end = ' ')
            optimizer = utils.get_optim(model_ft, args, params = False)
            
            criterion = nn.CrossEntropyLoss()
            print('\n*** START TRAINING ***', end = ' ')
            model, train_loss_history, train_acc_history, _, _ = Train_model_Classic(model = model_ft, trainLoaders = traingenerator,
                                             criterion = criterion, optimizer = optimizer, num_epochs = args.max_epochs, is_inception = (args.model_name == "inception"), 
                                             results_dir = args.result)
            
            torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'MODEL_Full'))
            
            df = pd.DataFrame(list(zip(train_loss_history, train_acc_history)), columns =['train_loss_history', 'train_acc_history'])
            
            df.to_csv(os.path.join(args.result, 'TRAIN_HISTORY_full' + '.csv'))
            
        else:
        
            print('\n*** IT IS A ' + str(args.k) + 'FOLD CROSS VALIDATION TRAINING FOR ' + targetLabel + '!***\n')
            
            patientID = np.array(patientsList)
            labels = np.array(labelsList)
            
            folds = args.k
            kf = StratifiedKFold(n_splits = folds, random_state = args.seed, shuffle = True)
            kf.get_n_splits(patientID, labels)
                
            args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
            os.makedirs(args.split_dir, exist_ok = True)
                
            counter = 0
            
            for train_index, test_index in kf.split(patientID, labels):

                print('**********************************************************************')
                print('START OF CROSS VALIDATION')     
                print('**********************************************************************')                
              

                testData_patientID = patientID[test_index]   
                testData_Labels = labels[test_index]
                
                if not len(patientID)<50:
                    val_index = random.choices(train_index, k = int(len(train_index) * 0.05))

                    valData_patientID = patientID[val_index]
                    valData_Labels = labels[val_index]
                
                train_index = [i for i in train_index if i not in val_index]
                
                trainData_patientID = patientID[train_index] 
                trainData_labels = labels[train_index] 
                
                print('\n*** LOAD TRAIN DATASET ***\n')
               
                train_data = GetTiles(patients = trainData_patientID, labels = trainData_labels, imgsList = slidesList, label = targetLabel, 
                                      slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = False, seed = args.seed)
                
                train_x = list(train_data['tilePath'])
                train_y = list(train_data[targetLabel])
                    
                df = pd.DataFrame(list(zip(train_x, train_y)), columns =['tilePath', 'label'])
                df.to_csv(os.path.join(args.split_dir, 'SPLIT_TRAIN_' + str(counter)+ '.csv'), index = False)
                print()   
                
                if not len(patientID)<50:
                    print('LOAD Validation DATASET\n')           
                    val_data = GetTiles(patients = valData_patientID, labels = valData_Labels, imgsList = slidesList, label = targetLabel, 
                                          slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = True, seed = args.seed)    
                    
                    val_x = list(val_data['tilePath'])   
                    val_y = list(val_data[targetLabel])
                        
                    df = pd.DataFrame(list(zip(val_x, val_y)), columns =['tilePath', 'label'])
                    df.to_csv(os.path.join(args.split_dir, 'SPLIT_VAL_' + str(counter)+ '.csv'), index = False)    
                    print()
                
                print('LOAD TEST DATASET')  
                test_data = GetTiles(patients = testData_patientID, labels = testData_Labels, imgsList = slidesList, label = targetLabel, 
                                      slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = True, seed = args.seed)  
                
                test_x = list(test_data['tilePath'])
                test_y = list(test_data[targetLabel])
                test_pid = list(test_data['PATIENT'])
                    
                df = pd.DataFrame(list(zip(test_pid, test_x, test_y)), columns = ['PATIENT', 'tilePath', 'label'])
                df.to_csv(os.path.join(args.split_dir, 'SPLIT_TEST_' + str(counter) + '.csv'), index = False)
                
                print()
                print("=========================================")
                print("====== K FOLD VALIDATION STEP => %d =======" % (counter))
                print("=========================================")
                
                model_ft, input_size = utils.Initialize_model(args.model_name, args.num_classes, use_pretrained = True)
                
                params = {'batch_size': args.batch_size,
                          'shuffle': True,
                          'num_workers': 0,
                          'pin_memory' : False}
                
                train_set = DatasetLoader_Classic(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                traingenerator = torch.utils.data.DataLoader(train_set, **params)
                
                if not len(patientID) < 50:
                    val_set = DatasetLoader_Classic(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                    valgenerator = torch.utils.data.DataLoader(val_set, **params)   
                else :
                    valgenerator = []
                params = {'batch_size': args.batch_size,
                          'shuffle': False,
                          'num_workers': 0, 
                          'pin_memory' : False}
                
                test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
                test_generator = torch.utils.data.DataLoader(test_set, **params)
                
                model_ft = model_ft.to(device)   
                
                noOfLayers = 0
                for name, child in model_ft.named_children():
                    noOfLayers += 1
                
                cut = int (args.freeze_Ratio * noOfLayers)
                
                ct = 0
                for name, child in model_ft.named_children():
                    ct += 1
                    if ct < cut:
                        for name2, params in child.named_parameters():
                            params.requires_grad = False
                            
                print('\nINIT OPTIMIZER ...', end = ' ')
                optimizer = utils.get_optim(model_ft, args, params = False)
                print('DONE!')
                
                criterion = nn.CrossEntropyLoss()
                
                print('\nSTART TRAINING ...', end = ' ')
                model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model_Classic(model = model_ft, trainLoaders = traingenerator,
                                                valLoaders = valgenerator, criterion = criterion, optimizer = optimizer, num_epochs = args.max_epochs, is_inception = (args.model_name == "inception"))
                
                print('DONE!')
                
                args.result = os.path.join(args.projectFolder, 'RESULTS')
                os.makedirs(args.result, exist_ok = True)
                
                torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'MODEL_FOLD_' + str(counter)))
                df = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)), 
                                  columns =['train_loss_history', 'train_acc_history', 'val_loss_history', 'val_acc_history'])
                
                df.to_csv(os.path.join(args.result, 'TRAIN_HISTORY_FOLD_' + str(counter) + '.csv'), index = False)
                
                # Evaluate_Performance
                
                epoch_loss, epoch_acc, predList  = Validate_model_Classic(model, test_generator, criterion)
                                
                scores = {}
                for index, key in enumerate(list(args.target_labelDict.keys())):
                    scores[key] = []
                    for item in predList:
                        scores[key].append(item[index])
                
                scores = pd.DataFrame.from_dict(scores)

                df = pd.DataFrame(list(zip(test_pid, test_x, test_y)), columns =['PATIENT', 'tilePath', 'label'])
                df = pd.concat([df, scores], axis=1)
                
                df.to_csv(os.path.join(args.result, 'TEST_RESULT_TILE_SCORES_' + str(counter) + '.csv'), index = False)
                CalculatePatientWiseAUC(resultCSVPath = os.path.join(args.result, 'TEST_RESULT_TILE_SCORES_' + str(counter) + '.csv'),
                                        uniquePatients = list(set(test_pid)), target_labelDict = args.target_labelDict, resultFolder = args.result,
                                        counter = counter , clamMil = False) 
                
                print('\n############################################################\n')
                print('')
                counter = counter + 1
                
            patientScores = []
            testResult = []
            for i in range(args.k):
                patientScores.append('TEST_RESULT_PATIENT_SCORES_' + str(i) + '.csv')
                testResult.append('TEST_RESULT_TILE_SCORES_' + str(i) + '.csv')      
                
            CalculateTotalROC(resultsPath = args.result, results = patientScores, target_labelDict =  args.target_labelDict) 
            MergeResultCSV(args.result, testResult)
                
##############################################################################





















