# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:34:17 2021

@author: Narmin Ghaffari Laleh
"""
##############################################################################

from dataGenerator.dataset_generic import Generic_MIL_Dataset, Save_splits
import utils.utils as utils
from extractFeatures import ExtractFeatures
from utils.core_utils import Train_MIL_CLAM
from utils.data_utils import SortClini_SlideTables, ConcatCohorts_Classic
from eval.eval_Classic import CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV
from sklearn.model_selection import StratifiedKFold

import numpy as np
import os
import pandas as pd
import random
import torch
from sklearn import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def ClamMILTraining(args):

    targetLabels = args.target_labels
    for targetLabel in targetLabels:
        
        args.target_label = targetLabel        
        random.seed(args.seed)
        args.projectFolder = utils.CreateProjectFolder(args.project_name, args.adressExp, targetLabel, args.model_name)
        
        if os.path.exists(args.projectFolder):
            continue
        else:
            os.mkdir(args.projectFolder) 
        
        args.feat_dir = args.feat_dir[0]
        reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
        reportFile.write('**********************************************************************'+ '\n')
        reportFile.write(str(args))
        reportFile.write('\n' + '**********************************************************************'+ '\n')
                
        if args.feature_extract:
            print('###############################')
            imgs = os.listdir(args.datadir_train[0])
            imgs = [os.path.join(args.datadir_train[0], i) for i in imgs]
                        
            ExtractFeatures(data_dir = imgs, feat_dir = args.feat_dir, batch_size = args.batch_size, target_patch_size = -1, filterData = True,self_supervised = args.self_supervised)
        
        
        
        patientList, args.csvPath, labelList = SortClini_SlideTables(imagesPath = args.datadir_train, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                label = args.target_label, outputPath = args.projectFolder, reportFile = reportFile, csvName = args.csv_name)

        patientList = list(set(patientList))
        
        print('TOTAL NUMBER OF PATIENTS:{}'.format(len(patientList)))
        labelsList = utils.CheckForTargetType(labelList)
        
        le = preprocessing.LabelEncoder()
        labelsList = le.fit_transform(labelsList)
        
        args.num_classes = len(set(labelsList))
        args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        

        utils.Summarize_Classic(args, list(labelsList), reportFile)
        
        print('\nLoad the DataSet...')  
        
        dataset = Generic_MIL_Dataset(csv_path = args.csvPath,
                            data_dir = args.feat_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict  = args.target_labelDict,
                            patient_strat = True,
                            label_col = args.target_label,
                            ignore = [],
                            normalize_targetNum = args.normalize_targetNum,
                            normalize_method = 'DownSample',
                            reportFile = reportFile)
                           
        args.result = os.path.join(args.projectFolder, 'RESULTS')
        os.makedirs(args.result, exist_ok = True)   
        args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
        os.makedirs(args.split_dir, exist_ok=True)
        
        if args.train_full:
            for i in range(1):
                df = pd.DataFrame({'train': pd.Series(patientList), 'test': pd.Series([]), 'val' : pd.Series([])})
                df.to_csv(os.path.join(args.split_dir, 'TrainFull.csv'))
                
                train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = os.path.join(args.split_dir, 'TrainFull.csv'))  
                
                datasets = (train_dataset, val_dataset, test_dataset)
                results, test_auc, val_auc, test_acc, val_acc  = Train_MIL_CLAM(datasets, i, args, trainFull = True)
                                                                         
        else:
            
            folds = args.k
            kf = StratifiedKFold(n_splits = folds, random_state = args.seed, shuffle = True)
            kf.get_n_splits(patientList, labelsList)
            
            i = 0

            for train_index, test_index in kf.split(patientList, labelsList):
                                                            
                print('**********************************************************************')
                print('START OF CROSS VALIDATION')     
                print('**********************************************************************')                
              
                patientList = np.array(patientList)
                labelsList = np.array(labelsList)
                
                testData_patientID = patientList[test_index]   
                print(len(testData_patientID))
                val_index = random.choices(train_index, k = int(len(train_index) * 0.5))

                valData_patientID = patientList[val_index]
                
                train_index = [i for i in train_index if i not in val_index]
                
                trainData_patientID = patientList[train_index] 
                
                df = pd.DataFrame({'train': pd.Series(trainData_patientID), 'test': pd.Series(testData_patientID), 'val' : pd.Series(valData_patientID)})
                df.to_csv(os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))
                                               
                
                train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = args.split_dir + '//splits_{}.csv'.format(i))  
                
                datasets = (train_dataset, val_dataset, test_dataset)
                patient_results, aucList  = Train_MIL_CLAM(datasets = datasets, cur = i, args = args)
                aucs.append(aucList)
                                            
                case_id_test = []
                slide_id_test = []
                labelList_test = []
                probs_test = {}
                
                for i_temp in range(args.num_classes):
                    key = utils.get_key_from_value(args.target_labelDict, i_temp)
                    probs_test[key] = []
                    
                for item in list(patient_results.keys()):
                    temp = patient_results[item]
                    case_id_test.append(temp['case_id'])
                    slide_id_test.append(temp['slide_id'])
                    labelList_test.append(temp['label'])
                    
                    for i_temp in range(args.num_classes):
                        key = utils.get_key_from_value(args.target_labelDict, i_temp)
                        probs_test[key].append(temp['prob'][0, i_temp])
                
                probs_test = pd.DataFrame.from_dict(probs_test)
                    
                df = pd.DataFrame(list(zip(case_id_test, slide_id_test, labelList_test)), columns =['PATIENT', 'slideName', 'label'])
                df = pd.concat([df, probs_test], axis = 1)
                df.to_csv(os.path.join(args.result, 'TEST_RESULT_SLIDE_BASED_FOLD_' + str(i) + '.csv'), index = False)
    
                returnList = CalculatePatientWiseAUC(resultCSVPath = os.path.join(args.result, 'TEST_RESULT_SLIDE_BASED_FOLD_' + str(i) + '.csv'), uniquePatients = list(set(case_id_test)), 
                                                     target_labelDict = args.target_labelDict, 
                                                     resultFolder = args.result, counter = i, clamMil = True)
                                
                for item in returnList:
                    reportFile.write(item + '\n')
                reportFile.write('**********************************************************************' + '\n')
                i += 1
    
            patientScores = []
            testResult = []
            for i in range(args.k):
                patientScores.append('TEST_RESULT_PATIENT_BASED_FOLD_' + str(i) + '.csv')
                testResult.append('TEST_RESULT_SLIDE_BASED_FOLD_' + str(i) + '.csv')
                
            returnList = CalculateTotalROC(args.result, patientScores, args.target_labelDict)
            for item in returnList:
                reportFile.write(item + '\n')
            reportFile.write('**********************************************************************' + '\n')
            
            MergeResultCSV(args.result, testResult)
##############################################################################   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
