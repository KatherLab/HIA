"""
Created on Wed Feb 24 12:34:17 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

from dataGenerator.dataSetGenerator_ClamMil import Generic_MIL_Dataset
import utils.utils as utils
from extractFeatures import ExtractFeatures
from utils.core_utils import Train_MIL_CLAM
from utils.data_utils import ConcatCohorts_Classic
from eval.eval import CalculatePatientWiseAUC, CalculateTotalROC, MergeResultCSV
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import pandas as pd
import random
from sklearn import preprocessing
import torch

##############################################################################

def CLAM_MIL_Training(args):

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
            print('\nLoad the DataSet...')          
            dataset = Generic_MIL_Dataset(csv_path = args.csvFile,
                                          data_dir = args.feat_dir,
                                          shuffle = False, 
                                          seed = args.seed, 
                                          print_info = True,
                                          label_dict  = args.target_labelDict,
                                          patient_strat = True,
                                          label_col = args.target_label,
                                          ignore = [],
                                          reportFile = reportFile)                               
            if len(patientsList) < 20:
                continue
            
            if args.train_full:
                print('-' * 30)
                print('IT IS A FULL TRAINING FOR ' + targetLabel + '!')            
                train_data = pd.DataFrame(list(zip(patientsList, yTrue, yTrueLabel)), columns = ['PATIENT', 'yTrue', 'yTrueLabel'])                              
                val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.1)) 
                train_data = train_data[~train_data['PATIENT'].isin(list(val_data['PATIENT']))]
                train_data.reset_index(inplace = True, drop = True)
                val_data.reset_index(inplace = True, drop = True)
                df = pd.DataFrame({'train': pd.Series(train_data['PATIENT']), 'test': pd.Series([]), 'val' : pd.Series(val_data['PATIENT'])})
                df.to_csv(os.path.join(args.split_dir, 'TrainValSplit.csv'), index = False)               
                train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = os.path.join(args.split_dir, 'TrainValSplit.csv'))                 
                datasets = (train_dataset, val_dataset, test_dataset)
                model, _, _  = Train_MIL_CLAM(datasets = datasets, fold = 'FULL', args = args, trainFull = True)  
                torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModel'))                
                print()
                print('-' * 30) 
                reportFile.close()                                                                       
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

                    testPatients = patientID[test_index]   
                    trainPatients = patientID[train_index] 
                    testyTrue = yTrue[test_index]   
                    trainyTrue = yTrue[train_index] 
                    testyTrueLabel = yTrueLabel[test_index]   
                    trainyTrueLabel = yTrueLabel[train_index]
                    
                    print('GENERATE NEW TILES...\n')    
                    print('FOR TRAIN SET...\n')                         
                    train_data = pd.DataFrame(list(zip(trainPatients, trainyTrue, trainyTrueLabel)), columns = ['PATIENT', 'yTrue', 'yTrueLabel'])             
                    print('FOR VALIDATION SET...\n')  
                    val_data = train_data.groupby('yTrue', group_keys = False).apply(lambda x: x.sample(frac = 0.1))                
                    train_data = train_data[~train_data['PATIENT'].isin(list(val_data['PATIENT']))]           
                    print('FOR TEST SET...\n')
                    test_data = pd.DataFrame(list(zip(testPatients, testyTrue, testyTrueLabel)), columns = ['PATIENT', 'yTrue', 'yTrueLabel'])             
                    train_data.reset_index(inplace = True, drop = True)
                    test_data.reset_index(inplace = True, drop = True)
                    val_data.reset_index(inplace = True, drop = True)                                        

                    print('-' * 30)
                    print("K FOLD VALIDATION STEP => {}".format(foldcounter))  
                    print('-' * 30)  
                             
                    df = pd.DataFrame({'train': pd.Series(train_data['PATIENT']), 'test': pd.Series(test_data['PATIENT']), 'val' : pd.Series(val_data['PATIENT'])})
                    df.to_csv(os.path.join(args.split_dir, 'TrainTestValSplit_{}.csv'.format(foldcounter)), index = False)                                                                       
                    train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = os.path.join(args.split_dir, 'TrainTestValSplit_{}.csv'.format(foldcounter)))                      
                    datasets = (train_dataset, val_dataset, test_dataset)
                    
                    model, results, test_auc  = Train_MIL_CLAM(datasets = datasets, fold = foldcounter, args = args, trainFull = False) 
                    reportFile.write('AUC calculated by CLAM' + '\n')
                    reportFile.write(str(test_auc) + '\n')
                    reportFile.write('-' * 30 + '\n')
                    
                    patients = []
                    filaNames = []
                    yTrue_test = []
                    yTrueLabe_test = []
                    probs = {}
                    
                    for i_temp in range(args.num_classes):
                        key = utils.get_key_from_value(args.target_labelDict, i_temp)
                        probs[key] = []
                        
                    for item in list(results.keys()):
                        temp = results[item]
                        patients.append(temp['PATIENT'])
                        filaNames.append(temp['FILENAME'])
                        yTrue_test.append(temp['label'])
                        yTrueLabe_test.append(utils.get_key_from_value(args.target_labelDict, temp['label']))   
                        
                        for key in list(args.target_labelDict.keys()):
                            if args.model_name in ['clam_sb', 'clam_mb']:
                                probs[key].append(temp['prob'][0][utils.get_value_from_key(args.target_labelDict, key)])
                    
                    probs = pd.DataFrame.from_dict(probs)                        
                    df = pd.DataFrame(list(zip(patients, filaNames, yTrue_test, yTrueLabe_test)), columns =['PATIENT', 'FILENAME', 'yTrue', 'yTrueLabel'])
                    df = pd.concat([df, probs], axis = 1)
                    testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_SLIDE_BASED_FOLD_' + str(foldcounter) + '.csv')
                    df.to_csv(testResultsPath, index = False)
                    CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = foldcounter , clamMil = True, reportFile = reportFile)
                    reportFile.write('-' * 30 + '\n')                
                    foldcounter +=  1
    
                patientScoreFiles = []
                slideScoreFiles = [] 
                for i in range(args.k):
                    patientScoreFiles.append('TEST_RESULT_PATIENT_BASED_FOLD_' + str(i + 1) + '.csv')
                    slideScoreFiles.append('TEST_RESULT_SLIDE_BASED_FOLD_' + str(i + 1) + '.csv')
                    
                CalculateTotalROC(resultsPath = args.result_dir, results = patientScoreFiles, target_labelDict =  args.target_labelDict, reportFile = reportFile)
                reportFile.write('-' * 30 + '\n')
                MergeResultCSV(args.result_dir, slideScoreFiles, milClam = True)
                reportFile.close()  
            

##############################################################################         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    