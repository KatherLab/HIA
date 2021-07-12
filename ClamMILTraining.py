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
        args.feature_extract = False
        
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
        
        if args.feature_extract:
                
            imgs = os.listdir(args.datadir_train[0])
            imgs = [os.path.join(args.datadir_train[0], i) for i in imgs]
            
            args.feat_dir = os.path.join(args.projectFolder, 'FEATURES')
            ExtractFeatures(data_dir = imgs, feat_dir = args.feat_dir, batch_size = args.batch_size, target_patch_size = -1, filterData = True)
        
        
        
        lengthList , args.csvPath, labelList = SortClini_SlideTables(imagesPath = args.datadir_train, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                label = args.target_label, outputPath = args.projectFolder, reportFile = reportFile, csvName = args.csv_name)

    
        labelsList = utils.CheckForTargetType(labelList)
        
        le = preprocessing.LabelEncoder()
        labelsList = le.fit_transform(labelsList)
        
        args.num_classes = len(set(labelsList))
        args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        

        
        print('\nLoad the DataSet...')  
        
        feat_dir = args.feat_dir[0]
        dataset = Generic_MIL_Dataset(csv_path = args.csvPath,
                            data_dir = feat_dir,
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
                   
        lf = 1.0 
        
        args.result = os.path.join(args.projectFolder, 'RESULTS')
        os.makedirs(args.result, exist_ok = True)   
        args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
        os.makedirs(args.split_dir, exist_ok=True)
        
        if args.train_full:
            dataset.Create_splits(k = 1, val_num = [0] * args.num_classes, test_num = [0] * args.num_classes, label_frac = lf)
            for i in range(1):
                dataset.Set_splits()
                descriptor_df = dataset.Test_split_gen(return_descriptor = True, reportFile = reportFile, fold = i)
                
                splits = dataset.Return_splits(from_id = True)
                Save_splits(split_datasets = splits, column_keys = ['train'], filename = os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))                                        
                descriptor_df.to_csv(os.path.join(args.split_dir, 'splits_{}_descriptor.csv'.format(i))) 
                
                train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = args.split_dir + '//splits_{}.csv'.format(i), trainFull = True)    
                
                datasets = (train_dataset, val_dataset, test_dataset)
                results, test_auc, val_auc, test_acc, val_acc  = Train_MIL_CLAM(datasets, i, args, trainFull = True)
                                                                         
        else:
            num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
            val_num = np.floor(num_slides_cls * 0.1).astype(int)
            test_num = np.floor(num_slides_cls * 0.2).astype(int)                

            dataset.Create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac = lf)
            for i in range(args.k):
                dataset.Set_splits()
                descriptor_df = dataset.Test_split_gen(return_descriptor = True, reportFile = reportFile, fold = i)
                splits = dataset.Return_splits(from_id = True)
                Save_splits(splits, ['train', 'val', 'test'], os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))
                descriptor_df.to_csv(os.path.join(args.split_dir, 'splits_{}_descriptor.csv'.format(i)))           

            

            
            all_test_auc = []
            all_val_auc = []
            all_test_acc = []
            all_val_acc = []
            
            folds = args.k
            
            for i in range(folds):
                print()
                print("=========================================")
                print("====== K FOLD VALIDATION STEP => %d =======" % (i))
                print("=========================================")
                        
                #utils.Seed_torch(device, args.seed)
                train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = args.split_dir + '//splits_{}.csv'.format(i))    
                
                datasets = (train_dataset, val_dataset, test_dataset)
                results, test_auc, val_auc, test_acc, val_acc  = Train_MIL_CLAM(datasets, i, args)
                
                all_test_auc.append(test_auc)
                all_val_auc.append(val_auc)
                all_test_acc.append(test_acc)
                all_val_acc.append(val_acc)
                
                #write results to pkl
                
                case_id_test = []
                slide_id_test = []
                labelList_test = []
                probs_test = {}
                
                for i_temp in range(args.num_classes):
                    key = utils.get_key_from_value(args.target_labelDict, i_temp)
                    probs_test[key] = []
                    
                for item in list(results.keys()):
                    temp = results[item]
                    case_id_test.append(temp['case_id'])
                    slide_id_test.append(temp['slide_id'])
                    labelList_test.append(temp['label'])
                    for i_temp in range(args.num_classes):
                        key = utils.get_key_from_value(args.target_labelDict, i_temp)
                        probs_test[key].append(temp['prob'][0, i_temp])
                
                probs_test = pd.DataFrame.from_dict(probs_test)
        
                df = pd.DataFrame(list(zip(case_id_test, slide_id_test, labelList_test)), columns =['patientID', 'X', 'y'])
                df = pd.concat([df, probs_test], axis = 1)
                df.to_csv(os.path.join(args.result, 'TEST_RESULT_FOLD_' + str(i) + '.csv'), index = False)
                
                CalculatePatientWiseAUC(os.path.join(args.result, 'TEST_RESULT_FOLD_' + str(i) + '.csv'), list(set(case_id_test)), args.target_labelDict, args.result, i, clamMil = True)
    
    
            final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})
        
            save_name = 'summary.csv'
            final_df.to_csv(os.path.join(args.result, save_name))
            patientScores = []
            testResult = []
            for i in range(args.k):
                patientScores.append('TEST_RESULT_PATIENT_SCORES_' + str(i) + '.csv')
                testResult.append('TEST_RESULT_FOLD_' + str(i) + '.csv')
                
            CalculateTotalROC(args.result, patientScores, args.target_labelDict, 0.7)
            MergeResultCSV(args.result, testResult)

##############################################################################   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    