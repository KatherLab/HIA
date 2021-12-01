# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:45:05 2021

@author: Narmin Ghaffari Laleh
"""
##############################################################################

import utils.utils as utils
from utils.core_utils import Validate_model_Classic
from utils.data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, GetTiles
from eval.eval import CalculatePatientWiseAUC, GenerateHighScoreTiles_Classic
import torch.nn as nn
import torchvision
import pandas as pd
import argparse
import torch
import os
import random
from sklearn import preprocessing

##############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"L:\Experiments\TCGA_MIL_TESTFULL.txt", help = 'Adress to the experiment File')
parser.add_argument('--modelAdr', type = str, default = r"L:\Experiments\DACHS_MIL_TRAINFULL_isMSIH_1\RESULTS\bestModel", help = 'Adress to the selected model')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\nTORCH Detected: {}\n'.format(device))

##############################################################################

if __name__ == '__main__':
              
    args = utils.ReadExperimentFile(args, deploy = True)    
    torch.cuda.set_device(args.gpuNo)
    random.seed(args.seed)        
    args.target_label = args.target_labels[0]  
    args.projectFolder = utils.CreateProjectFolder(ExName = args.project_name, ExAdr = args.adressExp, targetLabel = args.target_label,
                                                   model_name = args.model_name)
    
    print('-' * 30 + '\n')
    print(args.projectFolder)
    if os.path.exists(args.projectFolder):
        print('THIS FOLDER IS ALREADY EXITS!!! PLEASE REMOVE THE FOLDER, IF YOU WANT TO RE-RUN.')
    else:
        os.makedirs(args.projectFolder, exist_ok = True)
        
    args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
    os.makedirs(args.result_dir, exist_ok = True)
    args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
    os.makedirs(args.split_dir, exist_ok = True)
       
    reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
    reportFile.write('-' * 30 + '\n')
    reportFile.write(str(args))
    reportFile.write('-' * 30 + '\n')
    
    print('\nLOAD THE DATASET FOR TESTING...\n')     
    patientsList, labelsList, args.csvFile = ConcatCohorts_Classic(imagesPath = args.datadir_test, 
                                                                  cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                                  label = args.target_label, minNumberOfTiles = args.minNumBlocks,
                                                                  outputPath = args.projectFolder, reportFile = reportFile, csvName = args.csv_name,
                                                                  patientNumber = args.numPatientToUse)                        
    labelsList = utils.CheckForTargetType(labelsList)
    
    le = preprocessing.LabelEncoder()
    labelsList = le.fit_transform(labelsList)
    
    args.num_classes = len(set(labelsList))
    args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
  
    utils.Summarize(args, list(labelsList), reportFile)
    print('-' * 30)
    print('IT IS A DEPLOYMENT FOR ' + args.target_label + '!')            
    print('GENERATE NEW TILES...')                            
    test_data = GetTiles(csvFile = args.csvFile, label = args.target_label, target_labelDict = args.target_labelDict, maxBlockNum = args.maxBlockNum, test = True)                
    test_x = list(test_data['TilePath'])
    test_y = list(test_data['yTrue'])                
    test_data.to_csv(os.path.join(args.split_dir, 'TestSplit.csv'), index = False)                      
    print()
    print('-' * 30)
            
    model, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes, feature_extract = False, use_pretrained = True)      
    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': 0,
              'pin_memory' : False}
        
    test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
    testGenerator = torch.utils.data.DataLoader(test_set, **params)
    try:    
        model.load_state_dict(torch.load(args.modelAdr))  
    except:
        model = torch.load(args.modelAdr)
        
    model.to(device) 
    criterion = nn.CrossEntropyLoss()
    
    print('START DEPLOYING...')
    print('')
    
    probsList  = Validate_model_Classic(model = model, dataloaders = testGenerator)

    probs = {}
    for key in list(args.target_labelDict.keys()):
        probs[key] = []
        for item in probsList:
            probs[key].append(item[utils.get_value_from_key(args.target_labelDict, key)])

    probs = pd.DataFrame.from_dict(probs)
    testResults = pd.concat([test_data, probs], axis = 1)
    
    testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_TILE_BASED_FULL.csv')
    testResults.to_csv(testResultsPath, index = False)
    totalPatientResultPath = CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = None ,
                                                     clamMil = False, reportFile = reportFile)  
    GenerateHighScoreTiles_Classic(totalPatientResultPath = totalPatientResultPath, totalResultPath = testResultsPath, 
                           numHighScorePetients = args.numHighScorePatients, numHighScoreTiles = args.numHighScorePatients,
                           target_labelDict = args.target_labelDict, savePath = args.result_dir)       
    reportFile.write('-' * 100 + '\n')
    print('\n')
    print('-' * 30)
    reportFile.close()

                                







                