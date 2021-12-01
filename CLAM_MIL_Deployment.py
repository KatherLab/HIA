# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:37:23 2021

@author: Narmin Ghaffari Laleh
"""
##############################################################################

import utils.utils as utils
import torch.nn as nn
import numpy as np
import argparse
import torch
import os
import random
from sklearn import preprocessing
from dataGenerator.dataSetGenerator_ClamMil import Generic_MIL_Dataset
from utils.data_utils import ConcatCohorts_Classic
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
from utils.data_utils import Get_split_loader
from tqdm import tqdm
import pandas as pd
from eval.eval import CalculatePatientWiseAUC, GenerateHighScoreTiles
from extractFeatures import ExtractFeatures

##############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"L:\Experiments\TCGA_CLAM_TESTFULL.txt", help = 'Adress to the experiment File')
parser.add_argument('--modelAdr', type = str, default = r"L:\Experiments\DACHS_CLAM_TRAINFULL_isMSIH_1\RESULTS\bestModel", help = 'Adress to the selected model')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\nTORCH Detected: {}\n'.format(device))

##############################################################################

if __name__ == '__main__':

    args = utils.ReadExperimentFile(args, deploy = True)    
    random.seed(args.seed)        
    args.target_label = args.target_labels[0] 
    args.feat_dir = args.feat_dir[0]
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
    if args.extractFeature:                                      
        imgs = os.listdir(args.datadir_train[0])
        imgs = [os.path.join(args.datadir_train[0], i) for i in imgs]                            
        ExtractFeatures(data_dir = imgs, feat_dir = args.feat_dir, batch_size = args.batch_size, target_patch_size = -1, filterData = True)
    
    print('\nLOAD THE DATASET FOR TRAINING...\n')     
    patientsList, labelsList, args.csvFile = ConcatCohorts_Classic(imagesPath = args.datadir_test, 
                                                                  cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                                  label = args.target_label, minNumberOfTiles = args.minNumBlocks,
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
            
            

    print('IT IS A FULL TESTING FOR ' + args.target_label + '!')            
    test_data = pd.DataFrame(list(zip(patientsList, yTrue, yTrueLabel)), columns = ['PATIENT', 'yTrue', 'yTrueLabel'])                              
    df = pd.DataFrame({'train': pd.Series([]), 'test': pd.Series(test_data['PATIENT']), 'val' : pd.Series([])})
    df.to_csv(os.path.join(args.split_dir, 'TestSplit.csv'), index = False)               
    train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = os.path.join(args.split_dir, 'TestSplit.csv'))                 
    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.num_classes}    
    if args.model_name != 'mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})            
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.num_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()         
    if args.model_name in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
            
        if args.inst_loss == 'svm':
            from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_name =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn = instance_loss_fn)
        elif args.model_name == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError    
    else: 
        if args.num_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
            
    model.relocate()
    model.load_state_dict(torch.load(args.modelAdr))
    model.to(device)       
    model.eval()
    
    test_loader = Get_split_loader(split_dataset = test_dataset, training = False)  
    
    all_probs = np.zeros((len(test_loader), args.num_classes))
    yTrue = np.zeros(len(test_loader))

    testSlides = test_loader.dataset.slide_data['FILENAME']
    testpatients = test_loader.dataset.slide_data['PATIENT']
    testPatientResults = {}

    for batch_idx, (data, label, coords) in tqdm(enumerate(test_loader)):
        data, label = data.to(device), label.to(device)        
        slide = testSlides.iloc[batch_idx]
        patient = testpatients.iloc[batch_idx]
        with torch.no_grad():
           _, probs, Y_hat, tileScores, _ = model(data)
           
        probs = probs.cpu().tolist()[0]
        all_probs[batch_idx] = probs
        yTrue[batch_idx] = label.item()
        tileScores = list(tileScores.cpu().tolist())
        coords = list(coords[0].cpu().tolist())        
        testPatientResults.update({slide: {'PATIENT': patient,'FILENAME': slide, 'probs': probs, 'label': label.item(),
                                           'tileScores' : tileScores, 'coords' : coords}})


    patients = []
    filaNames = []
    yTrue_test = []
    yTrueLabe_test = []
    probs = {}
    tileScores = {}
    coords = {}
        
    for i_temp in range(args.num_classes):
        key = utils.get_key_from_value(args.target_labelDict, i_temp)
        probs[key] = []
        
    for item in list(testPatientResults.keys()):
        temp = testPatientResults[item]
        patients.append(temp['PATIENT'])
        filaNames.append(temp['FILENAME'])
        yTrue_test.append(temp['label'])
        yTrueLabe_test.append(utils.get_key_from_value(args.target_labelDict, temp['label']))   
        tileScores[temp['PATIENT'] + '*' + temp['FILENAME']] = temp['tileScores']
        coords[temp['PATIENT'] + '*' + temp['FILENAME']] = temp['coords']
        
        for key in list(args.target_labelDict.keys()):
            probs[key].append(temp['probs'][utils.get_value_from_key(args.target_labelDict, key)])
    
    probs = pd.DataFrame.from_dict(probs)
    tileScores = pd.DataFrame.from_dict(tileScores, orient='index')
    coords = pd.DataFrame.from_dict(coords, orient='index')
                        
    df = pd.DataFrame(list(zip(patients, filaNames, yTrue_test, yTrueLabe_test)), columns =['PATIENT', 'FILENAME', 'yTrue', 'yTrueLabel'])
    df = pd.concat([df, probs], axis = 1)
    testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_SLIDE_BASED_FULL.csv')
    df.to_csv(testResultsPath, index = False)
    
    tileScores.to_csv(os.path.join(args.result_dir, 'TileScores.csv'), index = True)
    coords.to_csv(os.path.join(args.result_dir, 'Coordinates.csv'), index = True)
    
    totalPatientResultPath = CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, foldcounter = None ,
                                                     clamMil = True, reportFile = reportFile)
    reportFile.write('-' * 30 + '\n')  
              
    GenerateHighScoreTiles(imgsPath = args.datadir_test, totalPatientResultPath = totalPatientResultPath, totalResultPath = testResultsPath,
                           tileScorePath = os.path.join(args.result_dir, 'TileScores.csv'),
                           coordsPath = os.path.join(args.result_dir, 'Coordinates.csv'),
                           numHighScorePetients = args.numHighScorePatients, numHighScoreTiles = args.numHighScorePatients,
                           target_labelDict = args.target_labelDict, savePath = args.result_dir)       
    print('\n')
    print('-' * 30)
    reportFile.close()                    

        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
