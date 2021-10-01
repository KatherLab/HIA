# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:37:23 2021

@author: Narmin Ghaffari Laleh
"""

import utils.utils as utils
import torch.nn as nn
import numpy as np
import argparse
import torch
import os
import random
from sklearn import preprocessing
from sklearn import metrics
from dataGenerator.dataset_generic import Generic_MIL_Dataset, Save_splits
from utils.data_utils import SortClini_SlideTables
from utils.core_utils import Accuracy_Logger
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
from utils.data_utils import Get_split_loader
from tqdm import tqdm
import pandas as pd
from eval.eval_Classic import CalculatePatientWiseAUC

##############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"C:\Users\Administrator\sciebo\deepHistology\labMembers\Narmin\Architecture Project\RCC\AACHEN_RCC_CLAM_TestFull.txt", help = 'Adress to the experiment File')
parser.add_argument('--modelAdr', type = str, default = r"C:\Users\Administrator\sciebo\deepHistology\labMembers\Narmin\Architecture Project\RCC\TCGA_RCC_CLAM_TrainFull_RCC_Subtype\RESULTS\s_0_checkpoint.pt", help = 'Adress to the selected model')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

##############################################################################

if __name__ == '__main__':
        
      
    args = utils.ReadExperimentFile(args, deploy = True)
    torch.cuda.set_device(args.gpuNo)
    random.seed(args.seed)

    targetLabels = args.target_labels
    args.feature_extract = False
    
    targetLabel = targetLabels[0]
    args.target_label = targetLabel  
    args.projectFolder = utils.CreateProjectFolder(args.project_name, args.adressExp, targetLabel, args.model_name)

    if os.path.exists(args.projectFolder):
        #raise NameError('THis PROJECT IS ALREADY EXISTS!')
        print('This Project Already Exists!')
    else:
        os.mkdir(args.projectFolder) 
            
    reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
    
    reportFile.write('**********************************************************************'+ '\n')
    reportFile.write('DEPLOYING...')
    reportFile.write('\n' + '**********************************************************************'+ '\n')
    targetLabels = args.target_labels
    
        
    print('\nLoad the DataSet...')   

    args.csvPath, labelList = SortClini_SlideTables(imagesPath = args.feat_dir, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
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
    
    dataset.Create_splits(k = 1, val_num = [0] * args.num_classes, test_num = [0] * args.num_classes, label_frac = lf)
    i = 0
    dataset.Set_splits()
    descriptor_df = dataset.Test_split_gen(return_descriptor = True, reportFile = reportFile, fold = i)
    
    splits = dataset.Return_splits(from_id = True)
    Save_splits(split_datasets = splits, column_keys = ['train'], filename = os.path.join(args.split_dir, 'splits_{}.csv'.format(i)))                                        
    descriptor_df.to_csv(os.path.join(args.split_dir, 'splits_{}_descriptor.csv'.format(i))) 
    
    train_dataset, val_dataset, test_dataset = dataset.Return_splits(from_id = False, csv_path = args.split_dir + '//splits_{}.csv'.format(i), trainFull = True)    
    
    datasets = (train_dataset, val_dataset, test_dataset)

    model_dict = {"dropout": args.drop_out, 'n_classes': args.num_classes}
    
    if args.model_size is not None and args.model_name != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
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
    
    else: # args.model_name == 'mil'
        if args.num_classes > 2:
            model = MIL_fc_mc(**model_dict)
            print('It is not there YET!')
        else:
            model = MIL_fc(**model_dict)
            
    model.relocate()
    model.load_state_dict(torch.load(args.modelAdr))
    model.to(device)   
    
    acc_logger = Accuracy_Logger(n_classes = args.num_classes)
    model.eval()
    test_error = 0.

    test_loader = Get_split_loader(train_dataset, testing = args.testing)
    all_probs = np.zeros((len(test_loader), args.num_classes))
    all_labels = np.zeros(len(test_loader))

    slide_ids = test_loader.dataset.slide_data['slide_id']
    case_ids = test_loader.dataset.slide_data['case_id']
    patient_results = {}

    for batch_idx, (data, label, coord) in tqdm(enumerate(test_loader)):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        case_id = case_ids.iloc[batch_idx]
        with torch.no_grad():
           logits, Y_prob, Y_hat, a_raw, results_dict = model(data)
           
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        tileScores = list(a_raw.cpu().detach().numpy())
        coords = list(coord[0].cpu().detach().numpy())
        
        patient_results.update({slide_id: {'case_id': case_id,'slide_id': slide_id, 'prob': probs, 'label': label.item(), 'tileScores' : tileScores, 'coords' : coords}})
        error = utils.calculate_error(Y_hat, label)
        test_error += error

    case_id_test = []
    slide_id_test = []
    labelList_test = []
    probs_test = {}
    tileScores = {}    
    coords = {}
    for i_temp in range(args.num_classes):
        key = utils.get_key_from_value(args.target_labelDict, i_temp)
        probs_test[key] = []
        
    for item in list(patient_results.keys()):
        temp = patient_results[item]
        case_id_test.append(temp['case_id'])
        slide_id_test.append(temp['slide_id'])
        labelList_test.append(temp['label'])
        tileScores[temp['case_id']] = temp['tileScores']
        coords[temp['case_id']] = temp['coords']
        
        for i_temp in range(args.num_classes):
            key = utils.get_key_from_value(args.target_labelDict, i_temp)
            probs_test[key].append(temp['prob'][0, i_temp])
    
    probs_test = pd.DataFrame.from_dict(probs_test)
    tileScores = pd.DataFrame.from_dict(tileScores, orient='index')
    coords = pd.DataFrame.from_dict(coords, orient='index')
   
    df = pd.DataFrame(list(zip(case_id_test, slide_id_test, labelList_test)), columns =['patientID', 'X', 'y'])
    df = pd.concat([df, probs_test], axis = 1)
    df.to_csv(os.path.join(args.result, 'TEST_RESULT_FULL_SCORES_temp.csv'), index = False)
    
    tileScores.to_csv(os.path.join(args.result, 'TileScores.csv'), index = True)
    coords.to_csv(os.path.join(args.result, 'Coords.csv'), index = True)
    
    CalculatePatientWiseAUC(resultCSVPath = os.path.join(args.result, 'TEST_RESULT_FULL_SCORES_temp.csv'), uniquePatients = list(set(case_id_test)),
                            target_labelDict = args.target_labelDict, resultFolder = args.result, counter = 'FULL', clamMil = True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
