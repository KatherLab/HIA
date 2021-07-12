"""
Created on Thu Mar 11 13:45:05 2021

@author: Narmin Ghaffari Laleh
"""

import utils.utils as utils
from utils.core_utils import Train_model_Classic, Validate_model_Classic
from utils.data_utils import ConcatCohorts_Classic, DatasetLoader_Classic, LoadTrainTestFromFolders, GetTiles
from eval.eval_Classic import PlotTrainingLossAcc, CalculatePatientWiseAUC, PlotBoxPlot,PlotROCCurve

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
from sklearn import metrics
from scipy.stats import ttest_ind

##############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"D:\ARCHITECTURE PROJECT\STAD\TCGA_VIT_512MAX_05FREEZE_TestFull.txt", help = 'Adress to the experiment File')
parser.add_argument('--modelAdr', type = str, default = r"D:\ARCHITECTURE PROJECT\STAD\BERN_VIT_512MAX_05FREEZE_TrainFull_EBV\RESULTS\MODEL_Full", help = 'Adress to the selected model')

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
    
    for targetLabel in targetLabels:
        
        args.target_label = targetLabel  
        args.projectFolder = utils.CreateProjectFolder(args.project_name, args.adressExp, targetLabel, args.model_name)
        
        if os.path.exists(args.projectFolder):
            continue
        else:
            os.mkdir(args.projectFolder) 
            
        reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
        
        reportFile.write('**********************************************************************'+ '\n')
        reportFile.write('DEPLOYING...')
        reportFile.write('\n' + '**********************************************************************'+ '\n')
        targetLabels = args.target_labels
        
            
        print('\n*** Load the DataSet ***\n')                      
        
        patientsList, labelsList, slidesList, clinicalTableList, slideTableList = ConcatCohorts_Classic(imagesPath = args.datadir_test, cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                        label = targetLabel, reportFile = reportFile)
                                                        
        labelsList = utils.CheckForTargetType(labelsList)
        
        le = preprocessing.LabelEncoder()
        labelsList = le.fit_transform(labelsList)
        
        args.num_classes = len(set(labelsList))
        args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
      
        utils.Summarize_Classic(args, list(labelsList), reportFile)
        
        args.result = os.path.join(args.projectFolder, 'RESULTS')
        os.makedirs(args.result, exist_ok = True)
        
        args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
        os.makedirs(args.split_dir, exist_ok = True)
        
        test_pid = []
        test_x = []
        test_y = []  
        
        patientID = np.array(patientsList)
        test_data = GetTiles(patients = patientID, labels = labelsList, imgsList = slidesList, label= targetLabel, 
                                  slideTableList = slideTableList, maxBlockNum = args.maxBlockNum, test = True)
        
            
        test_x = list(test_data['tileAd'])
        test_y = list(test_data[targetLabel])
        test_pid = list(test_data['patientID'])
                            
        df = pd.DataFrame(list(zip(test_x, test_y)), columns =['X', 'y'])
        df.to_csv(os.path.join(args.split_dir, 'FULL_TEST'+ '.csv'), index = False)
            
        print()  
            
                
        params = {'batch_size': args.batch_size,
                  'shuffle': False,
                  'num_workers': 0}
        
        model, input_size = utils.Initialize_model(args.model_name, args.num_classes, use_pretrained = True)
        test_set = DatasetLoader_Classic(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
        test_generator = torch.utils.data.DataLoader(test_set, **params)
        
        #model = torch.load(args.modelAdr)       
        #model = model.to(device)
        
        model.load_state_dict(torch.load(args.modelAdr))       
        model = model.to(device) 
        criterion = nn.CrossEntropyLoss()
        
        print('\n*** START DEPLOYING ***\n')
        
        epoch_loss, epoch_acc, predList  = Validate_model_Classic(model, test_generator, criterion)
        
        scores = {}
        for index, key in enumerate(list(args.target_labelDict.keys())):
            scores[key] = []
            for item in predList:
                scores[key].append(item[index])
        
        scores = pd.DataFrame.from_dict(scores)
            
        df = pd.DataFrame(list(zip(test_pid, test_x, test_y)), columns =['patientID', 'X', 'y'])
        df = pd.concat([df, scores], axis=1)
        
        df.to_csv(os.path.join(args.result, 'TEST_RESULT_FULL.csv'), index = False)
                                    
        CalculatePatientWiseAUC(os.path.join(args.result, 'TEST_RESULT_FULL.csv'),
                                list(set(test_pid)), args.target_labelDict, args.result, counter = 'FULL')  
        


                                







                