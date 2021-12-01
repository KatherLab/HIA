# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:23:50 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

import os 
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torchvision import models
import json
from pytorch_pretrained_vit import ViT
from ipywidgets import IntProgress
import warnings
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           
##############################################################################

def CreateProjectFolder(ExName, ExAdr, targetLabel, model_name, repeat = None):

    outputPath = ExAdr.split('\\')
    outputPath = outputPath[:-1]
    outputPath[0] = outputPath[0] + '\\'
    outputPath_root = os.path.join(*outputPath)
    if repeat:
        outputPath = os.path.join(outputPath_root, ExName + '_' + targetLabel + '_' + str(repeat))
    else:
        outputPath = os.path.join(outputPath_root, ExName + '_' + targetLabel)
    return outputPath
   
        
##############################################################################
       
def Print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)

##############################################################################
        
def get_optim(model, args, params = False):
   
    if params:
        temp = model
    else:
        temp = filter(lambda p: p.requires_grad, model.parameters())
        
    if args.opt == "adam":
        optimizer = optim.Adam(temp, lr = args.lr, weight_decay = args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(temp, lr = args.lr, momentum = 0.9, weight_decay = args.reg)
    else:
        raise NotImplementedError
        
    return optimizer

##############################################################################
    
def Initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0) 

##############################################################################
            
def Collate_features(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return  [img, coords]

##############################################################################
            
def calculate_error(Y_hat, Y):
    
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

##############################################################################

def save_pkl(filename, save_object):
    
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

##############################################################################

def load_pkl(filename):
    
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file

##############################################################################

def RenameTCGASLideNamesInSlideTable(slideTablePath, imgsFolder):
    
    imgs = os.listdir(imgsFolder)
    
    if slideTablePath.split('.')[-1] == 'csv':
        slideTable = pd.read_csv(slideTablePath, sep=r'\s*,\s*', header=0, engine='python')
    else:
        slideTable = pd.read_excel(slideTablePath)
        
    slides = list(slideTable['FILENAME'])
    for item in imgs:
        temp = item.split('.')[0]
        index = slides.index(temp)
        slides[index] = item
        
    slideTable['FILENAME'] = slides
    slideTable.to_csv(slideTablePath.replace('.csv', '_NEW.csv'), index=False)
        

###############################################################################

def Initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):

    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained = use_pretrained)
        Set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained = use_pretrained)
        Set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
                
    elif model_name == "vit":        
        model_ft = ViT('B_32_imagenet1k', pretrained = True)
        Set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 384
    elif model_name == 'efficient':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        Set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
    return model_ft, input_size

###############################################################################

def Set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

###############################################################################
            
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

###############################################################################    

def Summarize(args, labels, reportFile):
    
    print("label column: {}\n".format(args.target_label))
    reportFile.write("label column: {}".format(args.target_label) + '\n')    
    print("label dictionary: {}\n".format(args.target_labelDict))
    reportFile.write("label dictionary: {}".format(args.target_labelDict) + '\n')    
    print("number of classes: {}\n".format(args.num_classes))
    reportFile.write("number of classes: {}".format(args.num_classes) + '\n')    
    for i in range(args.num_classes):
        print('Patient-LVL; Number of samples registered in class %d: %d\n' % (i, labels.count(i)))
        reportFile.write('Patient-LVL; Number of samples registered in class %d: %d' % (i, labels.count(i)) + '\n')           
    print('-' * 30 + '\n')
    reportFile.write('-' * 30 + '\n')

###############################################################################

def ReadExperimentFile(args, deploy = False):

    with open(args.adressExp) as json_file:        
        data = json.load(json_file)
        
    args.csv_name = 'CLEANED_DATA'
    args.project_name = args.adressExp.split('\\')[-1].replace('.txt', '')  
    
    try:
        if not deploy :
            datadir_train = data['dataDir_train']
    except:
        raise NameError('TRAINING DATA ADRESS IS NOT DEFINED!')
               
    args.clini_dir = []
    args.slide_dir = []
    args.datadir_train = []
    args.feat_dir = []
    
    if not deploy :
        for index, item in enumerate(datadir_train):
            if os.path.exists(os.path.join(item , 'BLOCKS_NORM_MACENKO')):
                args.datadir_train.append(os.path.join(item , 'BLOCKS_NORM_MACENKO'))
                
            elif os.path.exists(os.path.join(item , 'BLOCKS_NORM_VAHADANE')):
                args.datadir_train.append(os.path.join(item , 'BLOCKS_NORM_VAHADANE'))
                
            elif os.path.exists(os.path.join(item , 'BLOCKS_NORM_REINHARD')):
                args.datadir_train.append(os.path.join(item , 'BLOCKS_NORM_REINHARD'))
                
            elif os.path.exists(os.path.join(item , 'BLOCKS')):
                args.datadir_train.append(os.path.join(item , 'BLOCKS'))
            else:
                raise NameError('NO BLOCK FOLDER FOR ' + item + ' TRAINNG IS FOUND!')
            
            if not deploy:
                if os.path.isfile(os.path.join(item, item.split('\\')[-1] + '_CLINI.xlsx')):
                     args.clini_dir.append(os.path.join(item, item.split('\\')[-1] + '_CLINI.xlsx'))
                else:
                     raise NameError('NO CLINI DATA FOR ' + item + ' IS FOUND!')
    
                if os.path.isfile(os.path.join(item, item.split('\\')[-1] + '_SLIDE.csv')):
                     args.slide_dir.append(os.path.join(item, item.split('\\')[-1] + '_SLIDE.csv'))
                else:
                     raise NameError('NO SLIDE DATA FOR ' + item + ' IS FOUND!')           
                                
                args.feat_dir.append(os.path.join(item , 'FEATURES'))
                    

    try:
        datadir_test = data['dataDir_test']
    except:
        if not deploy:
            print('TESTING DATA ADRESS IS NOT DEFINED!\n')   
        else:
            raise NameError('TESTING DATA ADRESS IS NOT DEFINED!')   
    if deploy:
        args.datadir_test = []
        
        for index, item in enumerate(datadir_test):
            if os.path.exists(os.path.join(item , 'BLOCKS_NORM_MACENKO')):
                args.datadir_test.append(os.path.join(item , 'BLOCKS_NORM_MACENKO'))
            elif os.path.exists(os.path.join(item , 'BLOCKS_NORM_VAHADANE')):
                args.datadir_test.append(os.path.join(item , 'BLOCKS_NORM_VAHADANE'))
            elif os.path.exists(os.path.join(item , 'BLOCKS_NORM_REINHARD')):
                args.datadir_test.append(os.path.join(item , 'BLOCKS_NORM_REINHARD'))
            elif os.path.exists(os.path.join(item , 'BLOCKS')):
                args.datadir_test.append(os.path.join(item , 'BLOCKS'))
            else:
                 raise NameError('NO BLOCK FOLDER FOR TESTING IS FOUND!')
            
            if os.path.isfile(os.path.join(item, item.split('\\')[-1] + '_CLINI.xlsx')):
                 args.clini_dir.append(os.path.join(item, item.split('\\')[-1] + '_CLINI.xlsx'))
            else:
                 raise NameError('NO CLINI DATA FOR ' + item + ' IS FOUND!')

            if os.path.isfile(os.path.join(item, item.split('\\')[-1] + '_SLIDE.csv')):
                 args.slide_dir.append(os.path.join(item, item.split('\\')[-1] + '_SLIDE.csv'))
            else:
                 raise NameError('NO SLIDE DATA FOR ' + item + ' IS FOUND!')           
                            
            args.feat_dir.append(os.path.join(item , 'FEATURES'))
              
    try:
        args.target_labels = data['targetLabels']
    except:
        raise NameError('TARGET LABELS ARE NOT DEFINED!')
    
    try:
        args.max_epochs = data['epochs']
    except:
        print('EPOCH NUMBER IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 5\n') 
        print('-' * 30)
        args.max_epochs = 8        

    try:
        args.numPatientToUse = data['numPatientToUse']
    except:
        print('NUMBER OF PATIENTS TO USE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : ALL\n') 
        print('-' * 30)
        args.numPatientToUse = 'ALL'     

    try:
        args.k = int(data['k'])   
    except:
        print('NUMBER OF K FOLD CROSS ENTROPY IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 3\n')   
        print('-' * 30)
        args.k = 3 
        
    try:
        args.seed = int(data['seed']) 
    except:
        print('SEED IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 1\n')   
        print('-' * 30)
        args.seed = 1    
        
    try:
        args.model_name = data['modelName']
    except:
        print('MODEL NAME IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : resnet18\n')  
        print('-' * 30)
        args.model_name = 'resnet18'    

    try:
        args.opt = data['opt']
    except:
        print('OPTIMIZER IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : adam\n') 
        print('-' * 30)
        args.opt = 'adam'
        
    try:
        args.lr = data['lr']
    except:
        print('LEARNING RATE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 0.0001\n')
        print('-' * 30)
        args.lr = 0.0001  
   
    try:
        args.reg = data['reg']
    except:
        print('DECREASE RATE OF LR IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 0.00001\n')   
        print('-' * 30)
        args.reg = 0.00001             
    try:
        args.batch_size = data['batchSize']
          
    except:
        print('BATCH SIZE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 64\n')
        print('-' * 30)
        args.batch_size = 64
        
    if deploy:        
        try:
            args.numHighScorePatients = data['numHighScorePatients']
              
        except:
            print('NUMBER OF HIGH SCORE PATIENTS IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 5\n')  
            print('-' * 30)
            args.numHighScorePatients = 5
    
        try:
            args.numHighScoreTiles = data['numHighScoreTiles']
              
        except:
            print('NUMBER OF HIGH SCORE TILES IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 5\n')
            print('-' * 30)
            args.numHighScoreTiles = 5   
            
    if args.model_name == 'clam_sb' or args.model_name == 'clam_mb' or args.model_name == 'mil' or args.model_name == 'attmil':
        args.useClassicModel = False
    else:
        args.useClassicModel = True

    if args.model_name == 'clam_sb' or args.model_name == 'clam_mb' or args.model_name == 'mil':
        args.batch_size = 1

    try:
        args.train_full = MakeBool(data['trainFull'])
    except:
        print('TRAIN FULL VALUE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : False\n')  
        print('-' * 30)
        args.train_full = False 
    try:
         args.repeatExperiment = int(data['repeatExperiment'])  
    except:
        print('REPEAT EXPERIEMNT NNUmBER IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : 1\n')  
        print('-' * 30)
        args.repeatExperiment = 1 
        
    try:
         args.minNumBlocks = int(data['minNumBlocks'])  
    except:
        print('MIN NUMBER OF BLOCKS IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : 0\n') 
        print('-' * 30)
        args.minNumBlocks = 0
        
    try:
        args.early_stopping = MakeBool(data['earlyStop'])
    except:
        print('EARLY STOPIING VALUE IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : TRUE\n')  
        print('-' * 30)
        args.early_stopping = True  
        
    if  args.early_stopping:        
        try:
            args.minEpochToTrain = data['minEpochToTrain']
        except:
            print('MIN NUMBER OF EPOCHS TO TRAIN IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : 10\n')   
            print('-' * 30)
            args.minEpochToTrain = 10 
         
        try:
            args.patience = data['patience']
        except:
            print('PATIENCE VALUE FOR EARLY STOPPING IS NOT DEFINED!\n DEFAULT VALUE WILL BE USED : 10\n') 
            print('-' * 30)
            args.patience = 10 
            
    if not args.model_name in ['clam_sb', 'clam_mb', 'mil', 'attmil']:
        try:
            args.freeze_Ratio = data['freezeRatio']
        except:
            print('FREEZE RATIO IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : 0.5')   
            print('-' * 30)
            args.freeze_Ratio = 0.5
        try:
            args.maxBlockNum = data['maxNumBlocks']
        except:
            print('MAX NUMBER OF BLOCKS IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 200\n') 
            print('-' * 30)
            args.maxBlockNum = 200
        try:
             args.gpuNo = int(data['gpuNo'])  
        except:
            print('GPU ID VALUE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 0\n')  
            print('-' * 30)
            args.gpuNo = 0  
        
    if args.model_name in ['clam_sb', 'clam_mb', 'mil']:
        
        try:
            args.bag_loss = data['bagLoss']
        except:
            print('BAG LOSS IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : ce\n')  
            print('-' * 30)
            args.bag_loss = 'ce'   
        try:
            args.drop_out = MakeBool(data['dropOut'])
        except:
            print('DROP OUT VALUE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : True\n') 
            print('-' * 30)
            args.drop_out = True 
        try:
            args.weighted_sample = MakeBool(data['weightedSample'])
        except:
            print('WEIGHTED SAMPLE VALUE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : FALSE\n')   
            print('-' * 30)
            args.weighted_sample = False 
            
    if args.model_name in ['clam_sb', 'clam_mb', 'mil', 'attmil']:            
        try:
            args.extractFeature = MakeBool(data['extractFeature'])
        except:
            print('EXTRACT FEATURE VALUE IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : FALSE\n')  
            print('-' * 30)
            args.extractFeature = False  
            
    if args.model_name in ['clam_sb', 'clam_mb']:              
        try:
            args.inst_loss = data['instanceLoss']
        except:
            print('INSTANCE LOSS IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : svm\n')  
            print('-' * 30)
            args.inst_loss = 'svm'                         
        try:
            args.model_size = data['modelSize']
        except:
            print('MODEL SIZE IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : small\n') 
            print('-' * 30)
            args.model_size = 'small' 
             
        try:
            args.B = int(data['B'])
        except:
            print('VALUE OF SAMPLES IN A BAG IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : 8\n') 
            print('-' * 30)
            args.B = 8 
            
        try:
            args.no_inst_cluster = MakeBool(data['noInstanceCluster'])
        except:
            print('NO INSTANCE CLUSTER IS NOT DEFINED! \nDEFAULT VALUE WILL BE USED : FALSE\n')  
            print('-' * 30)
            args.no_inst_cluster = False  
              
        try:
            args.bag_weight = float(data['bagWeight'])  
        except:
            print('BAG WEIGHT IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : 0.7\n')  
            print('-' * 30)
            args.bag_weight = 0.7      
            
        try:
            args.subtyping = MakeBool(data['subtyping'])
        except:
            print('EXTRACT Subtyping VALUE IS NOT DEFINED!\nDEFAULT VALUE WILL BE USED : FALSE\n')  
            print('-' * 30)
            args.subtyping = False
    return args


###############################################################################

def MakeBool(value):
    
    if value == 'True':
       return True
    else:
        return False
    
###############################################################################

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

###############################################################################

def isint(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

###############################################################################

def CheckForTargetType(labelsList):
    
    if len(set(labelsList)) >= 5:     
        labelList_temp = [str(i) for i in labelsList]
        checkList1 = [s for s in labelList_temp if isfloat(s)]
        checkList2 = [s for s in labelList_temp if isint(s)]
        if not len(checkList1) == 0 or not len (checkList2):
            med = np.median(labelsList)
            labelsList = [1 if i>med else 0 for i in labelsList]
        else:
            raise NameError('IT IS NOT POSSIBLE TO BINARIZE THE NOT NUMERIC TARGET LIST!')
    return labelsList
                    
###############################################################################            
    
def get_key_from_value(d, val):
    
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None   
    
 ###############################################################################   
    
def get_value_from_key(d, key):
    
    values = [v for k, v in d.items() if k == key]
    if values:
        return values[0]
    return None    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
