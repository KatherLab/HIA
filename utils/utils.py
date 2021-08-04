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

def Collate_features(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return  [img, coords]
           
##############################################################################

def CreateProjectFolder(ExName, ExAdr, targetLabel, model_name):

    outputPath = ExAdr.split('\\')
    outputPath = outputPath[:-1]
    outputPath[0] = outputPath[0] + '\\'
    outputPath_root = os.path.join(*outputPath)
    outputPath = os.path.join(outputPath_root, ExName + '_' + targetLabel)
       
    return outputPath

##############################################################################

def Seed_torch(device, seed=7):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
          
##############################################################################
       
#def nth(iterator, n, default=None):
	#if n is None:
		#return collections.deque(iterator, maxlen=0)
	#else:
		#return next(islice(iterator,n, None), default)
        
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

def Initialize_model(model_name, num_classes, use_pretrained = True):

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained = use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
        #input_size = 512
        
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "vgg16":
        """ VGG11_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        
    elif model_name == "vit":
        
        model_ft = ViT('B_32_imagenet1k', pretrained = True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 384
    elif model_name == 'efficient':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs = model_ft._fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

###############################################################################

 
#path = r'D:\CRC\TCGA-CRC-DX\FEATURES'
#pathContent = os.listdir(path)

#notptFiles = [i for i in pathContent if not '.pt' in i]
#for i in notptFiles:
    #temp = i.split('.')[0]
    #os.rename(os.path.join(path, temp +'.pt'), os.path.join(path, i + '.pt'))

###############################################################################    

def Summarize_Classic(args, labels, reportFile):
    
    print("label column: {}".format(args.target_label))
    reportFile.write("label column: {}".format(args.target_label) + '\n')
    
    print("label dictionary: {}".format(args.target_labelDict))
    reportFile.write("label dictionary: {}".format(args.target_labelDict) + '\n')
    
    print("number of classes: {}".format(args.num_classes))
    reportFile.write("number of classes: {}".format(args.num_classes) + '\n')
    
    for i in range(args.num_classes):
        print('Patient-LVL; Number of samples registered in class %d: %d' % (i, labels.count(i)))
        reportFile.write('Patient-LVL; Number of samples registered in class %d: %d' % (i, labels.count(i)) + '\n')
    
        
    print('##############################################################\n')
    reportFile.write('**********************************************************************'+ '\n')


###############################################################################

def ReadExperimentFile(args, deploy = False):

    with open(args.adressExp) as json_file:        
        data = json.load(json_file)
        
    args.csv_name = 'CLEANED_DATA'
    args.project_name = args.adressExp.split('\\')[-1].replace('.txt', '')  
    
    try:
        datadir_train = data['dataDir_train']
    except:
        raise NameError('TRAINING DATA ADRESS IS NOT DEFINED!')
               
    args.clini_dir = []
    args.slide_dir = []
    args.datadir_train = []
    args.feat_dir = []
    
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
            warnings.warn('TESTING DATA ADRESS IS NOT DEFINED!')   
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
        args.maxBlockNum = data['maxNumBlocks']
    except:
        warnings.warn('MAX NUMBER OF BLOCKS IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 1000')   
        args.maxBlockNum = 1000
    
    try:
        args.max_epochs = data['epochs']
    except:
        warnings.warn('EPOCH NUMBER IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 5')   
        args.max_epochs = 5        

    try:
        args.max_epochs = int(data['epochs'])
    except:
        warnings.warn('EPOCH NUMBER IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 5')   
        args.max_epochs = 5  

    try:
        args.k = int(data['k'])   
    except:
        warnings.warn('NUMBER OF K FOLD CROSS ENTROPY IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 3')   
        args.k = 3 
        
    try:
        args.seed = int(data['seed']) 
    except:
        warnings.warn('SEED IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 1')   
        args.seed = 1    
        
    try:
        args.model_name = data['modelName']
    except:
        warnings.warn('MODEL NAME IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : resnet')   
        args.model_name = 'resnet'    

    try:
        args.opt = data['opt']
    except:
        warnings.warn('OPTIMIZER IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : adam')   
        args.opt = 'adam'
        
    try:
        args.lr = data['lr']
    except:
        warnings.warn('LEARNING RATE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 0.0001')   
        args.lr = 0.0001  
    
    try:
        args.reg = data['reg']
    except:
        warnings.warn('DECREASE RATE OF LR IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 0.00001')   
        args.lr = 0.00001     
        
    try:
        args.batch_size = data['batchSize']
          
    except:
        warnings.warn('BATCH SIZE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 64')   
        args.batch_size = 64
        
    if args.model_name == 'clam_sb' or args.model_name == 'clam_mb' or args.model_name == 'mil':
        args.useClassicModel = False
        args.batch_size = 1
    else:
        args.useClassicModel = True
    try:
        args.freeze_Ratio = data['freezeRatio']
    except:
        warnings.warn('FREEZE RATIO IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 0.5')   
        args.freeze_Ratio = 0.5

    try:
        args.train_full = MakeBool(data['trainFull'])
    except:
        warnings.warn('TRAIN FULL VALUE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : False')   
        args.train_full = False
    try:
         args.gpuNo = int(data['gpuNo'])  
    except:
        warnings.warn('GPU ID VALUE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 0')   
        args.gpuNo = 0  
    try:
        args.numHighScorePatients = int(data['numHighScorePatients'])
    except:
        warnings.warn('THE NUMBER OF PATIENTS FOR HIGH SCORE TILES IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 10')   
        args.numHighScorePatients = 10
        
    try:
        args.numHighScoreBlocks = int(data['numHighScoreBlocks'])
    except:
        warnings.warn('THE NUMBER OF HIGH SCORE TILES FOR PER PATIENT IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 10')   
        args.numHighScoreBlocks = 20  
        
    
    if args.model_name == 'clam_sb' or args.model_name == 'clam_mb' or args.model_name == 'mil':        
        try:
            args.bag_loss = data['bagLoss']
        except:
            warnings.warn('BAG LOSS IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : ce')   
            args.bag_loss = 'ce'
        try:
            args.inst_loss = data['instanceLoss']
        except:
            warnings.warn('INSTANCE LOSS IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : svm')   
            args.inst_loss = 'svm'         
        try:
            args.log_data = MakeBool(data['logData'])
        except:
            warnings.warn('LOG DATA VALUEIS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : True')   
            args.log_data = True            
        try:
            args.drop_out = MakeBool(data['dropOut'])
        except:
            warnings.warn('DROP OUT VALUE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : True')   
            args.drop_out = True            
        try:
            args.weighted_sample = MakeBool(data['weightedSample'])
        except:
            warnings.warn('WEIGHTED SAMPLE VALUE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : FALSE')   
            args.weighted_sample = False              
        try:
            args.early_stopping = MakeBool(data['earlyStop'])
        except:
            warnings.warn('EARLY STOPIING VALUE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : TRUE')   
            args.weighted_sample = True             
        try:
            args.model_size = data['modelSize']
        except:
            warnings.warn('MODEL SIZE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : small')   
            args.model_size = 'small'              
        try:
            args.B = int(data['B'])
        except:
            warnings.warn('VALUE OF SAMPLES IN A BAG IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 8')   
            args.B = data['B']            
        try:
            args.no_inst_cluster = MakeBool(data['noInstanceCluster'])
        except:
            warnings.warn('NO INSTANCE CLUSTER IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : False')   
            args.no_inst_cluster = False                
        try:
            args.bag_weight = float(data['bagWeight'])  
        except:
            warnings.warn('BAG WEIGHT IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 0.7')   
            args.bag_weight = 0.7      
        try:
            args.feature_extract = MakeBool(data['extractFeature'])
        except:
            warnings.warn('EXTRACT FEATURE VALUE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : False')   
            args.bag_weight = 0.7   
            
        try:
            args.normalize_targetNum = MakeBool(data['normalizeTargetPopulation'])
        except:
            warnings.warn('NORMALIZE TAGER NUMBERS VALUE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : False')   
            args.normalize_targetNum = False
        args.subtyping = False
        args.testing = False            
  

    return args

###############################################################################

def GetPatintsVsLabels(x, y, targetLabel):
    labels = []
    tiles = list(x)
    tiles_unique = [i.split('\\') [-1]  for i in tiles]
    #tiles_unique = list(set(tiles_unique))
    tiles_unique = [i.split('.')[0] for i in tiles_unique]
    tiles_unique = ['-'.join(i.split('-')[0:3]) for i in tiles_unique]
    
    patientID = list(set(tiles_unique)) 

    data = pd.DataFrame(list(zip(tiles_unique, x, y)), 
                          columns =['patientID', 'tileAd',targetLabel ])
    for i in patientID:
        for j in tiles:
            if i in j:
                index = tiles.index(j)
                continue
        labels.append(y[index])
        
    return data, patientID, labels


###############################################################################

def MakeBool(value):
    
    if value == 'True':
       return True
    else:
        return False
    
###############################################################################

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
