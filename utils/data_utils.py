# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 08:48:43 2021

@author: Narmin Ghaffari Laleh
"""
##############################################################################

import os 
import random
import numpy as np
import pandas as pd
import cv2
import torch
import math
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from torchvision import transforms
import utils.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################

def Generate_batch(image_file_list, status_list, maxBlockNum):
    
    image_file_list= list (image_file_list)
    status_list = list(status_list)
    bags = []
    for index, each_path in enumerate(image_file_list):
        img = []
        image_pathContent = os.listdir(each_path)
        if len(image_pathContent) > maxBlockNum:
                image_pathContent = np.random.choice(image_pathContent, maxBlockNum, replace=False)
        num_ins = len(image_pathContent)

        label = status_list[index]

        if label == 1:
            curr_label = np.ones(num_ins,dtype = np.float32)
        else:
            curr_label = np.zeros(num_ins, dtype = np.float32)
            
        for each_img in image_pathContent:
            img_data = cv2.imread(os.path.join(each_path, each_img))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            img_data = np.asarray(img_data, dtype=np.float32)
            img.append(np.expand_dims(img_data,0))
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, curr_label))

    return bags   
     
##############################################################################        
        
def Get_train_valid_Path(trainSet, train_percentage = 0.8):

    indexes = np.arange(len(trainSet))
    random.shuffle(indexes)

    num_train = int(train_percentage * len(trainSet))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    #Model_Train = [trainSet[i] for i in train_index]
    #Model_Val = [trainSet[j] for j in test_index]
    Model_Train = trainSet.iloc[train_index, :]
    Model_Val = trainSet.iloc[test_index, :]
    return Model_Train, Model_Val        
                        
##############################################################################

def Generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def collate_MIL_Training(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

##############################################################################

def collate_MIL_Testing(batch):
    
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    coords = torch.LongTensor([item[2] for item in batch])
    return [img, label, coords]

##############################################################################

def Get_split_loader(split_dataset, training = False, weighted = True):

    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    if training:
        if weighted:
            weights = Make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL_Training, **kwargs)	
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL_Training, **kwargs)
    else:
        loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL_Testing, **kwargs)	
    return loader

##############################################################################

def Make_weights_for_balanced_classes_split(dataset):
    
    N = float(len(dataset))    
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.Getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

##############################################################################

# def Get_simple_loader(dataset, batch_size=1):
    
# 	kwargs = {'num_workers': 0, 'pin_memory': False} if device.type == "cuda" else {}
# 	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
# 	return loader 

##############################################################################

class SubsetSequentialSampler(Sampler):

	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

##############################################################################

def ConcatCohorts_Classic(imagesPath, cliniTablePath, slideTablePath,  label, reportFile, outputPath, csvName, patientNumber = 'ALL', minNumberOfTiles = 0):   

    patients = []
    
    slideTableList = []
    clinicalTableList = []
    
    imgsList = []

    patientList = []
    slideList = []
    slideAdr = []
    labelList = []

    labelList_return = []
    patientList_return = []
    
    for imgCounter in range(len(imagesPath)):
                
        print('LOADING DATA FROM ' + imagesPath[imgCounter] + '...\n')
        reportFile.write('LOADING DATA FROM ' + imagesPath[imgCounter] + '...' + '\n')
        
        imgPath = imagesPath[imgCounter]
        cliniPath = cliniTablePath[imgCounter]
        slidePath = slideTablePath[imgCounter]
        
        if cliniPath.split('.')[-1] == 'csv':
            clinicalTable = pd.read_csv(cliniPath)
        else:
            clinicalTable = pd.read_excel(cliniPath)
        
        if slidePath.split('.')[-1] == 'csv':
            slideTable = pd.read_csv(slidePath)
        else:
            slideTable = pd.read_excel(slidePath)

        clinicalTable[label] = clinicalTable[label].replace(' ', '')
        lenBefore = len(clinicalTable)
        clinicalTable = clinicalTable[clinicalTable[label].notna()]
        
        notAcceptedValues = ['NA', 'NA ', 'NAN', 'N/A', 'na', 'n.a', 'N.A', 'UNKNOWN', 'x', 'NotAPPLICABLE', 'NOTPERFORMED',
                             'NotPerformed', 'Notassigned', 'excluded', 'exclide', '#NULL', 'PerformedButNotAvailable', 'x_', 'NotReported', 'notreported', 'INCONCLUSIVE', 'Unknown']
        
        for i in notAcceptedValues:
            clinicalTable = clinicalTable[clinicalTable[label] != i]

        lenafter = len(clinicalTable)
        
        print('Remove the NaN values from the Target Label...\n')
        print('{} Patients didnt have the proper label for target label: {}\n'.format(lenBefore - lenafter, label))
        reportFile.write('{} Patients didnt have the proper label for target label: {}'.format(lenBefore-lenafter, label) + '\n')
        
        clinicalTable_Patient = list(clinicalTable['PATIENT'])
        clinicalTable_Patient = list(set(clinicalTable_Patient))
        
        slideTable_Patint = list(slideTable['PATIENT'])
        slideTable_Patint = list(set(slideTable_Patint))
    
        # MAKE SURE SLIDE TABLE AND CLINICAL TABLE HAS SAME PATIENT IDs
        inClinicalNotInSlide = []
        for item in clinicalTable_Patient:
            if not item in slideTable_Patint:
                inClinicalNotInSlide.append(item)
                
        print('Data for {} Patients from Clini Table is not found in Slide Table!\n'.format(len(inClinicalNotInSlide)))
        reportFile.write('Data for {} Patients from Clini Table is not found in Slide Table!'.format(len(inClinicalNotInSlide)) + '\n')
        
        inSlideNotInClinical = []
        for item in slideTable_Patint:
            if not item in clinicalTable_Patient:
                inSlideNotInClinical.append(item)
                
        print('Data for {} Patients from Slide Table is not found in Clini Table!\n'.format(len(inSlideNotInClinical)))
        reportFile.write('Data for {} Patients from Slide Table is not found in Clini Table!'.format(len(inSlideNotInClinical)) + '\n')
                
        print('-' * 30)
        reportFile.write('-' * 30 + '\n')
        
        patienID_temp = []
        for item in clinicalTable_Patient:
            if item in slideTable_Patint:
                patienID_temp.append(item)    

        patientIDs = []
        for item in patienID_temp:
            if item in clinicalTable_Patient:
                patientIDs.append(item)
        
        patientIDs = list(set(patientIDs))
        intersect = utils.intersection(patients, patientIDs)
        
        if not len(intersect) == 0:
            print(imagesPath[imgCounter])
            print(intersect)
            raise NameError('There are same PATIENT ID between COHORTS')
            
        imageNames = os.listdir(imgPath)
        imageNames =[os.path.join(imgPath, i) for i in imageNames]
        patients = patients + patientIDs
        imgsList = imgsList + imageNames
        
        clinicalTable = clinicalTable.loc[clinicalTable['PATIENT'].isin(patientIDs)]
        slideTable = slideTable.loc[slideTable['PATIENT'].isin(patientIDs)]
        
        clinicalTableList.append(clinicalTable[['PATIENT', label]])
        slideTableList.append(slideTable)
    
    clinicalTableList = pd.concat(clinicalTableList)
    slideTableList = pd.concat(slideTableList)
    
    slideTable_PatintNotUnique = list(slideTableList['PATIENT'])
    
    if patientNumber == 'ALL':        
        for patientID in tqdm(patients):            
            indicies = [i for i, n in enumerate(slideTable_PatintNotUnique) if n == patientID]
            matchedSlides = [list(slideTableList['FILENAME'])[i] for i in indicies] 
    
            temp = clinicalTableList.loc[clinicalTableList['PATIENT'] == str(patientID)]
            temp.reset_index(drop = True, inplace=True)
            
            for slide in matchedSlides:
                slideName = [i for i in imageNames if slide in i]
                if not len(slideName) == 0:
                    slideName = slideName[0]                         
                    if not len(os.listdir(slideName)) <= minNumberOfTiles:
                        patientList.append(patientID)
                        slideList.append(slideName.split('\\')[-1])
                        slideAdr.append(slideName) 
                        labelList.append(temp[label][0])                    
                        if not patientID in patientList_return:
                            patientList_return.append(patientID)
                            labelList_return.append(temp[label][0])
                else:
                    reportFile.write('Slide {} is dropped out because of Pre-Processing.'.format(slide) + '\n')
                
    else:
        patientsCopy = patients.copy()
        while len(patientList_return) < int(patientNumber):
            samplePatient = random.sample(patientsCopy, 1)[0]
            patientsCopy.remove(samplePatient)      
            indicies = [i for i, n in enumerate(slideTable_PatintNotUnique) if n == samplePatient]
            matchedSlides = [list(slideTableList['FILENAME'])[i] for i in indicies] 
            
            temp = clinicalTableList.loc[clinicalTableList['PATIENT'] == str(samplePatient)]
            temp.reset_index(drop = True, inplace=True)
             
            for slide in matchedSlides:
                slideName = [i for i in imageNames if slide in i]
                if not len(slideName) == 0:
                    slideName = slideName[0]                         
                    if not len(os.listdir(slideName)) < minNumberOfTiles:
                        patientList.append(samplePatient)
                        slideList.append(slideName.split('\\')[-1])
                        slideAdr.append(slideName) 
                        labelList.append(temp[label][0])                    
                        if not samplePatient in patientList_return:
                            patientList_return.append(samplePatient)
                            labelList_return.append(temp[label][0])
                else:
                    reportFile.write('Slide {} is dropped out because of Pre-Processing.'.format(slide) + '\n')
                                                                       
    data = pd.DataFrame()
    data['PATIENT'] = patientList
    data['FILENAME'] = slideList
    data['SlideAdr'] = slideAdr    
    data[label] = labelList
             
    data.to_csv(os.path.join(outputPath, csvName + '.csv'),  index = False)
    return patientList_return, labelList_return, os.path.join(outputPath, csvName + '.csv')
 
###############################################################################

def GetTiles(csvFile, label, maxBlockNum, target_labelDict, test = False, seed = 23, filterPatients = []):

    np.random.seed(seed)
    data = pd.read_csv(csvFile)
    
    if not len(filterPatients) == 0:
        patientsUnique = filterPatients
    else:
        patientsUnique = list(set(data['PATIENT']))        
    
    tilesPathList = []
    yTrueList = []
    yTrueLabelList = []
    patinetList = []
    
    for index, patientID in enumerate(tqdm(patientsUnique)):
        selectedData = data.loc[data['PATIENT'] == patientID]
        selectedData.reset_index(inplace = True)
        tempTiles = []
        for item in range(len(selectedData)):
            tempTiles.extend([os.path.join(selectedData['SlideAdr'][item], i) for i in os.listdir(selectedData['SlideAdr'][item])])
        if len(tempTiles) > maxBlockNum:   
            random.shuffle(tempTiles)
            tempTiles = np.random.choice(tempTiles, maxBlockNum, replace=False)
        for tile in tempTiles:
            tilesPathList.append(tile)
            yTrueList.append(utils.get_value_from_key(target_labelDict, selectedData[label][0]))
            yTrueLabelList.append(selectedData[label][0])
            patinetList.append(str(patientID))
                
    df = pd.DataFrame(list(zip(patinetList, tilesPathList, yTrueList, yTrueLabelList)), columns =['PATIENT', 'TilePath', 'yTrue', 'yTrueLabel'])     
    df_temp = df.dropna()
    
    if test:
        dfFromDict = df_temp
    else:            
        tags = list(df_temp['yTrue'].unique())
        tagsLength = []
        dfs = {}
        for tag in tags:
            temp = df_temp.loc[df_temp['yTrue'] == tag]
            temp = temp.sample(frac=1).reset_index(drop=True)
            dfs[tag] = temp 
            tagsLength.append(len(df_temp.loc[df_temp['yTrue'] == tag]))
        
        minSize = np.min(tagsLength)
        keys = list(dfs.keys())
        frames = []
        for key in keys:
            temp_len = len(dfs[key])
            diff_len = temp_len - minSize
            drop_indices = np.random.choice(dfs[key].index, diff_len, replace = False)
            frames.append(dfs[key].drop(drop_indices))
            
        dfFromDict = pd.concat(frames)
                    
    return dfFromDict

###############################################################################   

class DatasetLoader_Classic(torch.utils.data.Dataset):

    def __init__(self, imgs, labels, transform = None, target_patch_size = -1):
        self.labels = labels
        self.imgs = imgs
        self.target_patch_size = target_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        X = Image.open(self.imgs[index])
        y = self.labels[index]
        if self.target_patch_size is  not None:
            X = X.resize((self.target_patch_size, self.target_patch_size))
            X = np.array(X)
        if self.transform is not None:
            X = self.transform(X)
        return X, y

###############################################################################

def LoadTrainTestFromFolders(trainPath, testPath):
    
    pathContent = os.listdir(testPath)
    pathContent = [os.path.join(testPath , i) for i in pathContent]
    
    test_x = []
    test_y = []
    
    for path in pathContent:
        if path.split('\\')[-1] == 'MSIH':
            y = 1
        else:
            y = 0
        tiles = os.listdir(path)
        tiles = [os.path.join(path , i) for i in tiles]
        test_x = test_x + tiles
        test_y = test_y + [y]* len(tiles)
    
    pathContent = os.listdir(trainPath)
    pathContent = [os.path.join(trainPath , i) for i in pathContent]
    
    train_x = []
    train_y = []
    
    for path in pathContent:
        if path.split('\\')[-1] == 'MSIH':
            y = 1
        else:
            y = 0
        tiles = os.listdir(path)
        tiles = [os.path.join(path , i) for i in tiles]
        train_x = train_x + tiles
        train_y = train_y + [y]* len(tiles)
        
    return train_x, train_y, test_x, test_y

###############################################################################
    
