# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:47:16 2021

@author: Narmin Ghaffari Laleh

reference : https://github.com/mahmoodlab/CLAM

"""

##############################################################################

from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py
import utils.utils as utils
from utils.data_utils import Generate_split
    
##############################################################################

class Generic_WSI_Classification_Dataset(Dataset):

    def __init__(self, csv_path = '',
                    shuffle = False, 
                    seed = 7, 
                    print_info = True,
                    label_dict = {},
                    ignore = [],
                    patient_strat = False,
                    label_col = None,
                    patient_voting = 'max',
                    reportFile = None):

        self.label_dict = label_dict
        self.num_classes  = len(self.label_dict)
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None 
        self.label_col = label_col
        
        slide_data = pd.read_csv(csv_path)
        slide_data = self.Df_prep(data = slide_data, label_dict = self.label_dict, ignore = ignore, label_col = self.label_col)
        np.random.seed(seed)        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)  
            
        patients = np.unique(np.array(slide_data['PATIENT']))
        patient_labels  = []
        
        for p in patients:
            locations = slide_data[slide_data['PATIENT'] == p].index.tolist()
            assert len(locations) > 0
            label = slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max()
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            patient_labels.append(label)
        df_temp = pd.DataFrame(list(zip(patients, patient_labels)), columns =['PATIENT', 'label'])                             
        temp = list(df_temp['PATIENT'])
        slide_data = slide_data[slide_data['PATIENT'].isin(temp)]
        slide_data = slide_data.reset_index(drop = True)
        self.slide_data = slide_data
        self.patient_data = {'PATIENT' : np.array(df_temp['PATIENT']), 'label':np.array(df_temp['label'])}
        self.Cls_ids_prep()

                        
##############################################################################

    def Df_prep(self, data, label_dict, ignore, label_col):
        
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]
        return data

##############################################################################

    def Cls_ids_prep(self):
        
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(np.array(self.patient_data['label']) == i)[0]  
            
        self.slide_cls_ids  = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(np.array(self.slide_data['label']) == i)[0]  

##############################################################################   
 
    def Get_list(self, ids):
        return self.slide_data['slide_id'][ids]
    
##############################################################################

    def Getlabel(self, ids):
        return self.slide_data['label'][ids]
    
##############################################################################

    def __Getitem__(self, idx):
        return None
                         
##############################################################################

    def Return_splits(self, from_id=True, csv_path = None, trainFull = False):
                
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop = True)
                train_split = Generic_Split(train_data, data_dir = self.data_dir, num_classes = self.num_classes)                
            else:
                train_split = None        
        
            if len(self.val_ids) > 0:
                val_data  = self.slide_data.loc[self.val_ids].reset_index(drop = True)
                val_split = Generic_Split(val_data, data_dir = self.data_dir, num_classes = self.num_classes)                
            else:
                val_split = None         
        
            if len(self.test_ids) > 0:
                test_data  = self.slide_data.loc[self.test_ids].reset_index(drop = True)
                test_split  = Generic_Split(test_data , data_dir = self.data_dir, num_classes = self.num_classes)                
            else:
                test_split  = None         
        
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            if trainFull:
                train_split = self.Get_split_from_df(all_splits, 'train')
                val_split = []
                test_split = []
            else:
                train_split = self.Get_split_from_df(all_splits, 'train')
                val_split = self.Get_split_from_df(all_splits, 'val')
                test_split = self.Get_split_from_df(all_splits, 'test')        			
        return train_split, val_split, test_split        
 
##############################################################################
               
    def Get_split_from_df(self, all_splits, split_key='train'):
        
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        
        if len(split) > 0:
            mask = self.slide_data['PATIENT'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop = True)
            split = Generic_Split(df_slice, data_dir = self.data_dir, num_classes = self.num_classes)
        else:
            split = None
        		
        return split            
           
##############################################################################    

class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    
    def __init__(self, data_dir, **kwargs):        
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5  = False

    def load_from_h5(self, toggle):        
        self.use_h5 = toggle
        
    def __getitem__(self, idx):        
        slide_id = self.slide_data['FILENAME'][idx]
        label = self.slide_data['label'][idx]
        self.use_h5  = True
        if not self.use_h5:
            if self.data_dir:
                full_path = os.path.join(self.data_dir,'{}.pt'.format(slide_id))
                features = torch.load(full_path)
                return features, label
            else:
                return slide_id, label
        else:
            full_path = os.path.join(self.data_dir,'{}'.format(slide_id))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            features = torch.from_numpy(features)
            return features, label, coords

        
##############################################################################

class Generic_Split(Generic_MIL_Dataset):
    
	def __init__(self, slide_data, data_dir = None, num_classes = 2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		
##############################################################################














