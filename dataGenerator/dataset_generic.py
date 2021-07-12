# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:47:16 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from torch.utils.data import Dataset
import h5py
import utils.utils as utils
from utils.data_utils import Generate_split

##############################################################################

def Save_splits(split_datasets, column_keys, filename, boolean_style=False):

    splits = [split_datasets[i].slide_data['slide_id'] if split_datasets[i] else [] for i in range(len(split_datasets))]
    splits_1 = []
    for item in splits:
        if not len(item) == 0:
            splits_1.append(item)
    splits = splits_1
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])
    
    df.to_csv(filename)
    
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
                    normalize_targetNum = True, 
                    normalize_method = 'UpSample',
                    reportFile = None):

        self.label_dict = label_dict
        self.num_classes  = len(self.label_dict)
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if not label_col:
            label_col = 'label'
 
        self.label_col = label_col
        slide_data = pd.read_csv(csv_path)
        slide_data = self.Df_prep(slide_data, self.label_dict, ignore, self.label_col)
        np.random.seed(seed)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        
        patients = np.unique(np.array(slide_data['case_id']))
        patient_labels  = []
        
        for p in patients:
            locations = slide_data[slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max() # get patient label (MIL convention)
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                pass
            patient_labels.append(label)
        df_temp = pd.DataFrame(list(zip(patients, patient_labels)), columns =['case_id', 'label'])
                  
        if normalize_targetNum:
            if normalize_method == 'DownSample':
                df_temp = pd.DataFrame(list(zip(patients, patient_labels)), columns =['case_id', 'label'])
                tags = list(df_temp['label'].unique())
                tagsLength = []
                dfs = {}
                for tag in tags:
                    temp = df_temp.loc[df_temp['label'] == tag]
                    temp = temp.sample(frac = 1).reset_index(drop=True)
                    dfs[tag] = temp 
                    tagsLength.append(len(df_temp.loc[df_temp['label'] == tag]))
                                     
                minSize = min(tagsLength)
                keys = list(dfs.keys())
                
                frames = []
                for key in keys:
                    temp_len = len(dfs[key])
                    diff_len = temp_len - minSize
                    drop_indices = np.random.choice(dfs[key].index, diff_len, replace=False)
                    frames.append(dfs[key].drop(drop_indices))
                    
                dfFromDict = pd.concat(frames)
                dfFromDict = dfFromDict.sample(frac=1)
            else:
                df_temp = pd.DataFrame(list(zip(patients, patient_labels)), columns =['case_id', 'label'])
                tags = list(df_temp['label'].unique())
                tagsLength = []
                dfs = {}
                for tag in tags:
                    temp = df_temp.loc[df_temp['label'] == tag]
                    temp = temp.sample(frac = 1).reset_index(drop = True)
                    dfs[tag] = temp 
                    tagsLength.append(len(df_temp.loc[df_temp['label'] == tag]))
                                     
                maxSize = max(tagsLength)
                keys = list(dfs.keys())
                
                frames = []
                for key in keys:
                    temp_len = len(dfs[key])
                    diff_len = maxSize - temp_len
                    if not diff_len == 0:
                        add_indices = np.random.choice(dfs[key].index, diff_len)
                        for item in add_indices:
                            dfs[key] = dfs[key].append(dfs[key].iloc[item])
                        dfs[key].reset_index(inplace = True)
                        frames.append(dfs[key])
                    else:
                        frames.append(dfs[key])
                            
                dfFromDict = pd.concat(frames)
                dfFromDict = dfFromDict.sample(frac=1)
                
        else:
            dfFromDict = df_temp
            
        temp = list(dfFromDict['case_id'])
        slide_data = slide_data[slide_data['case_id'].isin(temp)]
        slide_data = slide_data.reset_index(drop=True)
        self.slide_data = slide_data
        self.patient_data = {'case_id' : np.array(dfFromDict['case_id']), 'label':np.array(dfFromDict['label'])}
        self.Cls_ids_prep()
        if print_info:
            self.Summarize(reportFile)
                        
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

    def Summarize(self, reportFile):
        print("label column: {}".format(self.label_col))
        reportFile.write("label column: {}".format(self.label_col) + '\n')
        print("label dictionary: {}".format(self.label_dict))
        reportFile.write("label dictionary: {}".format(self.label_dict) + '\n')
        print("number of classes: {}".format(self.num_classes))
        reportFile.write("number of classes: {}".format(self.num_classes) + '\n')
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            reportFile.write('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]) + '\n')
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))
            reportFile.write('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]) + '\n')
        print('##############################################################\n')
        reportFile.write('**********************************************************************'+ '\n')
        
##############################################################################
            
    def Create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = { 'n_splits' : k, 'val_num' : val_num, 'test_num': test_num,'label_frac': label_frac,'seed': self.seed,
                    'custom_test_ids': custom_test_ids}
        
        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})        
        else:
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})
        self.split_gen = Generate_split(**settings)
        
##############################################################################

    def Set_splits(self, start_from = None):
        
        if start_from:
            ids = utils.nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)
        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))] 
            for split in range(len(ids)):                 
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)    
                    
            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            self.train_ids, self.val_ids, self.test_ids = ids
                    
                   
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

    def Test_split_gen(self, return_descriptor = False, reportFile = None, fold = 0):
        
        if return_descriptor:          
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index, columns= columns)  
            
        count = len(self.train_ids)
        reportFile.write('Overvie of Data split for fold {} : '.format(fold) + '\n')
        print('\nnumber of training samples: {}'.format(count))
        reportFile.write('\nnumber of training samples: {}'.format(count) + '\n')
        
        labels = self.Getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)        
        for u in range(len(unique)):            
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            reportFile.write('number of samples in cls {}: {}'.format(unique[u], counts[u])+ '\n')
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]                    
                    
        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        reportFile.write('\nnumber of val samples: {}'.format(count) + '\n')
        labels = self.Getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        
        for u in range(len(unique)):     
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            reportFile.write('number of samples in cls {}: {}'.format(unique[u], counts[u])+ '\n')
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]            
            
        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        reportFile.write('\nnumber of test samples: {}'.format(count) + '\n')
        labels = self.Getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            reportFile.write('number of samples in cls {}: {}'.format(unique[u], counts[u])+ '\n')
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0            
        reportFile.write('**********************************************************************'+ '\n')
        if return_descriptor:
            return df            
            
##############################################################################

    def Return_splits(self, from_id=True, csv_path = None, trainFull = False):
        
        print(len(self.train_ids))
        print(len(self.val_ids))
        
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
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
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
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        if not self.use_h5:
            if self.data_dir:
                full_path = os.path.join(self.data_dir,'{}.pt'.format(slide_id))
                features = torch.load(full_path)
                return features, label
            else:
                return slide_id, label
        else:
            full_path = os.path.join(self.data_dir,'{}.h5'.format(slide_id))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            features = torch.from_numpy(features)
            return features, label, coords

        
##############################################################################

class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2):
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














