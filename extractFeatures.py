# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:28:05 2021

@author: Narmin Ghaffari Laleh
"""

from utils.utils import Collate_features
from dataGenerator.dataSet import Whole_Slide_Bag
from models.resnet_custom import Resnet50_baseline

from torch.utils.data import DataLoader
import h5py
import glob
import torch
import torch.nn as nn
import os
import time
import torchvision

##############################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

##############################################################################

def Save_hdf5(output_dir, asset_dict, mode='a'):
	file = h5py.File(output_dir, mode)

	for key, val in asset_dict.items():
		data_shape = val.shape
		if key not in file:
			data_type = val.dtype
			chunk_shape = (1, ) + data_shape[1:]
			maxshape = (None, ) + data_shape[1:]
			dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
			dset[:] = val
		else:
			dset = file[key]
			dset.resize(len(dset) + data_shape[0], axis=0)
			dset[-data_shape[0]:] = val  

	file.close()
	return output_dir

##############################################################################

def Compute_w_loader(file_path, output_path, model, batch_size = 8, verbose = 0, print_every = 20, pretrained = True, target_patch_size = -1):

    dataset = Whole_Slide_Bag(file_path = file_path, pretrained = pretrained, target_patch_size = target_patch_size)    
    #x, y = dataset[0]
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset = dataset, batch_size = batch_size, **kwargs, collate_fn = Collate_features)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))
        
    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        #batch = batch[0]
        with torch.no_grad():	
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)
            features = model(batch)
            features = features.cpu().detach().numpy()
            asset_dict = {'features': features, 'coords': coords}
            Save_hdf5(output_path, asset_dict, mode=mode)
            mode = 'a'
    return output_path
                        

##############################################################################
    
def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print("No weight could be loaded..")
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

##############################################################################
    
def ExtractFeatures(data_dir, feat_dir, batch_size, target_patch_size = -1, filterData = True, self_supervised = False):
    
    print('initializing dataset')
    if filterData:
        bags_dataset = data_dir
    else:
        bags_dataset = glob.glob(data_dir + '/*')
        
    os.makedirs(feat_dir, exist_ok = True)
    
    print('loading model checkpoint')
    if self_supervised ==True:
        model_path = "models/tenpercent_resnet18.ckpt"
        model = torchvision.models.__dict__["resnet18"](pretrained=False)
        state = torch.load(model_path, map_location="cuda:0")
        state_dict = state["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "").replace("resnet.", "")] = state_dict.pop(key)
        
        model = load_model_weights(model, state_dict)
        model.fc = torch.nn.Sequential()

    else:
        model = Resnet50_baseline(pretrained = True)
        
    model = model.to(device)
    model.eval()
    total = len(bags_dataset)

        
    for bag_candidate_idx in range(total):
        bag_candidate = bags_dataset[bag_candidate_idx]
        bag_name = os.path.basename(os.path.normpath(bag_candidate))
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        bag_base = bag_name.split('\\')[-1]
        if not os.path.exists(os.path.join(feat_dir, bag_base+'.pt')):
            print(bag_name)
            output_path = os.path.join(feat_dir, bag_name)
            file_path = bag_candidate
            time_start = time.time()
            output_file_path = Compute_w_loader(file_path, output_path, 
    												model = model, batch_size = batch_size, 
    												verbose = 1, print_every = 20,
    												target_patch_size = target_patch_size)
            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
            if os.path.exists (output_file_path):
                file = h5py.File(output_file_path, "r")
                features = file['features'][:]
                print('features size: ', features.shape)
                print('coordinates size: ', file['coords'].shape)
                features = torch.from_numpy(features)
                #bag_base, _ = os.path.splitext(bag_name)            
                torch.save(features, os.path.join(feat_dir, bag_base+'.pt'))
                file.close()

##############################################################################









































