# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:04:57 2021

@author: Narmin Ghaffari Laleh

reference : https://github.com/mahmoodlab/CLAM

"""

##############################################################################

import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils, models

##############################################################################

def eval_transforms(pretrained = False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

##############################################################################

class Whole_Slide_Bag (torch.utils.data.Dataset):
    def __init__(self, file_path, pretrained = False, custom_transforms = None, target_patch_size = -1):
        
        self.file_path = file_path
        self.raw_samples = glob.glob(file_path + '/*')
        self.roi_transforms = custom_transforms
        
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None
            
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

    
    def __len__(self):
        return len(self.raw_samples)
    
    
    
    def __getitem__(self, index):

        image_path = self.raw_samples[index]   
        temp = image_path.split('_(')   
        temp = temp[-1].replace(').jpg', '')
        coord =[int(temp.split(',')[0]) , int(temp.split(',')[1])]
                 
        image = Image.open(image_path)
        if self.target_patch_size is  not None:
            image = image.resize(self.target_patch_size)
            image = np.array(image)
        image = self.roi_transforms(image).unsqueeze(0)        
        return image, coord
    
##############################################################################



