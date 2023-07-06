# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:08:57 2021

@author: Narmin Ghaffari Laleh
"""

###############################################################################

from Classic_Training import Classic_Training
from CLAM_MIL_Training import CLAM_MIL_Training
from AttMIL_Training import AttMIL_Training

import utils.utils as utils

from pathlib import Path
import warnings
import argparse
import torch

# %%
parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
pathToExperimentFile = Path(".\Experiments\COHORT_RESNET18_CROSSVAL.txt")
parser.add_argument('--adressExp', type = str, default = pathToExperimentFile, help = 'Adress to the experiment File')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
print('\nTORCH Detected: {}\n'.format(device))

#%%

if __name__ == '__main__':
        
    args = utils.ReadExperimentFile(args)    
    
    if args.useClassicModel:
        Classic_Training(args)
        torch.cuda.set_device(args.gpuNo)
        
    elif args.model_name == 'attmil':
        AttMIL_Training(args)        
    else:            
        CLAM_MIL_Training(args)
        
        
        
        
         
        
        
        
        
        
        
        
        
        
        
        
        