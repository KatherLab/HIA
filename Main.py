# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:08:57 2021

@author: Narmin Ghaffari Laleh
"""

###############################################################################

from ClassicTraining import ClassicTraining
from ClamMILTraining import ClamMILTraining
import utils.utils as utils

import warnings
import argparse
import torch

###############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"F:\TCGA-BRCA_Experiments\TCGA-BRCA_SS_CLAM.txt", help = 'Adress to the experiment File')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

warnings.filterwarnings("ignore")

############################################### ################################

if __name__ == '__main__':
        
    args = utils.ReadExperimentFile(args)
    print(args.project_name)
    #torch.cuda.set_device(args.gpuNo)
    
    args.new_Split = True
    args.feature_extract = False
    args.self_supervised = False
    
    if args.useClassicModel:
        ClassicTraining(args)
    else:
        ClamMILTraining(args)
        
        
        
        
         
        
        
        
        
        
        
        
        
        
        
        
        