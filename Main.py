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
parser.add_argument('--adressExp', type = str, default = r"F:\CR07_Experiments\CR07_preOp_ResNet.txt", help = 'Adress to the experiment File')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

warnings.filterwarnings("ignore")

############################################### ################################

if __name__ == '__main__':
        
    args = utils.ReadExperimentFile(args)
    print(args.project_name)
    torch.cuda.set_device(args.gpuNo)
    args.useCsv = False
    if args.useClassicModel:
        stats_total, stats_df = ClassicTraining(args)
    else:
        ClamMILTraining(args)
        
        
        
        
         
        
        
        
        
        
        
        
        
        
        
        
        