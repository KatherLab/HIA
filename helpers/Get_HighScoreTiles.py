# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:03:09 2021

@author: Narmin Ghaffari Laleh
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

##############################################################################

totalPatientResultPath = r""
totalResultPath = r""
numHighScorePetients = 5
numHighScoreBlocks = 5

targetColName = 'MSIH'

# ATTENTION : if you want to get the tiles for true negative cases please change the following line to truePos = False

truePos = False

outputFolder = r''

imageName = 'HighScoreTiles.png'

##############################################################################
   
subplot_x = (numHighScoreBlocks)/5

data = pd.read_csv(totalPatientResultPath)
if truePos:
    dataTemp = data.loc[data['y_true_label'] == targetColName]
else:
    labels = list(set(data['y_true_label']))
    del labels[labels.index(targetColName)]
    dataTemp = data.loc[data['y_true_label'] == labels[0]]
    
dataTemp = dataTemp.sort_values(by = [targetColName])

if truePos:
    highScorePosPatients = list(dataTemp['patients'][-numHighScorePetients:])
    highScorePos = list(dataTemp[targetColName][-numHighScorePetients:])
else:
    highScorePosPatients = list(dataTemp['patients'][0 : numHighScorePetients])
    highScorePos = list(dataTemp[targetColName][0 : numHighScorePetients])
    
results = pd.read_csv(totalResultPath)
    
fig = plt.figure(figsize=(20,20))
i = 1

for index, patient in enumerate(highScorePosPatients):
    dataTemp = results.loc[results['patientID'] == patient]
    dataTemp = dataTemp.sort_values(by = [targetColName])
    if truePos:
        highScorePosTiles = list(dataTemp['X'][-numHighScoreBlocks:])
    else:
        highScorePosTiles = list(dataTemp['X'][0:numHighScoreBlocks]) 
        
    for tile in highScorePosTiles:            
        img = Image.open(tile)
        ax = plt.subplot(numHighScorePetients, numHighScoreBlocks, i)
        ax.set_axis_off()
        plt.imshow(img)
        i += 1  
plt.savefig(os.path.join(outputFolder,  imageName))