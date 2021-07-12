# HIA (Histopathology Image Analysis)

This repository contains the Python version of the general workflow offered by https://www.sciencedirect.com/science/article/pii/S0016508520348186?via%3Dihu for classification of the digital histopathology images.
This is important to notice that there are various changes in this version but it follows the same steps.

++ Scipts are still under the development and please always use the final version of it ++

## How to use this repository:
To use this workflow, you need to modfiy specific experiement file based on your project. Experiment file is a text file and an example of it can be find this repository. For this file you need to fill the following options:

Input Variable name | Description
--- | --- 
-projectDetails | This is an optional string input. In this section you can write down some keywords about your experiment.| 
-dataDir_train | Path to the directory containing the normalized tiles. For example : ["K:\\TCGA-CRC-DX"]. <br/> This folder should contain a subfolder of tiles which can have one of the following names: <br/> {BLOCKS_NORM_MACENKO, BLOCKS_NORM_VAHADANE, BLOCKS_NORM_REINHARD or BLOCKS}. <br/>The clinical table and the slide table of this data set should be also stored in this folder. <br/>This is an example of the structure for this folder: <br/> K:\\TCGA-CRC-DX: <br/> { <br/> 1. BLOCKS_NORM_MACENKO <br/>2. TCGA-CRC-DX_CLINI.xlsx <br/>3. TCGA-CRC-DX_SLIDE.csv <br/> }
-dataDir_test | If you are planning to have external validation for your experiemnt, this varibal is the path to the directory containing the normalized tiles which will be used in external validation. This folder should have the same structure as the 'dataDir_train'.
-targetLabels | This is the list of targets which you want to analyze. The clinical data should have the values for these targets. For Example : ["isMSIH", "stage"].
-trainFull | If you are planning to do cross validation, this variable should be defined as False. If you want to use all the data to train and then use the external validation, then this variable should be defined as True.
-maxNumBlocks | This integer variable, defines the maximum number of tiles which will be used per slide. Since the number of extracted tiles per slide can vary alot, we use limited number of tiles per slide. For more detail, please ckeck the paper.
-epochs | This integer variable, defines the number of epochs for training. 
-batchSize |  This integer variable, defines the batch size for training. 
-k | This integer variable, defined the number of K for cross validation experiment. This will be considered only if the trainFull variable has the value of False.
-modelName | This is a string variable which can be defined using one of the following neural network models. The script will download the pretrained weights for each of these models.<br/> {resnet, alexnet, vgg, squeezenet, densenet, inception, vit, efficient}
-opt | This is a string variable defining the name of optimizer to use for training. <br/> {"adam" or "sgd"}
-lr | This float variable defines the learning rate for the optimizer.
-reg | This float variable defines the weight_decay for the optimizer.
-gpuNo | If the computer has more than one gpu, this variable can be assigned to run the experiment on specified gpu. 
-freezeRatio | This is a float variable which can vary between [0, 1]. It will specified the ratio of the neural network layers to be freezed during the training. 

## Run training :

To start training, we use the Main.py script. The full path to the experiemnt file, should be used as an input variable in this script.

## External Validation:

If you used trainFull = True in the experiemnt file and you want to evaluate your model on the external data set, you should use the script named Deploy_Classic.py. In this script, following two inputs should be filled: <br/>  { <br/>  1. addressExp: is the full path to the experiment file created for external validation. This experiemnt file has the same features as explained above. DataDir_test is the path to folder of dataset which will be used for external validation. The targetLabels is a single target which you want to evaluate. <br/>  2. modelAdr is the full path to the model which is saved in the RESULT folder of the experiemnt which you defined trainFull as True.<br/> } 

