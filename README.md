# HIA (Histopathology Image Analysis)
# The Repository is under massive update - please do not use it !!!
# The latest updates will be soon uploaded.

This repository contains the Python version of a general workflow for end-to-end artificial intelligence on histopathology images. It is based on workflows which were previously described in [Kather et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0462-y) and [Echle et al., Gastroenterology 2020](https://www.sciencedirect.com/science/article/pii/S0016508520348186?via%3Dihu). The objective is to predict a given *label* directly from digitized histological whole slide images (WSI). The *label* is defined on the level of *patients*, not on the level of pixels in a given WSI. Thus, the problems addressed by HAI are *weakly supervised problems*. Common labels are molecular subtype of cancer, binarized clinical outcome or treatment response. Compared to previous Matlab-based implementations of this framework (e.g. [DeepHistology](https://github.com/jnkather/DeepHistology)), this version is implemented using Python and PyTorch and is highly scalable and extensively validated in multiple clincially relevant problems. A key feature of HIA is that it provides an implementation of multiple artificial intelligence algorithms, including
- Classical resnet-based training (similar to [Kather et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0462-y))
- Vision transformers (inspired by  8Dosovitskiy et al., conference paper at ICLR 2021](https://arxiv.org/abs/2010.11929)
- Multiple instance learning (similar to [Campanella et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0508-1))
- CLAM - Clustering-constrained attention multiple instance learning (described in [Lu et al., Nature Machine Intelligence 2020](https://www.nature.com/articles/s41551-020-00682-w))

This is important to notice that there are various changes in this version but it follows the same steps.

++ These scripts are still under the development and please always use the final version of it ++

## How to use this repository:
To use this workflow, you need to modfiy specific experiement file based on your project. Experiment file is a text file and an example of it can be find this repository. For this file you need to fill the following options:

Input Variable name | Description
--- | --- 
-projectDetails | This is an optional string input. In this section you can write down some keywords about your experiment.| 
-dataDir_train | Path to the directory containing the normalized tiles. For example : ["K:\\\TCGA-CRC-DX"]. <br/> This folder should contain a subfolder of tiles which can have one of the following names: <br/> {BLOCKS_NORM_MACENKO, BLOCKS_NORM_VAHADANE, BLOCKS_NORM_REINHARD or BLOCKS}. <br/>The clinical table and the slide table of this data set should be also stored in this folder. <br/>This is an example of the structure for this folder: <br/> K:\\TCGA-CRC-DX: <br/> { <br/> 1. BLOCKS_NORM_MACENKO <br/>2. TCGA-CRC-DX_CLINI.xlsx <br/>3. TCGA-CRC-DX_SLIDE.csv <br/> }
-dataDir_test | If you are planning to have external validation for your experiemnt, this varibal is the path to the directory containing the normalized tiles which will be used in external validation. This folder should have the same structure as the 'dataDir_train'. For train full and cross validation experiements you don't need to fill this variable.
-targetLabels | This is the list of targets which you want to analyze. The clinical data should have the values for these targets. For Example : ["isMSIH", "stage"].
-trainFull | If you are planning to do cross validation, this variable should be defined as False. If you want to use all the data to train and then use the external validation, then this variable should be defined as True. In this case, 10% of the data will be randomly selected for the validation set for the early stopping option.
-numPatientToUse | This variable defines the number of patients to use for the experiment. If you use 'ALL' it will use all the available patients. However you can always use an integer value like "n" to run the experiment for randomly selected "n" patients.
-maxNumBlocks | This integer variable, defines the maximum number of tiles which will be used per slide. Since the number of extracted tiles per slide can vary alot, we use limited number of tiles per slide. For more detail, please ckeck the paper.
-minNumBlocks | This integer variable, defines the minimum number of tiles which are required for an slide to be used in the experiment.
-numHighScoreTiles | This integer variable, defines the number of high score tiles which will be generated from the final result. This variable is only necessary in case of external validation.
-numHighScorePatients | This integer variable, defines the number of high score patients which will be generated from the final result. This variable is only necessary in case of external validation.
-epochs | This integer variable, defines the number of epochs for training. 
-batchSize |  This integer variable, defines the batch size for training. 
-k | This integer variable, defined the number of K for cross validation experiment. This will be considered only if the trainFull variable has the value of False.
-repeatExperiment | This integer variable defines the number of repeatation for an experiment. The default value of this variable is 1.
-modelName | This is a string variable which can be defined using one of the following neural network models. The script will download the pretrained weights for each of these models.<br/> {resnet18, resnet50, vit, efficient}
-opt | This is a string variable defining the name of optimizer to use for training. <br/> {"adam" or "sgd"}
-lr | This float variable defines the learning rate for the optimizer.
-reg | This float variable defines the weight_decay for the optimizer.
-gpuNo | If the computer has more than one gpu, this variable can be assigned to run the experiment on specified gpu. 
-freezeRatio | This is a float variable which can vary between [0, 1]. It will specified the ratio of the neural network layers to be freezed during the training. 
-earlyStop | If you set the value of this variable to "True", then it will check the performance of validation set and will stop training if the validation loss after the defined epoch using "minEpochToTrain" for "patience" iteration does not decrease.
-minEpochToTrain | This integer variable, defines the minimum number of epoch which we want to train the model before applying the early stopping condition.
-patience | This integer variable, defines the number of iteration which it will wait to stop the training in case that the validation loss does not decrease.

## Run training :

To start training, we use the Main.py script. The full path to the experiemnt file, should be used as an input variable in this script.

## External Validation:

If you used trainFull = True in the experiemnt file and you want to evaluate your model on the external data set, you should use the script named Classic_Deployment.py. In this script, following two inputs should be filled: <br/>  { <br/>  1. addressExp: is the full path to the experiment file created for external validation. This experiemnt file has the same features as explained above. DataDir_test is the path to folder of dataset which will be used for external validation. The targetLabels is a single target which you want to evaluate. <br/>  2. modelAdr is the full path to the model which is saved in the RESULT folder of the experiemnt which you defined trainFull as True.<br/> } 

