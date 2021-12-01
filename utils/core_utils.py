# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:30:07 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################  

from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
import utils.utils as utils
from utils.data_utils import Get_split_loader

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from tqdm import tqdm
from sklearn import metrics
import torch.nn as nn
import numpy as np
import time
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
    
class EarlyStopping:
    def __init__(self, patience = 20, stop_epoch = 50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss    
        
##############################################################################

def Train_MIL_CLAM(datasets, fold, args, trainFull = False):
    
    patient_results = [] 
    test_auc = 0     
    train_split, val_split, test_split = datasets        
    print('-' * 30)
    print("Training on {} samples".format(len(train_split)))
    if val_split:
        print("Validating on {} samples".format(len(val_split)))   
    
    if not trainFull:        
        print("Testing on {} samples".format(len(test_split)))  
        
    print('-' * 30)                  
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.num_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()      
    model_dict = {"dropout": args.drop_out, 'n_classes': args.num_classes}
    
    if args.model_name != 'mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
    
    if args.model_name in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_name =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn = instance_loss_fn)
        elif args.model_name == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    elif args.model_name == 'mil':
        if args.num_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    model.relocate()        
    utils.Print_network(model)
    print('-' * 30)
    
    optimizer = utils.get_optim(model, args)
    
    train_loader = Get_split_loader(split_dataset = train_split, training = True, weighted = args.weighted_sample)
    val_loader = Get_split_loader(split_dataset = val_split, training = False)
    test_loader = Get_split_loader(split_dataset = test_split, training = False)

    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch = args.minEpochToTrain, verbose = True)

    else:
        early_stopping = None

    for epoch in range(args.max_epochs):       
        print('EPOCH: {}'.format(epoch))        
        if args.model_name in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:            
            Train_loop_CLAM(epoch = epoch, model = model, loader = train_loader, optimizer = optimizer, n_classes = args.num_classes,
                            bag_weight = args.bag_weight,loss_fn =  loss_fn)  
            if args.early_stopping:
                stop = validate_CLAM(fold = fold, epoch = epoch, model = model, loader = val_loader, n_classes = args.num_classes, early_stopping = early_stopping,
                                 loss_fn = loss_fn, results_dir = args.result_dir)                
        elif args.model_name == 'mil':           
            Train_loop_MIL(epoch = epoch, model = model, loader = train_loader, optimizer = optimizer, n_classes = args.num_classes,loss_fn = loss_fn) 
            if args.early_stopping:
                stop = Validate_MIL(fold = fold, epoch = epoch, model = model, loader = val_loader, n_classes = args.num_classes, early_stopping = early_stopping,
                                 loss_fn = loss_fn, results_dir = args.result_dir)                          
        if stop: 
            print('-' * 30)
            print("The Validation Loss Didn't Decrease, Early Stopping!!")
            break

    if (args.early_stopping) and (not trainFull):
        model.load_state_dict(torch.load(os.path.join(args.result_dir,"bestModelFold_" + str(fold))))
    if not trainFull:    
        patient_results, test_error, test_auc = Summary(model = model, loader = test_loader, n_classes = args.num_classes, modelName = args.model_name)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))           
    return model, patient_results, test_auc
                    
##############################################################################
   
def Train_loop_MIL(epoch, model, loader, optimizer, n_classes, loss_fn = None):   
    
    model.train()
    acc_logger = Accuracy_Logger(n_classes = n_classes)
    train_loss = 0.
    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in tqdm(enumerate(loader)):
        data, label = data.to(device), label.to(device)       
        logits, Y_prob, Y_hat, y_probs, results_dict = model(data)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()        
        train_loss += loss_value
        error = utils.calculate_error(Y_hat, label)
        train_error += error        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

##############################################################################
    
def Validate_MIL (fold, epoch, model, loader, n_classes, early_stopping = None, loss_fn = None, results_dir = None):
    
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label, _) in tqdm(enumerate(loader)):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, _ = model(data)
            acc_logger.log(Y_hat, label)           
            loss = loss_fn(logits, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()            
            val_loss += loss.item()
            error = utils.calculate_error(Y_hat, label)
            val_error += error            
    val_error /= len(loader)
    val_loss /= len(loader)
    if n_classes == 2:
        for i in range(n_classes):
            fpr, tpr, thresholds = metrics.roc_curve(labels, prob[:, i],  pos_label = i)
            auc = metrics.auc(fpr, tpr)   
    else:
        auc = roc_auc_score(labels, prob, multi_class = 'ovr')   
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))  
    if early_stopping:  
        
        if fold == 'FULL':
            ckpt_name = os.path.join(results_dir, "bestModel")
        else:
            ckpt_name = os.path.join(results_dir, "bestModelFold_" + str(fold))                         
        early_stopping(epoch, val_loss, model, ckpt_name = ckpt_name)
        
        if early_stopping.early_stop:
            return True
    print('-' * 30 + '\n')
    return False    
        
##############################################################################   
 
def Train_loop_CLAM(epoch, model, loader, optimizer, n_classes, bag_weight, loss_fn = None):
    
    model.train()
    acc_logger = Accuracy_Logger(n_classes = n_classes)
    inst_logger = Accuracy_Logger(n_classes = n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    print('\n')
    for batch_idx, (data, label) in tqdm(enumerate(loader)):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label = label, instance_eval = True)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        bag_weight = torch.tensor(bag_weight).to(device)
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss 
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)
        train_loss += loss_value
        error = utils.calculate_error(Y_hat, label)
        train_error += error        
        total_loss.backward()        
        optimizer.step()
        optimizer.zero_grad()
    train_loss /= len(loader)
    train_error /= len(loader)    
    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

##############################################################################    

def validate_CLAM(fold, epoch, model, loader, n_classes, early_stopping = None, loss_fn = None, results_dir = None):
    
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    val_inst_loss = 0.
    inst_count=0    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label, _) in tqdm(enumerate(loader)):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)            
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            instance_loss = instance_dict['instance_loss']            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value
            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()            
            error = utils.calculate_error(Y_hat, label)
            val_error += error
    val_error /= len(loader)
    val_loss /= len(loader)
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('clustering: class {}, acc {}, correct {}/{}'.format(i, acc, correct, count))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     
    if early_stopping:
        if fold == 'FULL':
            ckpt_name = os.path.join(results_dir, "bestModel")
        else:
            ckpt_name = os.path.join(results_dir, "bestModelFold_" + str(fold))
        early_stopping(epoch, val_loss, model, ckpt_name = ckpt_name)        
        if early_stopping.early_stop:
            return True
    print('-' * 30 + '\n')
    return False
        
##############################################################################
    
class Accuracy_Logger(object):

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count    
    
##############################################################################
    
def Summary(model, loader, n_classes, modelName = None):
    
    model.eval()
    test_error = 0.
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['FILENAME']
    case_ids = loader.dataset.slide_data['PATIENT']
    patient_results = {}

    for batch_idx, (data, label, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        fileName = slide_ids.iloc[batch_idx]
        patient = case_ids.iloc[batch_idx]
        
        with torch.no_grad():                
            _, Y_prob, Y_hat, _, _ = model(data)

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({fileName: {'PATIENT': patient,'FILENAME': fileName, 'prob': probs, 'label': label.item()}})
        error = utils.calculate_error(Y_hat, label)
        test_error += error
    test_error /= len(loader)
    if n_classes == 2:
        for i in range(n_classes):
            fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs[:, i],  pos_label = i)
            auc = metrics.auc(fpr, tpr)
            aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))
    return patient_results, test_error, auc    
    
##############################################################################    
    
def Train_model_Classic(model, trainLoaders, args, valLoaders = [], criterion = None, optimizer = None, fold = False):    
    
    since = time.time()    
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []  
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch = args.minEpochToTrain, verbose = True)    
    for epoch in range(args.max_epochs):
        phase = 'train'
        print('Epoch {}/{}\n'.format(epoch, args.max_epochs - 1))
        print('\nTRAINING...\n')        
        model.train() 
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(trainLoaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, y_hat = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(y_hat == labels.data)
        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)        
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)        
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print()        
        if valLoaders:
            print('VALIDATION...\n')
            phase = 'val'    
            model.eval()        
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(valLoaders):
                inputs = inputs.to(device)
                labels = labels.to(device)        
                with torch.set_grad_enabled(phase == 'train'):            
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, y_hat = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(y_hat == labels.data)        
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset)            
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc)) 
            if fold == 'FULL':
                ckpt_name = os.path.join(args.result_dir, "bestModel")
            else:
                ckpt_name = os.path.join(args.result_dir, "bestModelFold" + fold)                         
            early_stopping(epoch, val_loss, model, ckpt_name = ckpt_name)
            if early_stopping.early_stop:
                print('-' * 30)
                print("The Validation Loss Didn't Decrease, Early Stopping!!")
                print('-' * 30)
                break
            print('-' * 30)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history 
    
    
##############################################################################    
    
def Validate_model_Classic(model, dataloaders):
    
    phase = 'test'
    model.eval()
    probsList = []    
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(phase == 'train'):            
            probs = nn.Softmax(dim=1)(model(inputs)) 
            probsList = probsList + probs.tolist()
    return probsList 
        
##############################################################################    
    
def Train_model_AttMIL(model, trainLoaders, args, valLoaders = [], criterion = None, optimizer = None, fold = False):    
    
    since = time.time()    
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []   
    early_stopping = EarlyStopping(patience = args.patience, stop_epoch = args.minEpochToTrain, verbose = True)    
    for epoch in range(args.max_epochs):
        phase = 'train'
        print('Epoch {}/{}\n'.format(epoch, args.max_epochs - 1))
        print('\nTRAINING...\n')        
        model.train() 
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(trainLoaders):
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, y_hat = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(y_hat == labels.data)
        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)        
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)        
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print()        
        if valLoaders:
            print('VALIDATION...\n')
            phase = 'val'    
            model.eval()        
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(valLoaders):
                labels = labels.to(device)        
                with torch.set_grad_enabled(phase == 'train'):            
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, y_hat = torch.max(outputs, 1)
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(y_hat == labels.data)        
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset)            
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc)) 
            if fold == 'FULL':
                ckpt_name = os.path.join(args.result_dir, "bestModel")
            else:
                ckpt_name = os.path.join(args.result_dir, "bestModelFold" + fold)                         
            early_stopping(epoch, val_loss, model, ckpt_name = ckpt_name)
            if early_stopping.early_stop:
                print('-' * 30)
                print("The Validation Loss Didn't Decrease, Early Stopping!!")
                print('-' * 30)
                break
            print('-' * 30)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history     
    
##############################################################################   
    
def Validate_model_AttMIL(model, dataloaders):
    
    phase = 'test'
    model.eval()
    probsList = []    
    for inputs in tqdm(dataloaders):
        with torch.set_grad_enabled(phase == 'train'):            
            probs = nn.Softmax(dim=1)(model(inputs[0])) 
            probsList = probsList + probs.tolist()
    return probsList     
    
    
    
    
    
    
    
    
    
    
    