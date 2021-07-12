# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:30:07 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################  

from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc
import utils.utils as utils
from utils.data_utils import Get_split_loader

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
#from focal_loss.focal_loss import FocalLoss
from tqdm import tqdm
from sklearn import metrics


import torch.nn as nn
import numpy as np
import time
import torch
import os
import copy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

##############################################################################

def Train_MIL_CLAM(datasets, cur, args, trainFull = False):

    print('\nTraining Fold {}!'.format(cur))
    
    writer_dir = os.path.join(args.result, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

  
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets        
    print('******************************************************************')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))        
    print('******************************************************************')          
    print('Done!')
    
    print('\nInit loss function...', end = ' ')
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.num_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
         loss_fn = FocalLoss(alpha=2, gamma=5)
    else:
        #sc = torch.tensor([1., 10.]).cuda()
        #loss_fn = nn.CrossEntropyLoss(weight = sc)
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

        
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.num_classes}
    
    if args.model_name == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_name != 'mil':
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
    
    else: # args.model_name == 'mil'
        if args.num_classes > 2:
            #model = MIL_fc_mc(**model_dict)
            print('It is not there YET!')
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    print('Done!')
    
    utils.Print_network(model)

    print('\nInit optimizer ...', end = ' ')
    optimizer = utils.get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end = ' ')
    train_loader = Get_split_loader(train_split, training = True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = Get_split_loader(val_split,  testing = args.testing)
    test_loader = Get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch = 50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        print('EPOCH: {}'.format(epoch))
        if args.model_name in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            Train_loop_CLAM(epoch, model, train_loader, optimizer, args.num_classes, args.bag_weight, writer, loss_fn)
            if not trainFull:
                stop = validate_CLAM(cur, epoch, model, val_loader, args.num_classes, 
                    early_stopping, writer, loss_fn, args.result)
            else:
                stop = False
        else:
            Train_loop_MIL(epoch = epoch, model = model, loader = train_loader, optimizer = optimizer, n_classes = args.num_classes, writer = writer, loss_fn = loss_fn)
            if not trainFull:
                stop = Validate_MIL(cur, epoch, model, val_loader, args.num_classes, 
                    early_stopping, writer, loss_fn, args.result)
            else:
                 stop = False   
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.result, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.result, "s_{}_checkpoint.pt".format(cur)))
    
    if not trainFull:
        _, val_error, val_auc, _= Summary(model = model, loader = val_loader, n_classes = args.num_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    
        results_dict, test_error, test_auc, acc_logger = Summary(model, test_loader, args.num_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    
        for i in range(args.num_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    
            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    
        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
        
            writer.close()
    else:
        results_dict = {}
        test_auc = 0
        val_auc = 0
        test_error = 0
        val_error = 0
        
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 
        
        
    
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss    
    
##############################################################################
   
def Train_loop_MIL(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes = n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, y_probs, results_dict = model(data)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        #if (batch_idx + 1) % 30 == 0:
            #print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = utils.calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
##############################################################################   
 
def Train_loop_CLAM(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes = n_classes)
    inst_logger = Accuracy_Logger(n_classes = n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        #print('')
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label = label, instance_eval = True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        bag_weight = torch.tensor(bag_weight).to(device)
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)
        #n_mask = np.equal(inst_labels, 0)
        #n_acc = (inst_labels[n_mask] == inst_preds[n_mask]).mean()
        #p_acc = (inst_labels[~n_mask] == inst_preds[~n_mask]).mean()

        train_loss += loss_value
        #if (batch_idx + 1) % 5 == 0:
            #print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                #'inst_p_acc: {:.4f}, inst_n_acc: {:.4f}, label: {}, bag_size: {}'.format(p_acc, n_acc, label.item(), data.size(0)))

        error = utils.calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
        
            if writer:
                writer.add_scalar('train/inst_class_{}_acc'.format(i), acc, epoch)

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

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
    
def Validate_MIL (cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
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
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False    
    
    
##############################################################################    

def validate_CLAM(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    #val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    #sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
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
            #n_mask = np.equal(inst_labels, 0)
            #n_acc = (inst_labels[n_mask] == inst_preds[n_mask]).mean()
            #p_acc = (inst_labels[~n_mask] == inst_preds[~n_mask]).mean()

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
            print('Clustering: class {}, acc {}, correct {}/{}'.format(i, acc, correct, count))
        
            if writer:
                writer.add_scalar('val/inst_class_{}_acc'.format(i), acc, epoch)
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

##############################################################################
    
def Summary(model, loader, n_classes):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    case_ids = loader.dataset.slide_data['case_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        case_id = case_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'case_id': case_id,'slide_id': slide_id, 'prob': probs, 'label': label.item()}})
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


    return patient_results, test_error, auc, acc_logger    
    
##############################################################################    
    
def Train_model_Classic(model, trainLoaders, valLoaders = [], criterion = None, optimizer = None, num_epochs = 25, is_inception = False, results_dir = ''):
    
    since = time.time()

    train_acc_history = []
    train_loss_history = []

    val_acc_history = []
    val_loss_history = []
    
    early_stopping = EarlyStopping(patience = 20, stop_epoch = 20, verbose = True)
    
    for epoch in range(num_epochs):
        phase = 'train'
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train() 
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(trainLoaders):
            inputs = inputs.to(device)
            #print(labels)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if is_inception and phase == 'train':
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)
        
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print()
        
        if valLoaders:
            phase = 'val'
    
            model.eval()   # Set model to evaluate mode
        
            running_loss = 0.0
            running_corrects = 0
            predList = []
            
            # Iterate over data.
            for inputs, labels in tqdm(valLoaders):
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                with torch.set_grad_enabled(phase == 'train'):            
                    #outputs = model(inputs)
                    outputs = nn.Softmax(dim=1)(model(inputs)) 
                    loss = criterion(outputs, labels)
        
                    _, preds = torch.max(outputs, 1)
                    predList = predList + outputs.tolist()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset)
            
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc))
            
            early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "checkpoint.pt"))
            if early_stopping.early_stop:
                print("Early stopping")
                break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history     
    
##############################################################################    
    
def Validate_model_Classic(model, dataloaders, criterion):
    
    phase = 'test'

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    predList = []
    
    # Iterate over data.
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(phase == 'train'):            
            #outputs = model(inputs)
            outputs = nn.Softmax(dim=1)(model(inputs)) 
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            predList = predList + outputs.tolist()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_acc = running_corrects.double() / len(dataloaders.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc, predList 
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    