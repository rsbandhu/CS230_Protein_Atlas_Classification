import numpy as np
import logging
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import utils

#Inception_V3_finetune
#Densenet169

# Change the following 4 lines for new models
#import models.resnet18.net as net 
import models.Densenet169.net as net
from models import data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
# Change the following 1 lines for new models
parser.add_argument('--model_dir', default='/home/ec2-user/CS230_Project/Code/models/Densenet169',help="Directory of the params.json file under models dir")
'''
parser.add_argument('--data_dir', default='data/train299_test', help="Directory containing the dataset")

parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
'''

def train(model, optimizer, loss_fn, dataloader, metrics, params, img_count, threshold, cuda_present):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    epoch_metric_summ = []
    loss_avg = utils.RunningAverage()
    loss_class_wts = torch.tensor(params.wts, dtype=torch.float32)
    
    #threshold = params.threshold #threshold value above which a class is considered present
    threshold = threshold  #used for parametric threshold

    y_pred = torch.zeros(img_count, params.class_count)
    y_true = torch.zeros(img_count, params.class_count)
    
    if cuda_present:
        loss_class_wts = loss_class_wts.cuda()
    k= 0
        
    for i, (train_batch, labels_batch) in enumerate(dataloader):

        batch_size = labels_batch.size()[0] 
        y_true[k:k+ batch_size, :] = labels_batch #build entire array of predicted labels
        
        batchlabel = labels_batch

        #If CUDA available, move data to GPU
        if cuda_present:
            train_batch = train_batch.cuda(async=True)
            labels_batch = labels_batch.cuda(async=True)

        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # compute model output and loss
        prim_out = model(train_batch)

        #Compute primary, Aux and total weighted loss
        loss_prim =loss_fn(prim_out, labels_batch,loss_class_wts)
        
        loss = loss_prim

        #send the primary output after thresholding for metrics calc
        yp = ((prim_out > threshold).int()*1).cpu()
        y_pred[k:k+ batch_size, :] = yp #build entire array of predicted labels
        k += batch_size

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        
        # Evaluate metrics only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            #prim_out = prim_out.data.cpu()
            #labels_batch = labels_batch.data.cpu()

            # compute all metrics on this batch
            summary_batch = metrics(yp, batchlabel, threshold)
            summary_batch['loss'] = loss.item()
            epoch_metric_summ.append(summary_batch)

            metrics_string = " ; ".join("{}: {:06.4f}".format(k, v) for k, v in summary_batch.items())
            logging.info("Batch: {} : - Train metrics: ".format(i) + metrics_string)

        # update the average loss
        loss_avg.update(loss.item())
        
    #Calculate the metrics of the entire training dataset
    epoch_metrics = metrics(y_pred, y_true, threshold)
    epoch_metrics['loss'] = loss_avg()

    # compute epoch mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in epoch_metric_summ]) for metric in epoch_metric_summ[0]}
    metrics_string = " ; ".join("{}: {:06.4f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("Batch: {} : - Train set average metrics: ".format(i) + metrics_string)
    
    train_metrics_string = " ; ".join("{}: {:06.4f}".format(k, v) for k, v in epoch_metrics.items())
    logging.info("Batch: {} : - metrics for Entire train dataset: ".format(i) + train_metrics_string)
            
def train_and_evaluate(model, params, dataloader, optimizer, loss_fn, metrics, model_dir, threshold, cuda_present):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """

    for epoch in range(params.num_epochs):
        
        t0 = time.time()
        '''Do the following for every epoch'''
        # Run one epoch
        logging.info("Epoch {}/{} , threshold: {}".format(epoch + 1, params.num_epochs, threshold))
        
        train_image_dict = dataloader.load_data("train", params)
        train_labels_dict = dataloader.load_labels("train", params)
        train_img_count = len(train_image_dict)
        train_data_generator = dataloader.data_iterator(params, "train", train_image_dict, train_labels_dict)
        
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_data_generator, metrics, params, train_img_count, threshold, cuda_present)

        # Evaluate for one epoch on validation set
        val_image_dict = dataloader.load_data("val", params)
        val_labels_dict = dataloader.load_labels("val", params)
        val_img_count = len(val_image_dict)
        val_data_generator = dataloader.data_iterator(params, "val", val_image_dict, val_labels_dict)
        val_metrics = evaluate(model, loss_fn, val_data_generator, metrics, params, val_img_count, threshold, cuda_present)
        
        #val_acc = val_metrics['accuracy']
        #is_best = val_acc>=best_val_acc
        is_best = False
        chk_file_name = 'last.pth.tar'
        lr = params.learning_rate
        # Save weights every 3rd epoch
        if ((epoch+1) % 3 == 0):
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()},
                                   is_best,
                                   model_dir, chk_file_name)
        t1 = time.time()
        logging.info("Time taken for this epoch = {}".format(t1-t0))
        
    # Save weights in the end
    chk_file_name = 'D161_'+'epoch_'+str(epoch)+':'+'lr_'+str(lr)+':'+'th_'+str(threshold)+'.pth.tar'
    utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best, model_dir, chk_file_name)
if __name__ == "__main__":
    print('first line in main')
    args = parser.parse_args()
    model_dir = args.model_dir
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train_2.log'))
    
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration found at {}".format(json_path)
    
    logging.info("json path : "+json_path)
    
    #Read params file
    params = utils.Params(json_path)

    for item in params.dict:
        logging.info(item + " : " + str(params.dict[item]))
    #Generate Dataloader
    logging.info("Generating the dataloader")
    dataloader = data_loader.Dataloader(params)    
    logging.info("Done loading the Dataloader")
    
    # use GPU if available
    cuda_present = torch.cuda.is_available() #Boolean
    
    if cuda_present:
        logging.info("using CUDA")
    else:
        logging.info("cuda not available, using CPU")
    
    logging.info("Loading model and weights")
    

    for t in range(1):
        # Change the following 1 lines for new models
        model = net.myDensenet161(model_dir, params.class_count)
        
        threshold = 0
        logging.info("Transferring model to GPU if CUDA available")
        for param in model.parameters():
            param.requires_grad = True
        if cuda_present:
            model = model.cuda()
        
        #Specify the loss function Optimizer    
        optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)
    
        #loss_fn = nn.BCEWithLogitsLoss()  # moving to net.py
        loss_fn = net.loss_fn
        #Calculate metrics
        metrics = net.metrics
    
        # Train and Evaluate the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        #train_and_evaluate(params, image_dict, labels_dict)
        train_and_evaluate(model, params, dataloader, optimizer, loss_fn, metrics, model_dir, threshold, cuda_present)
