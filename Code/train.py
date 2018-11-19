import numpy as np
import logging
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import utils

# THISneeds modification
import models.Inception_V3_finetune.net as net
from models.Inception_V3_finetune import data_loader
from evaluate import EvaluateMetrics
from models.Inception_V3_finetune import inception

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='models/Inception_V3_finetune',help="Directory of the params.json file under models dir")
parser.add_argument('--data_dir', default='data/train_299', help="Directory containing the dataset")
'''
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
'''

def train(model, optimizer, loss_fn, dataloader, metrics, params, cuda_present):
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
    summ = []
    loss_avg = utils.RunningAverage()
    # for i, (train_batch, labels_batch) in enumerate(dataloader):
    
    num_batches =2
    
    for batch in range(num_batches):
        train_batch, labels_batch = next(dataloader)
        if cuda_present:
           train_batch = train_batch.cuda(async=True)
           labels_batch = labels_batch.cuda(async=True)
           
# convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
        # compute model output and loss
        output_batch = model(train_batch)
        loss = sum((loss_fn(out_batch, labels_batch)) for out_batch in output_batch)
        
        
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        
        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)
            
        # update the average loss
        loss_avg.update(loss.data[0])
        
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
            
def train_and_evaluate(params, train_dataloader, optimizer, loss_fn, metrics, model_dir, cuda_present):
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
        
        '''Do the following for every epoch'''
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, cuda_present)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        
        '''
        
        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc
        
        '''
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)
        
        '''
        
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            
        '''
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        
        
if __name__ == "__main__":
    print('first line in main')
    args = parser.parse_args()
    model_dir = args.model_dir
    json_path = os.path.join(args.model_dir, 'params.json')
    
    print(json_path)
    assert os.path.isfile(json_path), "No json configuration found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    
    #Old method of dataloading
    dataloader = data_loader.Dataloader(params)
    image_dict = dataloader.load_data("train", params)
    labels_dict = dataloader.load_labels("train", params)

    data_generator = dataloader.data_iterator(params, image_dict, labels_dict)
    
    
    #get the train and val dataloaders
    
    logging.info("Loading the datasets ")
    
    '''
    dataloaders = Dataloader(['train', 'val'], args.data_dir, params)
    train_dl = Dataloader['train']
    val_dl = Dataloader['val']
    '''
    
    logging.info("Done loading the Dataloader")
    
    # use GPU if available
    cuda_present = torch.cuda.is_available() #Boolean
    
    #cuda_present = False
    
    # Specify the model and the Optimizer 
    inceptionV3 = inception.inception_v3() # load model from local repo
    pretrained_wts = os.path.join(model_dir, 'inception_v3_google-1a9a5a14.pth')
    inceptionV3.load_state_dict(torch.load(pretrained_wts))
    
    for param in inceptionV3.parameters():
        param.requires_grad = True
    
    if cuda_present:
        model = inceptionV3.cuda()
    else:
        model = inceptionV3 
    optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)
    
    #get the Loss function and metrics from net.py
    loss_fn = nn.BCELoss()
    metrics = net.metrics
    
    # Train and Evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    #train_and_evaluate(params, image_dict, labels_dict)
    train_and_evaluate(params, data_generator, optimizer, loss_fn, metrics, model_dir, cuda_present)
	
