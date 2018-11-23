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
parser.add_argument('--data_dir', default='data/train299_test', help="Directory containing the dataset")
'''
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
'''

def train(model, optimizer, loss_fn, dataloader, metrics, params, img_count, cuda_present):
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
    
    threshold = params.threshold #threshold value above which a class is considered present
    
    y_pred = torch.zeros(img_count, params.class_count)
    y_true = torch.zeros(img_count, params.class_count)
    
    if cuda_present:
        loss_class_wts = loss_class_wts.cuda()
    k= 0
    
    for _ in tqdm(range(img_count//params.batch_size)):
    
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            batch_size = labels_batch.size()[0] 
            y_true[k:k+ batch_size, :] = labels_batch #build entire array of predicted labels

            #If CUDA available, move data to GPU
            if cuda_present:
               train_batch = train_batch.cuda(async=True)
               labels_batch = labels_batch.cuda(async=True)

            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            prim_out, aux_out = model(train_batch)

            #Compute primary, Aux and total weighted loss
            loss_prim =loss_fn(prim_out, labels_batch,loss_class_wts)
            loss_aux =loss_fn(aux_out, labels_batch, loss_class_wts)
            loss = loss_prim + 0.2 * loss_aux

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
                prim_out = prim_out.data.cpu()
                labels_batch = labels_batch.data.cpu()

                # compute all metrics on this batch
                summary_batch = metrics(prim_out, labels_batch, 0.5)

                summary_batch['loss'] = loss.item()
                epoch_metric_summ.append(summary_batch)

                metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in summary_batch.items())
                logging.info("Batch: {} : - Train metrics: ".format(i) + metrics_string)

            # update the average loss
            loss_avg.update(loss.item())
        
    #Calculate the metrics of the entire training dataset
    epoch_metrics = metrics(y_pred, y_true, threshold)
    epoch_metrics['loss'] = loss_avg()
    
    # compute epoch mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in epoch_metric_summ]) for metric in epoch_metric_summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("Batch: {} : - Train set average metrics: ".format(i) + metrics_string)
    
    train_metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in epoch_metrics.items())
    logging.info("Batch: {} : - metrics for Entire train dataset: ".format(i) + train_metrics_string)
            
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
        
        image_dict = train_dataloader.load_data("train", params)
        labels_dict = train_dataloader.load_labels("train", params)
        
        img_count = len(image_dict)
        data_generator = train_dataloader.data_iterator(params, image_dict, labels_dict)
        
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, data_generator, metrics, params, img_count, cuda_present)

        # Evaluate for one epoch on validation set
        #val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        
        #val_acc = val_metrics['accuracy']
        #is_best = val_acc>=best_val_acc
    is_best = True
        
   # Save weights
    utils.save_checkpoint({'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'optim_dict' : optimizer.state_dict()},
                           is_best=is_best,
                           checkpoint=model_dir)
    
if __name__ == "__main__":
    print('first line in main')
    args = parser.parse_args()
    model_dir = args.model_dir
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration found at {}".format(json_path)
    
    logging.info("json path : "+json_path)
    
    #Read params file
    params = utils.Params(json_path)
    
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
    inceptionV3 = net.myInceptionV3(model_dir, 28)

    logging.info("Transferring model to GPU if CUDA available")
    for param in inceptionV3.parameters():
        param.requires_grad = True
    if cuda_present:
        model = inceptionV3.cuda()
    else:
        model = inceptionV3 
        
    #Specify the loss function Optimizer    
    optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)
    
    #loss_fn = nn.BCEWithLogitsLoss()  # moving to net.py
    loss_fn = net.loss_fn
    #Calculate metrics
    metrics = net.metrics
    
    # Train and Evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    #train_and_evaluate(params, image_dict, labels_dict)
    train_and_evaluate(params, dataloader, optimizer, loss_fn, metrics, model_dir, cuda_present)
