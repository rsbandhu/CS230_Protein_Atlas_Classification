"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import models.Inception_V3_finetune.net as net
from models import data_loader

import sklearn
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params, img_count, threshold, cuda_present):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
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
    
    # compute metrics over the dataset
    with torch.no_grad():
        for i, (data_batch, labels_batch) in enumerate(dataloader):

            batch_size = labels_batch.size()[0] 
            y_true[k:k+ batch_size, :] = labels_batch #build entire array of predicted labels
        
            batchlabel = labels_batch

            # move to GPU if available
            if cuda_present:
                data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
            # compute model output and loss
            prim_out = model(data_batch)

            #Compute primary, Aux and total weighted loss
            loss_prim =loss_fn(prim_out, labels_batch,loss_class_wts)
            loss = loss_prim
        
            #send the primary output after thresholding for metrics calc
            yp = ((prim_out > threshold).int()*1).cpu()
            y_pred[k:k+ batch_size, :] = yp #build entire array of predicted labels
            k += batch_size
        
            summary_batch = metrics(yp, batchlabel, threshold)
            summary_batch['loss'] = loss.item()
            epoch_metric_summ.append(summary_batch)

            loss_avg.update(loss.item())
    # compute epoch mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in epoch_metric_summ]) for metric in epoch_metric_summ[0]}
    metrics_string = " ; ".join("{}: {:06.4f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("Batch: {} : - Dev set average metrics: ".format(i) + metrics_string)
    
    #Calculate the metrics of the entire dev dataset
    epoch_metrics = metrics(y_pred, y_true, threshold)
    epoch_metrics['loss'] = loss_avg()
    
    train_metrics_string = " ; ".join("{}: {:06.4f}".format(k, v) for k, v in epoch_metrics.items())
    logging.info("Batch: {} : - metrics for Entire Dev dataset: ".format(i) + train_metrics_string)
    
    return epoch_metrics

def metrics(outputs, labels, threshold):
    """
    Compute the accuracy, given the outputs and labels for all tokens. 
    Args:
        outputs: (torch tensor) dimension batch_size* Class Size (1/0 value for each entry)
        labels: (torch tensor) dimension batch_size* Class Size (1/0 value for each entry)
    Returns: Dictionary of accuracy, Hamming Loss, precision, Recall and F1_score macro 
    """

    #convert the torch tensors to numpy
    y_pred = outputs.numpy()
    y_true = labels.numpy()
    
    #Predict 0/1 for each class based on threshold
    #y_pred[y_pred > threshold] = 1
    #y_pred[y_pred <= threshold] = 0
    
    #Calculate various metrics, for multilabel, multiclass problem
    accuracy = accuracy_score(y_true, y_pred)
    Hloss = hamming_loss(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'macro')
    recall = recall_score(y_true, y_pred, average = 'macro')
    F1_score = f1_score(y_true, y_pred, average = 'macro')
    
    macro_score = {'accuracy': accuracy, 'Hloss': Hloss, 'precision': precision, 'recall':recall, 'F1_score':F1_score }
    
    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return macro_score



if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # use GPU if available
    cuda_present = torch.cuda.is_available() #Boolean

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.myDensenet161(model_dir, params.class_count)

    #Load the pretrained weights
    pretrained_wts = os.path.join(model_dir, 'inception_v3_google-1a9a5a14.pth')
    model.load_state_dict(torch.load(pretrained_wts))

    print('done loading weights')
    # Change the number of output classes from 1000 to 28
    
    if cuda_present:
        model = model.cuda()
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
