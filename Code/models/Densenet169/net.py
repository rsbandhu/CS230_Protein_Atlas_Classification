import torchvision.models as models
import torch.nn as nn

"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sklearn
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
from models.Densenet169 import densenet
import os

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.
    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()

        # 1 fully connected layers to transform the output feature of VGG19_BN 
        features_in = params.features_in
        self.fc1 = nn.Linear(features_in, 2048)  # input size: batch_size x 4096, output size: batch_size x 1024
        self.fc2 = nn.Linear(2048, 2048)  # input size: batch_size x 4096, output size: batch_size x 1024
        self.fc3 = nn.Linear(2048, 28)  # input size: batch_size x 1024, output size: batch_size x 28
        
    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        
        fc1_out = self.fc1(s)
        fc1_act = F.dropout(F.relu(fc1_out), p=0.3)
        fc2_out = self.fc2(fc1_act)
        fc2_act = F.dropout(F.relu(fc2_out), p=0.3)
        fc3_out = self.fc3(fc2_act)
        output = F.sigmoid(fc3_out)                      # output size = batch_size x 28

        return output

def myDensenet169(model_dir, num_classes):
                                    
    densenet169 = torchvision.models.densenet169(pretrained=True)
    #densenet169 = densenet.densenet169() # load model from local repo
    #pretrained_wts = os.path.join(model_dir, 'densenet169-b2777c0a.pth')
    #densenet169.load_state_dict(torch.load(pretrained_wts))

    print('done loading weights')
    # Change the number of output classes from 1000 to 28

    #Handle the primary FC layers
    num_ftrs = densenet169.classifier.in_features
    densenet169.classifier = nn.Linear(num_ftrs, num_classes)
    
    return densenet169

def myDensenet161(model_dir, num_classes):
                                    
    densenet161 = torchvision.models.densenet161(pretrained=True)
    #densenet169 = densenet.densenet169() # load model from local repo
    #pretrained_wts = os.path.join(model_dir, 'densenet169-b2777c0a.pth')
    #densenet169.load_state_dict(torch.load(pretrained_wts))

    print('done loading weights')
    # Change the number of output classes from 1000 to 28

    #Handle the primary FC layers
    num_ftrs = densenet161.classifier.in_features
    densenet161.classifier = nn.Linear(num_ftrs, num_classes)
    
    return densenet161


def loss_fn(outputs, labels, wts):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.
    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.
    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    loss_noreduce = nn.BCEWithLogitsLoss(reduce=False)
    loss = torch.mean(loss_noreduce(outputs, labels)*wts)
	
    # compute cross entropy loss for all tokens
    return loss
    
    
def metrics(outputs, labels, threshold):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.
    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.
    Returns: (float) accuracy in [0,1]
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


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
#metrics = {'accuracy': accuracy[0]}
    # could add more metrics such as accuracy for each token type
