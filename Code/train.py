import numpy as np
import argparse
import os
import torch
import torch.optim as optim
import utils

#import model.net as net
from models.data_loader import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='models/',help="Directory of the param file under models dir")

def train(data_iterator, params, num_batches):
    
    for batch in range(num_batches):
       train_batch, labels_batch = next(data_iterator) 
    
def train_and_evaluate(params, image_dict, labels_dict):
    image_count = len(image_dict)
    num_batches = (image_count +1) // params.batch_size
    for epoch in range(params.num_epochs):
        
        '''Do the following for every epoch'''
        
        train_data_iterator = data_loader.data_iterator(params, image_dict, labels_dict)
        train(train_data_iterator, params, num_batches)
        
if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    
    print(json_path)
    assert os.path.isfile(json_path), "No json configuration found at {}".format(json_path)

    params = utils.Params(json_path)
    data_loader = Dataloader(params)
    image_dict = data_loader.load_data("train", params)
    labels_dict = data_loader.load_labels("train", params)
    train_and_evaluate(params, image_dict, labels_dict)
    
    