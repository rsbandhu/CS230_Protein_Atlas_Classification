#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:41:47 2018

@author: bony
"""
import os
import numpy as np
from scipy import misc
import torch
from torch.autograd import Variable

class Dataloader():
    
    def __init__(self, params):
        
        # loading dataset_params
        
        train_path = params.train_data_path
        print(train_path)
        #assert os.path.isfile(train_path), "No training file found at {}".format(train_path)
        
        test_path = params.test_data_path
        #assert os.path.isfile(test_path), "No test file found at {}".format(test_path)
        
    
    def load_data(self, data_type, params, ch_filter = ["red", "blue", "green"]):
        """
        Reads the image file names  
        Creates a dict, key is the image count, value is the image file name without the extension
        Args:
            channel filter: blue, green, red
        """
        image_dict = {}
        image_count = 0
        print(data_type)
        if (data_type == 'train'):
            current_path = params.train_data_path
            print("datatype is train")
        elif data_type == 'test':
            current_path = params.test_data_path
        
        img_files = os.listdir(current_path)
        file_count = len(img_files)
        img_files_sorted = sorted(img_files)  # sort the file list by name
        
        for i in range(file_count//4):  # there are 4 channels
            id_1 = img_files_sorted[i*4].split("_")[0] # split the file name by id and channel color
            image_dict[id_1] = []
            for j in range(4):
                
                id = img_files_sorted[i*4].split("_")[0]
                channel = img_files_sorted[i*4].split("_")[1]
                
                if (id_1 == id):
                    for x in ch_filter:
                        if x in channel:
                            image_dict[id_1].append(img_files_sorted[i*4+j])
                else:
                    print("image ids not matching")
        
            
            image_count += 1
        return image_dict
        
    
    
    def load_labels(self, data_type, params, ch_filter = "_green"):
        
        """
        Loads the labels from their corresponding files. 
        Creates a dict, key is the image name, value is the list of labels
        Args:
            labels_file: each line contains the labels for the corresponding image
        """
        
        #label_file = open("../train.csv", 'r')
        label_dict = {}
        label_file = open(params.train_label_file, 'r')
        
        line = label_file.readline()
        line = label_file.readline()
        while line:
            
            ytrain = np.zeros((28))
            line = line.strip()
            line_split = line.split(",") #split each line into image name and labels
            image_id = line_split[0]
            #image_name = line_split[0]+ch_filter+".png" # add the filter and .png at the end
            label = line_split[1]
            trainlabel = label.split(' ')
            
            for j in range(len(trainlabel)):
                
                index = int(trainlabel[j])
                ytrain[index]=1
            label_dict[image_id] = ytrain  #create the dict item for the image
            line = label_file.readline()

        label_file.close()
        return label_dict
        
    
    def data_iterator(self, params, image_dict, label_dict, shuffle = True):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.
        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled
        Yields:
            batch_data: (Variable) dimension batch_size x channels X image_size_X X image_size_Y 
            batch_labels: (Variable) dimension batch_size x 4 with the corresponding labels
        """
        
        image_count = len(image_dict)
        channels = params.channels
        size_X = params.im_size_X
        size_Y = params.im_size_Y
        
        order = list(image_dict.keys())
        #Shuffle the order of the images if needed
        if shuffle:
            np.random.seed(230)
            np.random.shuffle(order)
        minibatch_size = params.batch_size
        
        print( image_count, (image_count+1)//minibatch_size, minibatch_size)
        # one pass over data
        for i in range((image_count+1)//minibatch_size):
            # "i" denotes batch number
            mini_batch_index_start = i*minibatch_size
            mini_batch_index_end = (i+1)*minibatch_size
            batch_image_index = order[mini_batch_index_start:mini_batch_index_end ]
            
            # Create initial arrays of zeros for batch data and labels
            batch_data = torch.zeros([minibatch_size, channels, size_X, size_Y], dtype=torch.float32)
            batch_labels = torch.zeros([28, minibatch_size], dtype=torch.float32)
            
            print("mini batch index = ", i)
            # one pass over all the images in the mini batch
            for k in range(minibatch_size):
                #j = batch_image_index[k]
                image_id = batch_image_index[k]
                
                for m in range(channels):
                    image_name = image_dict[image_id][m] # get the image file name
                    image_path = os.path.join(params.train_data_path,image_name)
                    img1 = misc.imread(image_path)  #convert to a numpy array
                
                    batch_data[k,m,:,:] = torch.tensor(img1, dtype = torch.float32)
                batch_labels[:,k] = torch.tensor(label_dict[image_id] , dtype = torch.float32)
            
            # since all data are indices, we convert them to torch LongTensors
            #batch_data, batch_labels = torch.HalfTensor(batch_data), torch.ShortTensor(batch_labels)
            
            # shift tensors to GPU if available
            #if params.cuda:
                #batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
                

            # convert them to Variables to record operations in the computational graph
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
    
            yield batch_data, batch_labels
