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
from torchvision import transforms
from PIL import Image

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
        img_stat = {}
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
            #img_stat[id_1] = []
        
            for j in range(4):
                img_file = img_files_sorted[i*4+j]
                img_id = img_file.split("_")[0]
                channel = img_file.split("_")[1]
                
                if (id_1 == img_id):
                    for x in ch_filter: # keep only the 3 channels that are in the list "ch_filter"
                        if x in channel:
                            image_dict[id_1].append(img_file)
                            #image_path = os.path.join(params.train_data_path,img_file)
                            #img1 = misc.imread(image_path)  #convert to a numpy array
                            #img_mean = np.mean(img1)
                            #img_std = np.std(img1)
                            #img_stat[img_file] = [img_mean, img_std]                                             
                else:
                    print("image ids not matching")
            
            
            image_count += 1
        print("image id count = ", image_count)
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
        

    def fetch_dataloader(data_type, data_dir, params):
        """
        Fetches the DataLoader object for each type in types from data_dir.

        types: (list) has one or more of 'train', 'val', depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
        Returns:
        data: (dict) contains the DataLoader object for each type in types
        """
        
        dataloaders = {}
        
        for split in ['train', 'val']:
            if split in data_type:
                path = os.path.join(data_dir,"{}_resized".format(split))
                
                if split == 'train':
                    dl = DataLoader(ResizedDataSet(path, train_transform), batch_size=params.batch_size, shuffle=True, num_workers=1)
                else:
                    dl = DataLoader(ResizedDataSet(path, eval_transform), batch_size=params.batch_size, shuffle=False, num_workers=1)
                
                dataloaders[split] = dl
        
        return dataloaders

    
    def data_iterator(self, params, image_dict, label_dict, shuffle = True):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.
        Args:
        
            params: (Params) hyperparameters of the training process.
            image_dict: dictionary of image files, keis image id, value is image file name
            label_dict: dictionary of labels, key is image id, value is a tensor of multi-one hot of size 28
            shuffle: (bool) whether the data should be shuffled
        Yields:
            batch_data: (Variable) dimension batch_size x channels X image_size_X X image_size_Y 
            batch_labels: (Variable) dimension batch_size x 4 with the corresponding labels
        """
        
        image_count = len(image_dict)
        channels = params.channels
        size_X = params.im_size_X
        size_Y = params.im_size_Y
        print("output image size =  ", size_X, size_Y )
        
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
            batch_labels = torch.zeros([minibatch_size, 28], dtype=torch.float32)
            
            print("mini batch index = ", i)
            # one pass over all the images in the mini batch
            for k in range(minibatch_size):
                #j = batch_image_index[k]
                image_id = batch_image_index[k]
                #raw_img = torch.zeros([channels, 512,512], dtype = torch.float32)
                for m in range(channels):
                    image_name = image_dict[image_id][m] # get the image file name
                    image_path = os.path.join(params.train_data_path,image_name)
                    
                    #img1 = misc.imread(image_path)  #read image to a numpy array
                    
                    img1 = Image.open(image_path) #read image using PIl.Image
                    #raw_img[m,:,:] = img1
                    transform = self.image_transform("train", params.im_size_X)
                    img_tansformed = transform(img1)
                    #retrieve mean and std
                    img_mean = torch.mean(img_tansformed)
                    img_std = torch.std(img_tansformed)
                    #normalize the image to zero mean and std = 1
                    img_normalized = (img_tansformed-img_mean)/img_std

                    #image_data[0,m,:,:] = img_normalized
                    
                    batch_data[k,m,:,:] = img_normalized
                batch_labels[k, :] = torch.tensor(label_dict[image_id] , dtype = torch.float32)
            
            
            # shift tensors to GPU if available
            #if params.cuda:
                #batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
                

            # convert them to Variables to record operations in the computational graph
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
    
            yield batch_data, batch_labels
        
    def image_transform(self, data_type, size):
        if (data_type == 'train'):
            transform = transforms.Compose([transforms.Resize(size),transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.resize(size),transforms.ToTensor()])
        
        return transform
        
    def get_random_image_id(self, data_type, params):
        '''
        get the image id of a random image from the image_type specified
        Inputs:
        imgae type: train or test
        
        '''
        if (data_type == 'train'):
            current_path = params.train_data_path
            #print("datatype is train")
        elif data_type == 'test':
            current_path = params.test_data_path
        
        img_files = os.listdir(current_path)
        file_count = len(img_files)
        img_files_sorted = sorted(img_files)  # sort the file list by name
        
        img_num = np.random.randint(0,file_count)
        img_file = img_files_sorted[img_num]
        
        img_id = img_file.split("_")[0]
        
        return img_id
    
    def load_single_image(self, data_type, params, img_id, ch_filter = ["red", "blue", "green"]):
        if (data_type == 'train'):
            current_path = params.train_data_path
        elif data_type == 'test':
            current_path = params.test_data_path
        
        img_channels = []
        image_data = torch.zeros([1, 3, params.im_size_X, params.im_size_Y], dtype=torch.float32)
        
        ch_id = 0 
        for chn in ch_filter:
            file_name = img_id+"_"+chn+".png"
            img_channels.append(file_name)
            image_path = os.path.join(params.train_data_path,file_name)   
            #img1 = misc.imread(image_path)  #read image to a numpy array
            img1 = Image.open(image_path)
            
            transform = self.image_transform("train", params.im_size_X)
            img_tansformed = transform(img1)
            #retrieve mean and std
            img_mean = torch.mean(img_tansformed)
            img_std = torch.std(img_tansformed)
            #normalize the image to zero mean and std = 1
            img_normalized = (img_tansformed-img_mean)/img_std
                
            image_data[0,ch_id,:,:] = img_normalized
            
            ch_id += 1
            
        return image_data
        
        
        
        
