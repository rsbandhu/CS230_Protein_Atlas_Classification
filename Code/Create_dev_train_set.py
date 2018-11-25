
# coding: utf-8

# In[1]:


import argparse
import random
import os
import numpy as np
import utils
import time as time
import shutil


# In[2]:


def load_labels(labels_file_path, ch_filter = "_green"):
        
        """
        Loads the labels from their corresponding files. 
        Creates a dict, key is the image name, value is the list of labels
        Args:
            labels_file: each line contains the labels for the corresponding image
        """
        
        label_file = open(labels_file_path, 'r')
        label_dict = {}
        imgid_dict = {}
        single_label_img = {}
        multi_label_img = {}
        
        # = open(params.train_label_file, 'r')
        
        line = label_file.readline()
        line = label_file.readline()
        while line:
            
            ytrain = np.zeros((28))
            line = line.strip()
            
            line_split = line.split(",") #split each line into image name and labels
            image_id = line_split[0] 
            #image_name = line_split[0]+ch_filter+".png" # add the filter and .png at the end
            label = line_split[1]
            imgid_dict[image_id] = label
            #print(image_id, label)
            labels = label.split(" ")
            if (len(labels) == 1):
                unq_label = labels[0]
                if (unq_label not in single_label_img):
                    single_label_img[unq_label] = [image_id]
                else:
                    single_label_img[unq_label].append(image_id)
            else:
                if (label not in multi_label_img):
                    multi_label_img[label] = [image_id]
                else:
                    multi_label_img[label].append(image_id)
                #print(label, image_id)
            
            line = label_file.readline()

        label_file.close()
        return (single_label_img, multi_label_img, imgid_dict)


# In[3]:


def Dev_class_count(class_count, p_dev):
    '''
    Given class count returns the number of items in dev class
    Args: class count, fraction of items that need to be moved to dev set
    Retruns: number of items in dev set
    
    '''
    if class_count < 5:
        dev_class_count = class_count
    else:
        dev_class_count = max(5, int(p_dev*class_count))
    return dev_class_count
            


# In[4]:


def create_dev_set_single_labels(single_label_img, p_dev):
    '''
    Randomly selects a subset of images that will be assigned to the dev set
    The random subset os created for each class separately
    All images will have single labels
    '''
    #Seed the random number generator for shuffling
    #np.random.seed(230)
    dev_count = 0 
    dev_image_set_1label = []
    for item in single_label_img:
        class_count = len(single_label_img[item])
        
        img_list = list(single_label_img[item])
        
        #Shuffle the order of the images if needed
        np.random.shuffle(img_list)
        
        dev_class_count = Dev_class_count(class_count, p_dev)
        #print(class_count, 0.05*class_count)
        dev_count += dev_class_count
        #print(item, "  dev_class_count =  ", dev_class_count, class_count)
        #print(img_list[:dev_class_count])
        dev_image_set_1label = dev_image_set_1label + img_list[:dev_class_count]
              
    print(dev_count, len(dev_image_set_1label))
    
    return dev_image_set_1label
    


# In[5]:


def create_dev_set_multiple_labels(multi_label_img, p_dev):
    '''
    Randomly selects a subset of images that will be assigned to the dev set
    These images have multiple labels
    Classes that occur infrequently are treated separately
    Arg:
    
    Returns: List of image id, channel names are removed
    '''
    #Seed the random number generator for shuffling
    np.random.seed(230)
    dev_count = 0 
    critical_class_list = [8,9,10, 15, 27] #list of classes that appear very few times
    img_list_class_label = {}
    
    multilabel_dev_img_list = [] #final list of images with multilabels in dev set
    
    for class_label in critical_class_list:
        img_list_class_label[str(class_label)] = []
    
    # Iterate over all different multilabels
    for item in multi_label_img:
        multilabellist = item.split(" ")
        crtitical_label_found = False
        
        #Check if the multilabel contains a critical class
        for class_label in critical_class_list:
            if (str(class_label) in multilabellist):
                crtitical_label_found = True
                img_list_class_label[str(class_label)] += multi_label_img[item]
        
        #if the label doesn't contain a critical class add all of them to a list
        if not(crtitical_label_found):
            dev_list = multi_label_img[item]
            multilabel_dev_img_list += dev_list
    
    #Take a fraction =p_dev of the images with non-critical labels to a list
    class_count = len(multilabel_dev_img_list)
    dev_class_count = Dev_class_count(class_count, p_dev)
    np.random.shuffle(multilabel_dev_img_list)
    multilabel_dev_img_list = multilabel_dev_img_list[:dev_class_count]
        
    # Choose a fraction =p_dev of the images that contains labels with critical class
    for class_label in critical_class_list:
        np.random.shuffle(img_list_class_label[str(class_label)])
        class_count = len(img_list_class_label[str(class_label)])
        dev_class_count = Dev_class_count(class_count, p_dev)
        
        multilabel_dev_img_list += img_list_class_label[str(class_label)][:dev_class_count]
         
    return multilabel_dev_img_list


# In[6]:


def Create_new_train_set(dev_imgid_list, imgid_dict_all):
    
    
    train_imgid_list = list(imgid_dict_all.keys())
    for img in dev_imgid_list:
        train_imgid_list.remove(img)
        
    return train_imgid_list


# In[7]:


def Create_img_files_path(imgid_list, home_dir, channels =["_red", "_blue", "_green", "_yellow"]):
    '''
    Create full path name of all dev set images
    Args: list of image id for dev set, directory of current location of images, channels
    Returns: list of full path names of all dev set images
    '''
    
    file_pathname_list = []
    for img_id in imgid_list:
        for channel in channels:
            file_name = img_id+channel+".png"
            file_pathname = os.path.join(home_dir, file_name)
            file_pathname_list.append(file_pathname)
            
    return file_pathname_list
        


# In[ ]:


if __name__ == '__main__':
    #Specify the root directory where the current train directory sits
    data_dir_root = '/home/bony/Deep_Learning_Stanford_CS230/Project/Data'
    original_train_filesdir = os.path.join(data_dir_root, 'train_224')
    new_train_filesdir = os.path.join(data_dir_root, 'train_224_new')
    new_dev_filesdir = os.path.join(data_dir_root, 'dev_224_new')
    labels_file_path = '/home/bony/Deep_Learning_Stanford_CS230/Project/Data/train.csv'

    np.random.seed(230)

    (single_label_img, multi_label_img, imgid_dict_all) = load_labels(labels_file_path)

    img_list = list(single_label_img.keys())
    print("Classes that have unique labels in some image:  ", img_list)
    for i in range(28):
        label =str(i)
        if label not in img_list:
            print(label, "This class doesn't appear as a unique label in any image")

    #create dev image set with single labels
    dev_image_set_1label = create_dev_set_single_labels(single_label_img, 0.05)
    print("dev set images with single labels = ", len(dev_image_set_1label))
    #create dev image set with multiple labels
    multilabel_dev_img_list = create_dev_set_multiple_labels(multi_label_img, 0.05)
    print("dev set images with multiple labels = ",len(multilabel_dev_img_list))
    #create full dev image set
    dev_imgid_list = dev_image_set_1label + multilabel_dev_img_list
    dev_imgid_list = set(dev_imgid_list) #remove duplicate items
    print("Number of dev set images = ", len(dev_imgid_list))

    #create new full train image set
    train_imgid_list = Create_new_train_set(dev_imgid_list, imgid_dict_all)
    print("Number of train images = ", len(train_imgid_list))

    #Create full path names for new  and dev images    
    dev_file_pathname_list = Create_img_files_path(dev_imgid_list, original_train_filesdir)
    train_file_pathname_list = Create_img_files_path(train_imgid_list, original_train_filesdir)

    #write new dev and train images to the new directory
    if not os.path.exists(new_dev_filesdir):
        print("Dev image files Directory does not exist! Making directory {}".format(new_dev_filesdir))
        os.mkdir(new_dev_filesdir)
        print("Copying files to new dev directory")
        for file_path in dev_file_pathname_list:
            shutil.copy2(file_path, new_dev_filesdir) # target filename is /dst/dir/file.ext
        print("done copying dev files")
    else:
        print("Dev image files Directory exists! ")


    if not os.path.exists(new_train_filesdir):
        print("Dev image files Directory does not exist! Making directory {}".format(new_train_filesdir))
        os.mkdir(new_train_filesdir)
        print("Copying files to new train directory")
        for file_path in train_file_pathname_list:
            shutil.copy2(file_path, new_train_filesdir) # target filename is /dst/dir/file.ext
        print("done copying train files")
    else:
        print("Train image files Directory exists! ")

