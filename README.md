# CS230_Protein_Atlas_Classification
Human Protein Atlas Image Classification Classify subcellular protein patterns in human cells. In this competition, Kagglers will develop models capable of classifying mixed patterns of proteins in microscope images

Task

Data Organization

Root Folder has two main subfolder: Code and Data

"Data" folder:
"test" and "train" subfolder. The label for train data "train.csv" is in the "Data" folder
All the .png files are either inside "train" or "test" folder under "Data" folder

"Code" folder

train.py: main program that will be run to train and validate
evaluate.py: evaluates the network and calculates metrics
utils.py: several helper functions
Resize_dataset.py 
evaluate_manual_CAM.pynb: generates CAM plots
create_dev_train_set.py: helper functions for augmenting dataset

Folder "models":
data_loader.py: used to load the data. Currently configured only to load train data

Subfolders for each network
params.json: all the parameters needed for the model
net.py: creates the network and loss function

to run the training, issue the following command from the command line window

python train.py model_dir

where "model_dir" is the full path of the models directory
