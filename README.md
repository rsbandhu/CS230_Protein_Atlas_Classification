# CS230_Protein_Atlas_Classification
Human Protein Atlas Image Classification Classify subcellular protein patterns in human cells. In this competition, Kagglers will develop models capable of classifying mixed patterns of proteins in microscope images

Task

Data Organization

Root Folder has two main subfolder: Code and Data

The "Data" folder has "test" and "train" subfolder. The label for train data "train.csv" is in the "Data" folder
All the .png files are either inside "train" or "test" folder under "Data" folder

"Code" folder

train.py: main program that will be run to train and validate
utils.py:

Folder "models":
data_loader.py: used to load the data. Currently configured only to load train data
params.json: all the parameters needed for the model

to run the training, issue the following command from the command line window

python train.py model_dir

where "model_dir" is the full path of the models directory
