import torchvision.models as models
import torch.nn as nn


class FineTunemodel():

	def __init__(self):
		super(FineTunemodel, self).__init__()

	def vgg19_bn():
		model_vgg19 = models.vgg19_bn(pretrained=True)
		
		for param in model_vgg19_2.parameters():
			param.requires_grad = False
		
		layer7_in_ftrs = model_vgg19_2.classifier[6].in_features
		features = list(model_vgg19_2.classifier.children())[:-1]
		features.extend([torch.nn.Linear(layer7_in_ftrs, 28), torch.nn.Sigmoid()]) # add new layers and sigmoid output
		model_vgg19_2.classifier = torch.nn.Sequential(*features)

