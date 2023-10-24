import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import sys
import clip

src_path = os.path.abspath("../")
sys.path.append(src_path)

from configs.data_configs import data_configs
from utils.tinyimagenet_utils import *


class ClassificationDataset(Dataset):
	def __init__(self, dataset_name, split, transform=None):
		self.dataset_name = dataset_name
		self.split = split
		self.transform = transform

		self.config = data_configs.datasets[self.dataset_name]
		self.data = torch.load(self.config.preprocessed_data_path)
		self.classes = [item.lower().replace("_", " ") for item in self.data["class_list"]]
		self.split_data = self.data[self.split]
	
	def __len__(self):
		return len(self.split_data["labels"])
	
	def __getitem__(self, idx):
		image_path = self.split_data["image_paths"][idx]
		label = self.split_data["labels"][idx]

		image = Image.open(image_path)
		if self.transform is not None:
			image = self.transform(image)
			image = image.unsqueeze(0)
		
		return image, label


class PretrainDataset2(Dataset):
	def __init__(self, dataset_name, split, transform=None):
		self.dataset_name = dataset_name
		self.split = split
		self.transform = transform

		self.config = data_configs.datasets[self.dataset_name] 
		self.data = torch.load(self.config.preprocessed_data_path)
		self.classes = [item.lower().replace("_", " ") for item in self.data["class_list"]]
		self.split_data = self.data[self.split]

	
	def __len__(self):
		return len(self.split_data["image_paths"])
	
	def __getitem__(self, idx):
		image_path = self.split_data["image_paths"][idx]
		caption = self.split_data["labels"][idx]["caption"]

		image = Image.open(image_path)
		if self.transform is not None:
			image = self.transform(image)
			image = image.unsqueeze(0)
		
		return image, caption

class PretrainDataset(Dataset):
	def __init__(self, split, transform=None):
		self.split = split
		self.transform = transform
		
		helper_path, classes, ids = get_split_data(self.split)
		self.data = torch.load(helper_path)
		self.classes = [c.lower() for c in classes]
		self.ids = ids

		other_helper_path = '/workspace/clip-tests/prompting/src/not_prompt_helper_tokenized_batched.pt'
		self.other_helper = torch.load(other_helper_path)
	
	def __len__(self):
		return len(self.data.keys())

	def __getitem__(self, idx):
		keys = list(self.data.keys())
		k = keys[idx]
		image_path = k[3:]
		image_path = image_path.replace("../data", "/workspace/datasets")
		image = Image.open(image_path).convert("RGB")

		if self.transform is not None:
			image = self.transform(image)
				
		label = int(self.data[k])
		cname = self.classes[label]
		c = cname
		p = self.other_helper[c]['!bin']['P 1x77']
		n1 = self.other_helper[c]['!bin']['N1 kx77']
		n2 = self.other_helper[c]['!bin']['N2 1x77']
		cr = self.other_helper[c]['!bin']['C^R_1 kx77']
		return image, p, n1, n2, cr

