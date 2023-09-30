import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

from configs.data_configs import data_configs


class ClassificationDataset(Dataset):
	def __init__(self, dataset_name, split, transform):
		self.dataset_name = dataset_name
		self.split = split
		self.transform = transform

		self.config = data_configs.datasets[self.dataset_name]
		self.data = torch.load(self.config.preprocessed_data_path)
		self.classes = self.data["class_list"]
		self.split_data = self.data[self.split]
	
	def __len__(self):
		return len(self.split_data["labels"])
	
	def __getitem__(self, idx):
		image_path = self.split_data["image_paths"][idx]
		label = self.split_data["labels"][idx]

		image = Image.open(image_path)
		if self.transform is not None:
			image = self.transform(image)
		
		return image.unsqueeze(0), label
