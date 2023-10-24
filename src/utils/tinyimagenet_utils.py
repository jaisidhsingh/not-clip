import torch
import os
from tqdm import tqdm
import clip


def id2class_mapping():
	save_path = "/workspace/datasets/tinyimagenet/id2class_mapping.pt"	
	if os.path.exists(save_path):
		return save_path

	else:
		anno_file = "/workspace/datasets/tinyimagenet/words.txt"
		mapping = {}

		with open(anno_file) as f:
			for line in tqdm(f.readlines()):
				line = line.strip()
				id_code = line[:9]
				class_name = line[10:].lower()
				
				if "," in class_name:
					class_name = " or ".join(class_name.split(",")[:3])

				mapping[id_code] = class_name

		torch.save(mapping, save_path)
		return save_path

def train_image_class():
	train_data_folder = "/workspace/datasets/tinyimagenet/train"
	mapping_path = id2class_mapping()	
	mapping = torch.load(mapping_path)

	train_ids = os.listdir(train_data_folder)
	train_ids.sort()
	train_classes = [mapping[i] for i in train_ids]

	save_path = "/workspace/datasets/tinyimagenet/train_dataset_helper.pt"
	if os.path.exists(save_path):
		return save_path, train_classes, train_ids
	
	else:
		dataset_mapping = {}
		train_data_folder = "/workspace/datasets/tinyimagenet/train"
		train_subfolders = 	[os.path.join(train_data_folder, i) for i in train_ids]
		
		for i, subfolder in enumerate(train_subfolders):
			image_subfolder = os.path.join(subfolder, "images")
			for fname in os.listdir(image_subfolder):
				image_path = os.path.join(image_subfolder, fname)
				dataset_mapping[image_path] = i
	
		torch.save(dataset_mapping, save_path)
		return save_path, train_classes, train_ids

def val_image_class():
	_, classes, ids = train_image_class()
	save_path = "/workspace/datasets/tinyimagenet/val_dataset_helper.pt"

	if os.path.exists(save_path):
		return save_path, classes, ids

	else:
		mapping_path = "/workspace/datasets/tinyimagenet/val_image_ids.pt"	
		mapping = torch.load(mapping_path)
		dataset_mapping = {}

		for k, v in mapping.items():
			image_path = k.replace("tiny-imagenet-200/", "")
			idx = ids.index(v)
			dataset_mapping[image_path] = idx
		
		torch.save(dataset_mapping, save_path)
		return save_path, classes, ids

def get_split_data(split):
	if split == "train":
		return train_image_class()
	elif split == "val":
		return val_image_class()

