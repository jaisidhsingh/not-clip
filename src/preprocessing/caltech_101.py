import torch
from tqdm import tqdm
import os
import sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

from configs.data_configs import data_configs


def prepare_classes(root_folder):
	tqdm.write("Preparing class list ...")

	class_list = os.listdir(root_folder)
	tqdm.write("Done!")
	tqdm.write(" ")
	return class_list

def prepare_split(split_name, root_folder, class_list):
	tqdm.write(f"Preparing {split_name} split ...")

	image_paths = []
	labels = []

	for class_name in tqdm(class_list):
		subdir = os.path.join(root_folder, class_name)
		
		for fname in os.listdir(subdir):
			image_paths.append(os.path.join(subdir, fname))
			labels.append(class_list.index(class_name))

	tqdm.write("Done!")
	tqdm.write(" ")
	return {"image_paths": image_paths, "labels": labels}

def run_preprocess():
	tqdm.write("Starting preprocessing of Caltech 101 dataset ...")

	config = data_configs.datasets["caltech_101"]

	class_list = prepare_classes(config.root_folder)
	test_split = prepare_split("test", config.root_folder, class_list)

	preprocessed_data = {"class_list": class_list, "test": test_split}
	torch.save(preprocessed_data, config.preprocessed_data_path)
	tqdm.write("Data preprocessing done!")


if __name__ == "__main__":
	run_preprocess()