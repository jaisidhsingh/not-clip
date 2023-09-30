import torch
from tqdm import tqdm
import os
import sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

from configs.data_configs import data_configs


def prepare_classes(class_list_path):
	tqdm.write("Preparing class list ...")

	num_classes = 37
	class_list = []
	current_class_id = 1

	with open(class_list_path) as f:
		for line in tqdm(f.readlines()):
			line = line.strip()
			
			if line[0] == "#":
				continue
			else:
				entry = line.split(" ")
				class_id = int(entry[1])
				
				if current_class_id == class_id:
					current_class_id += 1

					class_name = "_".join(entry[0].split("_")[:-1])
					class_list.append(class_name)

				else:
					continue

	tqdm.write("Done!")
	tqdm.write(" ")
	return class_list

def prepare_split(split_name, split_info_path, class_list, image_folder, extension=".jpg"):
	tqdm.write(f"Preparing {split_name} split ...")

	image_paths = []
	labels = []

	with open(split_info_path) as f:
		for line in tqdm(f.readlines()):
			line = line.strip()
			entry = line.split(" ")

			label = int(entry[1]) - 1
			path = os.path.join(image_folder, entry[0] + extension)

			labels.append(label)
			image_paths.append(path)

	tqdm.write("Done!")
	tqdm.write(" ")	
	return {"image_paths": image_paths, "labels": labels}

def run_preprocess():
	tqdm.write("Starting preprocessing of Oxford Pets dataset ...")
	
	config = data_configs.datasets["oxford_pets"]
	
	class_list = prepare_classes(config.class_list_path)
	train_split = prepare_split("train", config.split_info_paths["train"], class_list, config.image_folder)
	test_split = prepare_split("test", config.split_info_paths["test"], class_list, config.image_folder)

	preprocessed_data = {
		"class_list": class_list,
		"train": train_split,
		"test": test_split
	}
	torch.save(preprocessed_data, config.preprocessed_data_path)
	tqdm.write("Data preprocessing done!")


if __name__ == "__main__":
	run_preprocess()