import torch
from datasets import load_dataset
import scipy.io
from tqdm import tqdm
import os
import sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

from configs.data_configs import data_configs


def prepare_classes(class_list_path):
	tqdm.write("Preparing class list ...")

	class_list = scipy.io.loadmat(class_list_path)["class_names"].tolist()[0]
	class_list = [class_name.item() for class_name in class_list]
	print(class_list)	
	tqdm.write("Done!")
	tqdm.write(" ")
	return class_list

def prepare_split(split_name, image_folder, extension=".jpg"):
	tqdm.write(f"Preparing {split_name} split ...")

	image_paths = []
	labels = []

	image_names = os.listdir(image_folder)
	image_names.sort()
	split_info = load_dataset(f"Multimodal-Fatima/StanfordCars_{split_name}")

	for i in tqdm(range(len(split_info[split_name]))):
		data = split_info[split_name][i]

		label = int(data["label"])
		path = os.path.join(image_folder, image_names[int(data["id"])])

		labels.append(label)
		image_paths.append(path)

	tqdm.write("Done!")
	tqdm.write(" ")	
	return {"image_paths": image_paths, "labels": labels}

def run_preprocess():
	tqdm.write("Starting preprocessing of Stanford Cars dataset ...")
	
	config = data_configs.datasets["stanford_cars"]
	
	class_list = prepare_classes(config.class_list_path)
	train_split = prepare_split("train", config.train_image_folder)
	test_split = prepare_split("test", config.test_image_folder)

	preprocessed_data = {
		"class_list": class_list,
		"train": train_split,
		"test": test_split
	}
	torch.save(preprocessed_data, config.preprocessed_data_path)
	tqdm.write("Data preprocessing done!")


if __name__ == "__main__":
	run_preprocess()
