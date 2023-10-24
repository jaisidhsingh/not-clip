import webdataset as wds
import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

src_path = os.path.abspath("../")
sys.path.append(src_path)

from configs.data_configs import data_configs


def prepare_classes():
    config = data_configs.datasets["cc3m"]
    data = torch.load(config.preprocessed_data_path)

    class_list = []
    N = len(data["test"]["image_paths"])

    bar = tqdm(total=N)
    for i in range(N):
        annos = data["test"]["labels"][i]
        labels = annos["labels"]
        
        if type(labels) == float:
            print(i)
            sys.exit()
        
        labels_list = labels.split(",")

        for item in labels_list:
            if item not in class_list:
                class_list.append(item)

        bar.update(1)
        bar.set_postfix({"num_classes": len(class_list)})

    data["class_list"] = class_list
    torch.save(data, config.preprocessed_data_path)

def run_preprocess():
    config = data_configs.datasets["cc3m"]

    len_dataset = 302123
    base_path = f"{config.root_folder}"
    file_paths = [f"{base_path}/{str(i).zfill(5)}.tar" for i in range(34)]
    dataset = wds.WebDataset(file_paths).decode("rgb").to_tuple("__url__", "jpg", "json", "txt")
    helper = pd.read_csv(os.path.join(config.root_folder, '../cc3m_image_labels.tsv'), sep='	')
    helper = {
        url: {
            'caption': caption,
            'labels':  labels,
            'MIDs': MIDs,
            'confidence_scores': confidence_scores,
        } for _, (caption, url, labels, MIDs, confidence_scores) in helper.iterrows()
    }

    output_folder = '/workspace/datasets/cc3m/cc3m_subset_images_extracted_final'
    ptfile = {}
    ptfile['test'] = {}
    ptfile['test']['image_paths'] = []
    ptfile['test']['labels'] = []

    bar = tqdm(total=len_dataset)
    counts = [0, 0]
    for tar_path, image, json, caption in dataset:
        image_url = json['url']
        image_number = json['key']
        image_path_new = os.path.join(output_folder, f'{image_number}.jpg')

        if image_url not in helper:
            counts[0] += 1
            bar.update(1)
            bar.set_postfix({'count_in': counts[1], 'counts_out': counts[0]})
            continue
        else:
            caption, labels, MIDs, confidence_scores = helper[image_url].values()
            annotations = {
                'tar_path': tar_path,
                'image_number': image_number,
                'file_extension': 'jpg', # kind of unnecessary since we are only saving jpgs but okay
                'json': json,
                'caption': caption,
                'labels': labels,
                'MIDs': MIDs,
                'confidence_scores': confidence_scores,
            }

            if type(annotations["labels"]) == str:

                ptfile['test']['labels'].append(annotations)
                ptfile['test']['image_paths'].append(image_path_new)

                image_array_float = image.astype(np.float32)
                image_array_uint8 = (image_array_float * 255).astype(np.uint8)
                Image.fromarray(image_array_uint8).save(image_path_new)

                counts[1] += 1
            
            else:
                counts[0] += 1

            bar.update(1)
            bar.set_postfix({'count_in': counts[1], 'counts_out': counts[0]})
    bar.close()

    torch.save(ptfile, config.preprocessed_data_path)


if __name__ == "__main__":
    # run_preprocess()
    prepare_classes()