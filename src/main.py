import torch
import clip
import sys

from configs.data_configs import data_configs

config = data_configs.datasets['cc3m']
data = torch.load(config.preprocessed_data_path)
classes = [item.lower().replace("_", " ") for item in data["class_list"]]
print(f'splits: {data.keys()}')
# split_data = data[split]