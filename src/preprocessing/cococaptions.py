from collections import defaultdict
import json
import numpy as np
import torchvision.datasets as dsets
from tqdm import tqdm
import torch
import os
from PIL import Image


def test():
    split = "val"
    root_folder = f"/workspace/datasets/coco/coco_{split}2017/{split}2017"
    coco_dataset = dsets.CocoCaptions(
        root=root_folder,
        annFile=f"/workspace/datasets/coco/coco_ann2017/annotations/captions_{split}2017.json"
    )

    results = {"image_paths": [], "captions": []}
    N = 5000

    for i in tqdm(range(N)):
        id = coco_dataset.ids[i]
        image_name = coco_dataset.coco.loadImgs(id)[0]["file_name"]
        image_path = os.path.join(root_folder, image_name)

        annotations = coco_dataset.coco.loadAnns(coco_dataset.coco.getAnnIds(id))
        captions = [ann["caption"] for ann in annotations]

        # image, captions = coco_dataset[i]
        # selected_caption = max(captions, key=lambda x: len(x))


        if i in [5, ]:
            print(captions)
            print(image_path)
            for item in captions:
                print(item, len(item))

        results["image_paths"].append(image_path)
        results["captions"].append(captions)
    
    torch.save(results, f"/workspace/datasets/coco/coco_helper_{split}_image_captions.pt")

test()