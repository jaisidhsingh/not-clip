import torch
from tqdm import tqdm
import os
import sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

from configs.data_configs import data_configs


def prepare_classes(class_list_path=None):
	index2class = {
		0: 'pink primrose',
		1: 'hard-leaved pocket orchid',
		2: 'canterbury bells',
		3: 'sweet pea',
		4: 'english marigold',
		5: 'tiger lily',
		6: 'moon orchid',
		7: 'bird of paradise',
		8: 'monkshood',
		9: 'globe thistle',
		10: 'snapdragon',
		11: "colt's foot",
		12: 'king protea',
		13: 'spear thistle',
		14: 'yellow iris',
		15: 'globe-flower',
		16: 'purple coneflower',
		17: 'peruvian lily',
		18: 'balloon flower',
		19: 'giant white arum lily',
		20: 'fire lily',
		21: 'pincushion flower',
		22: 'fritillary',
		23: 'red ginger',
		24: 'grape hyacinth',
		25: 'corn poppy',
		26: 'prince of wales feathers',
		27: 'stemless gentian',
		28: 'artichoke',
		29: 'sweet william',
		30: 'carnation',
		31: 'garden phlox',
		32: 'love in the mist',
		33: 'mexican aster',
		34: 'alpine sea holly',
		35: 'ruby-lipped cattleya',
		36: 'cape flower',
		37: 'great masterwort',
		38: 'siam tulip',
		39: 'lenten rose',
		40: 'barbeton daisy',
		41: 'daffodil',
		42: 'sword lily',
		43: 'poinsettia',
		44: 'bolero deep blue',
		45: 'wallflower',
		46: 'marigold',
		47: 'buttercup',
		48: 'oxeye daisy',
		49: 'common dandelion',
		50: 'petunia',
		51: 'wild pansy',
		52: 'primula',
		53: 'sunflower',
		54: 'pelargonium',
		55: 'bishop of llandaff',
		56: 'gaura',
		57: 'geranium',
		58: 'orange dahlia',
		59: 'pink-yellow dahlia',
		60: 'cautleya spicata',
		61: 'japanese anemone',
		62: 'black-eyed susan',
		63: 'silverbush',
		64: 'californian poppy',
		65: 'osteospermum',
		66: 'spring crocus',
		67: 'bearded iris',
		68: 'windflower',
		69: 'tree poppy',
		70: 'gazania',
		71: 'azalea',
		72: 'water lily',
		73: 'rose',
		74: 'thorn apple',
		75: 'morning glory',
		76: 'passion flower',
		77: 'lotus',
		78: 'toad lily',
		79: 'anthurium',
		80: 'frangipani',
		81: 'clematis',
		82: 'hibiscus',
		83: 'columbine',
		84: 'desert-rose',
		85: 'tree mallow',
		86: 'magnolia',
		87: 'cyclamen ',
		88: 'watercress',
		89: 'canna lily',
		90: 'hippeastrum ',
		91: 'bee balm',
		92: 'ball moss',
		93: 'foxglove',
		94: 'bougainvillea',
		95: 'camellia',
		96: 'mallow',
		97: 'mexican petunia',
		98: 'bromelia',
		99: 'blanket flower',
		100: 'trumpet creeper',
		101: 'blackberry lily'
 	}

	class_list = list(index2class.values())
	return class_list


def prepare_split(split_name, split_info_path, class_list, image_folder, extension=".jpg"):
	tqdm.write(f"Preparing {split_name} split ...")

	image_paths = []
	labels = []

	with open(split_info_path) as f:
		for line in tqdm(f.readlines()):
			line = line.strip()
			entry = line.split(" ")

			label = int(entry[1])
			path = os.path.join(image_folder, entry[0][4:])

			labels.append(label)
			image_paths.append(path)

	tqdm.write("Done!")
	tqdm.write(" ")	
	return {"image_paths": image_paths, "labels": labels}

def run_preprocess():
	tqdm.write("Starting preprocessing of Flowers 102 dataset ...")
	
	config = data_configs.datasets["flowers_102"]
	
	class_list = prepare_classes()
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