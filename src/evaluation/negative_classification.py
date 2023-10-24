import os
import sys

src_dir = os.path.abspath("../")
sys.path.append(src_dir)

import torch
from torch.utils.data import DataLoader
from data import ClassificationDataset
from configs.data_configs import data_configs
# from training.augment_clip import augment_clip
import loraclip
from loraclip.loralib import utils as lora_utils
import clip
from tqdm import tqdm
import argparse


def prepare_models(args):
	if args.training_method == "selective_freeze":
		model, preprocess = clip.load(args.clip_model_name, device=args.device)
		# words_to_retrain = []
		# model = augment_clip(args, model, words_to_retrain, disable_coop_cocoop=True)
		model.load_state_dict(torch.load('../../checkpoints/selective_freeze/model_lora_10.pt')["model"], strict=False)
		print("Model loaded")

	if args.training_method == "loraclip":
		model, preprocess = loraclip.load(args.clip_model_name, device=args.device, r=args.lora_rank)
		# load in lora checkpoint
		# --------------------------------------------------------------------------------------------------
		# Load the pretrained checkpoint first
		model.load_state_dict(torch.load('../../checkpoints/loraclip/model_lora_6.pt')["model"], strict=False)
		# Then load the LoRA checkpoint
		model.load_state_dict(torch.load('../../checkpoints/loraclip/model_lora_6.pt')["lora"], strict=False)
	
	return model, preprocess

@torch.no_grad()
def eval(args):
	default_model, preprocess = clip.load(args.clip_model_name, device=args.device)
	trained_model, preprocess = prepare_models(args)

	default_model.eval()
	trained_model.eval()

	dataset = ClassificationDataset(args.dataset_name, args.split, transform=preprocess)
	loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=args.pin_memory)

	prompts = [f"this is not a photo of a {c}" for c in dataset.classes]
	prompts_tokenized = clip.tokenize(prompts).to(args.device)

	default_prompt_features = default_model.encode_text(prompts_tokenized)
	trained_prompt_features = trained_model.encode_text(prompts_tokenized)

	default_prompt_features = default_prompt_features.to(args.device)
	trained_prompt_features = trained_prompt_features.to(args.device)

	bar = tqdm(total=len(loader))
	default_matches, default_mismatches, = 0, 0
	trained_matches, trained_mismatches = 0, 0
	total = 0

	for (images, labels) in loader:
		images = images.squeeze(1).to(args.device)
		labels = labels.long().to(args.device)

		image_features = default_model.encode_image(images)
		image_features = image_features.to(args.device)

		default_sim = (100 * image_features @ default_prompt_features.T).softmax(dim=-1)
		default_preds = default_sim.argmax(dim=-1)

		trained_sim = (100 * image_features.float() @ trained_prompt_features.float().T).softmax(dim=-1)
		trained_preds = trained_sim.argmax(dim=-1)

		default_matches += (default_preds == labels).sum().item()
		default_mismatches += (default_preds != labels).sum().item()

		trained_matches += (trained_preds == labels).sum().item()
		trained_mismatches += (trained_preds != labels).sum().item()

		total += labels.shape[0]
		bar.set_postfix({
			"default_mismatch_accuracy": round(default_mismatches / total, 4),
			"trained_mismatch_accuracy": round(trained_mismatches / total, 4)
		})
		bar.update(1)
	
	results = {}
	results["default_match_accuracy"] = round(default_matches / total, 4)
	results["default_mismatch_accuracy"] = round(default_mismatches / total, 4)
	
	results["trained_match_accuracy"] = round(trained_matches / total, 4)
	results["trained_mismatch_accuracy"] = round(trained_mismatches / total, 4)
	results["configuration"] = args

	for k, v in results.items():
		if k != "configuration":
			print(f"{k} --- {v}")

	model_name = args.clip_model_name.replace("/", "-")
	save_name = f"results_{args.dataset_name}_{model_name}_{args.training_method}.pt"
	save_dir =  "/workspace/btp/results/negative_classification"
	save_path = os.path.join(save_dir, save_name)
	print(save_path)

	torch.save(results, save_path)


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--clip-model-name", type=str, default="ViT-B/16")
	parser.add_argument("--dataset-name", type=str, default="food_101")
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--pin-memory", type=bool, default=True)
	parser.add_argument("--lora-rank", type=int, default=4)
	parser.add_argument("--num-epochs", type=int, default=10)
	parser.add_argument("--split", type=str, default="test")

	# optimizer hyperparameters
	parser.add_argument("--lr", type=float, default=5e-5)
	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.98)
	parser.add_argument("--eps", type=float, default=1e-6)
	parser.add_argument("--weight-decay", type=float, default=0.2)

	parser.add_argument("--save-point", type=int, default=2)
	parser.add_argument("--training-method", type=str, default="selective_freeze")
	parser.add_argument("--make-report", type=str, default="no")
	args = parser.parse_args()

	return args

def make_results_file(args):
	datasets_list = list(data_configs.datasets.keys())
	datasets_list = datasets_list[:-3]
	save_dir =  "/workspace/btp/results/negative_classification"
	
	with open(f"../../results/negative_classification_results_{args.training_method}.txt", "w") as f:
		for dataset_name in datasets_list:
			model_name = args.clip_model_name.replace("/", "-")
			save_name = f"results_{dataset_name}_{model_name}_{args.training_method}.pt"
			save_path = os.path.join(save_dir, save_name)

			results = torch.load(save_path)
			
			f.write(f"Dataset: {dataset_name} \n")
			f.write("-------------------------------------------------------------------\n")

			f.write(f"default_match_accuracy --- {results['default_match_accuracy']}\n")
			f.write(f"default_mismatch_accuracy --- {results['default_mismatch_accuracy']}\n")
			f.write("   \n")
			f.write(f"trained_match_accuracy --- {results['trained_match_accuracy']}\n")
			f.write(f"trained_mismatch_accuracy --- {results['trained_mismatch_accuracy']}\n")

			f.write("   \n")
			f.write("   \n")

def main(args):
	if args.make_report == "yes":
		make_results_file(args)
		return
	
	else:
		eval(args)
		return

if __name__ == "__main__":
	args = get_args()
	main(args)





