import os
import sys

src_dir = os.path.abspath("../")
sys.path.append(src_dir)

import torch
from torch.utils.data import DataLoader
from data import PretrainDataset
from training.augment_clip import augment_clip
import loraclip
from loraclip.loralib import utils as lora_utils
import clip
from tqdm import tqdm
import argparse


def prepare_models(args):
	if args.training_method == "selective_freeze":
		model, preprocess = clip.load(args.clip_model_name, device=args.device)
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
def main(args):
	default_model, preprocess = clip.load(args.clip_model_name, device=args.device)
	trained_model, preprocess = prepare_models(args)

	trained_model.eval()
	default_model.eval()

	dataset = PretrainDataset(args.split, transform=preprocess)
	loader = DataLoader(dataset, batch_size=1, pin_memory=args.pin_memory)

	bar = tqdm(total=len(loader))
	default_errors, trained_errors = 0, 0
	default_corrects, trained_corrects = 0, 0
	total = 0

	for (images, p, n1, n2, cr) in loader:
		images = images.to(args.device)
		n1 = n1.to(args.device)

		if 1 == n1.shape[0]:
			n1 = n1.squeeze(0)

		k = n1.shape[0]

		image_features = default_model.encode_image(images)

		n1_trained = trained_model.encode_text(n1)
		n1_default = default_model.encode_text(n1)

		trained_sim = (100 * image_features.float() @ n1_trained.float().T) # 1xk shape
		default_sim = (100 * image_features.float() @ n1_default.float().T) # 1xk shape

		sims = torch.cat([default_sim, trained_sim], dim=0) # 2xk shape
		preds = sims.argmin(dim=0) # k shape

		default_errors += preds.sum().item()
		trained_errors += len(preds) - default_errors

		default_corrects += len(preds) - preds.sum().item()
		trained_corrects += preds.sum().item()

		total += k
		bar.set_postfix({
			"default_accuracy": round(default_corrects / total, 4),
			"trained_accuracy": round(trained_corrects / total, 4)
		})
		bar.update(1)
	
	results = {}
	results["default_accuracy"] = round(default_corrects / total, 4)
	results["trained_accuracy"] = round(trained_corrects / total, 4)

	results["default_error_rate"] = round(default_errors / total, 4)
	results["trained_error_rate"] = round(trained_errors / total, 4)
	
	results["configuration"] = args

	for k, v in results.items():
		if k != "configuration":
			print(f"{k} --- {v}")

	model_name = args.clip_model_name.replace("/", "-")
	save_name = f"results_{model_name}_{args.training_method}.pt"
	save_dir = "../../results/pretraining_val"
	save_path = os.path.join(save_dir, save_name)

	torch.save(results, save_path)


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--clip-model-name", type=str, default="ViT-B/16")
	parser.add_argument("--batch-size", type=int, default=48)
	parser.add_argument("--pin-memory", type=bool, default=True)
	parser.add_argument("--lora-rank", type=int, default=4)
	parser.add_argument("--num-epochs", type=int, default=10)
	parser.add_argument("--split", type=str, default="val")

	# optimizer hyperparameters
	parser.add_argument("--lr", type=float, default=5e-5)
	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.98)
	parser.add_argument("--eps", type=float, default=1e-6)
	parser.add_argument("--weight-decay", type=float, default=0.2)

	parser.add_argument("--save-point", type=int, default=2)
	parser.add_argument("--training-method", type=str, default="loraclip")
	args = parser.parse_args()

	return args


if __name__ == "__main__":
	args = get_args()
	main(args)