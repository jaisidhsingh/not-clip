import os
import sys

src_dir = os.path.abspath("../")
sys.path.append(src_dir)

import torch
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from data import PretrainDataset
from loss_functions import SemanticContrastLoss
import loraclip
from loraclip.loralib import utils as lora_utils

from augment_clip import augment_clip


def prepare_model(args):
	if args.training_method == "loraclip":
		model, preprocess = loraclip.load(args.clip_model_name, device=args.device, r=args.lora_rank)
		loraclip.print_trainable_parameters(model)

	elif args.training_method == "selective_freeze_with_mlp":
		assert False, "Not implemented"
		model, preprocess = None, None
	
	elif args.training_method == "selective_freeze":
		# modify here
		model, preprocess = clip.load(args.clip_model_name, device=args.device)
		retrain_words = ['not', 'without', 'no']
		augment_clip(args, model, retrain_words, disable_coop_cocoop=True) #! disable coop cocoop because otherwise state dict matching MIGHT not happen with the added coop cocoop parameters.

	model.train()
	return model, preprocess

def main(args):
	model, preprocess = prepare_model(args)
	assert model is not None, "Not implemented"
	assert preprocess is not None, "Not implemented"

	dataset = PretrainDataset(args.split, transform=preprocess)
	loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=args.pin_memory)

	# training_parameters = [p for p in model.parameters() if p.requires_grad]

	# optimizer = torch.optim.Adam(
	# 	training_parameters, 
	# 	lr=args.lr,
	# 	betas=(args.beta1, args.beta2),
	# 	eps=args.eps,
	# 	weight_decay=args.weight_decay
	# )
	# criterion = SemanticContrastLoss()


	# for epoch in range(args.num_epochs):
	# 	bar = tqdm(total=len(loader))

	# 	for (images, p, n1, n2, cr) in loader:
	# 		# cast to cuda
	# 		images, p, n1, n2, cr = images.to(args.device), p.to(args.device), n1.to(args.device), n2.to(args.device), cr.to(args.device)
	# 		B = images.shape[0]
	# 		K = n1.shape[1]

	# 		p = p.squeeze(1)
	# 		n2 = n2.squeeze(1)

	# 		n1 = n1.view(B * K, n1.shape[-1])
	# 		cr = cr.view(B * K, cr.shape[-1])

	# 		# zero grads
	# 		optimizer.zero_grad()

	# 		# forward pass
	# 		image_features = model.encode_image(images)
	# 		p_features = model.encode_text(p)
	# 		n1_features = model.encode_text(n1)
	# 		n2_features = model.encode_text(n2)
	# 		cr_features = model.encode_text(cr)

	# 		loss = criterion(
	# 			image_features, p_features,
	# 			n1_features, n2_features, cr_features
	# 		)

	# 		running_loss = loss.item()

	# 		loss.backward()
	# 		# write a zero_frozen_token_grads_with_check() function that fails if some condition is not met.
	# 		try:
	# 			# model.zero_frozen_token_grads()
	# 			model.zero_frozen_token_grads_with_check() #! just do this to make sure that everything is fine
	# 													   #! replace with the line above it in actual training
	# 		except:
	# 			pass
	# 		optimizer.step()

	# 		bar.set_postfix({"epoch": epoch+1, "loss": running_loss})
	# 		bar.update(1)
		
	# 	if (epoch + 1) % args.save_point == 0:
	# 		if args.training_method == "loraclip":
	# 			dump = {
	# 				"model": model.state_dict(), 
	# 				"optimizer": optimizer.state_dict(), 
	# 				"lora": lora_utils.lora_state_dict(model)
	# 		}
	# 		save_name = f"model_lora_{epoch+1}.pt"
	# 	else:
	# 		dump = {
	# 			"model": model.state_dict(), 
	# 			"optimizer": optimizer.state_dict(), 
	# 		}
	# 		save_name = f"model_{epoch+1}.pt"

	# 	save_dir = f"../../checkpoints/{args.training_method}"
	# 	os.makedirs(save_dirs, exist_ok=True)
	# 	save_path = os.path.join(save_dir, save_name)

	# 	torch.save(dump, save_path)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--clip-model-name", type=str, default="ViT-B/16")
	parser.add_argument("--batch-size", type=int, default=48)
	parser.add_argument("--pin-memory", type=bool, default=True)
	parser.add_argument("--lora-rank", type=int, default=4)
	parser.add_argument("--num-epochs", type=int, default=5)
	parser.add_argument("--split", type=str, default="train")

	# optimizer hyperparameters
	parser.add_argument("--lr", type=float, default=5e-5)
	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.98)
	parser.add_argument("--eps", type=float, default=1e-6)
	parser.add_argument("--weight-decay", type=float, default=0.2)

	parser.add_argument("--save-point", type=int, default=1)
	parser.add_argument("--training-method", type=str, default="loraclip")
	args = parser.parse_args()

	return args


if __name__ == "__main__":
	args = get_args()
	main(args)
