import os
import sys

src_dir = os.path.abspath("../")
sys.path.append(src_dir)

from tqdm import tqdm
import clip
from data import PretrainDataset
import torch
from torch.utils.data import DataLoader
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import warnings
warnings.simplefilter("ignore")


# model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# top_k = 3

"""
IN-CONTEXT LEARNING
"""
pre_text = "In this game, you have to understand the scene represented by a prompt. Then, based on your understanding, choose a real-world concept which is absent from the scene and re-write the prompt such that the concept is not there. Solve the following after looking at an example: "
example = "'a xmas tree in a living room beside a sofa' -> 'a xmas tree in a room which does not have snow'. Now what is " 

def get_negative_prompts(caption, top_k, model, tokenizer, pre_text):
	text = f"'{caption}' -> ? Give your {top_k} best answers separated by commas."
	all_text = pre_text + example + text
	
	inputs = tokenizer(all_text, return_tensors="pt").to(0)

	out = model.generate(**inputs)
	answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

	answer = answer.replace("\n", "")
	results = answer.split(f"'{caption}' -> ")[-1].split(",")
	results = [item.replace("'", "") for item in results]
	return results[:top_k]

def diversify_negative_prompts(negative_prompts):
	check = 0
	for prompt in negative_prompts:
		if "without" in prompt:
			check += 1
	
	replacements = ["not containing", "which does not have", "without"]
	if check > (2/3 * len(negative_prompts)):
		for i in range(len(negative_prompts)):
			idx = i % len(replacements)
			negative_prompts[i] = negative_prompts[i].replace("without", replacements[idx])
	
	return negative_prompts

def test():
	caption = "a row of chairs at the edge of a pool on a sunny day"

	top_k = 3
	results = get_negative_prompts(caption, top_k, model, tokenizer, pre_text)
	results = diversify_negative_prompts(results)

	print(f"For the input: '{caption}' we get the following negative prompts:")
	print(results)

def encode_captions(captions):
	all_text = [pre_text + example + f"'{caption}' -> ? Give your {top_k} best answers separated by commas." for caption in captions]
	inputs = tokenizer(all_text, padding=True, return_tensors="pt").to(0)

	out = model.generate(**inputs)
	out = tokenizer.batch_decode(out, skip_special_tokens=True)
	answers = [out[i].strip() for i in range(len(out))]

	results_store = []
	for i, answer in enumerate(answers):
		answer = answer.replace("\n", "")
		results = answer.split(f"'{captions[i]}' -> ")[-1].split(",")
		results = [item.replace("'", "") for item in results]
		results = diversify_negative_prompts(results[:top_k])
		results_store.append(results)
	
	return results_store

def main():
	clip_model, preprocess = clip.load("ViT-L/14", device="cpu")

	dataset = PretrainDataset("cc3m", "test", preprocess)
	# loader = DataLoader(dataset, batch_size=4, pin_memory=True)
	results_store = {}

	print("Dataset loaded.")

	for i in range(100):
		print(dataset[i][1])	

	# bar = tqdm(total=len(loader))
	# for _, captions in loader:
	# 	results = encode_captions(captions)
		
	# 	for i in range(len(results)):
	# 		results_store[captions[i]] = results[i]

	# 	bar.update(1)
	# 	del results

	# torch.save(results_store, "../../results/negative_prompting_results_cc3m.pt")

if __name__ == "__main__":
	main()