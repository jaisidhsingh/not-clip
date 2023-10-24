import sys
from PIL import Image
from diffusers import DiffusionPipeline, LMSDiscreteScheduler, AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import argparse
import secrets
import string
import sys
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from tqdm.auto import tqdm
import clip
from types import SimpleNamespace


def init_clip():
	tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
	text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
	
	return tokenizer, text_encoder

def set_clip_embeddings(text_encoder, ckpt_path):
	clip_model, _ = clip.load("ViT-B/16", device="cpu")
	ckpt = torch.load(ckpt_path)["model"]

	clip_model.token_embedding.weight.data = ckpt["token_embedding.weight"]
	trained_embeddings = clip_model.token_embedding
	text_encoder.set_input_embeddings(trained_embeddings)
	print("CLIP embeddings loaded")

def make_name():
	random_string = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(6))
	return random_string

def pipe(args):
	pipeline = DiffusionPipeline.from_pretrained(
		"runwayml/stable-diffusion-v1-5", 
		use_safetensors=True
	)
	pipeline.to("cuda")

	name = make_name()
	prompt = args.prompt
	print(f"Making an image of {prompt}")

	image = pipeline(prompt).images[0]
	image.save(f"../../results/generation/{name}.png")

def custom(args, ckpt=False):
	cfg = SimpleNamespace(**{})
	cfg.run_name = "inpainting-content"                 # the name of the experiment run
	cfg.results_dir = "../inference"        # the directory to save results in
	cfg.height = 512                        # default height of Stable Diffusion
	cfg.width = 512                         # default width of Stable Diffusion
	cfg.num_inference_steps = 150           # Number of denoising steps (scheduler dependant)
	cfg.guidance_scale = 7.5                # Scale for classifier-free guidance
	cfg.generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
	cfg.batch_size = 1                      # The batch size controls the number of images you get
	cfg.device = "cuda"                     # Whether or not to use GPU (recommended)

	# load in the models and objects
	# image -> latent space
	vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(cfg.device)

	# get text embedding to condition the UNet
	tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
	text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(cfg.device)

	if ckpt:
		set_clip_embeddings(text_encoder, args.clip_embeddings_ckpt)

	# UNet for conditioned diffusion over image latents
	unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(cfg.device)

	scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
	
	text_input = tokenizer(
		args.prompt, 
		padding="max_length",
		max_length=tokenizer.model_max_length,
		truncation=True,
		return_tensors="pt"
	)

	# make the CLIP text embeddings for conditioning the latent
	with torch.no_grad():
		text_embeddings = text_encoder(text_input.input_ids.to(cfg.device))[0]
		
	max_length = text_input.input_ids.shape[-1]
	uncond_inputs = tokenizer(
		[""] * cfg.batch_size, 
		padding="max_length", 
		max_length=max_length, 
		return_tensors="pt"
	) 

	with torch.no_grad():
		uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(cfg.device))[0]   

	# use both conditioned and unconditioned embeddings, as we have classifier free-guidance
	text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

	# first make random noisy latents, and then denoise them, conditioned on the semantic information in the text embedding
	latents = torch.randn(
		(cfg.batch_size, unet.config.in_channels, cfg.height // 8, cfg.width // 8),
		generator=cfg.generator,
	)
	latents = latents.to(cfg.device)

	# set the timesteps for the backward process
	scheduler.set_timesteps(cfg.num_inference_steps)
	latents = latents * scheduler.init_noise_sigma

	# the backward process loop to get the image
	for t in tqdm(scheduler.timesteps):
		# expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
		latent_model_input = torch.cat([latents] * 2)

		latent_model_input = scheduler.scale_model_input(latent_model_input, t)

		# predict the noise residual
		with torch.no_grad():
			noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

		# perform guidance
		noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
		noise_pred = noise_pred_uncond + cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

		# compute the previous noisy sample x_t -> x_t-1
		latents = scheduler.step(noise_pred, t, latents).prev_sample

	# scale and decode the image latents with vae
	latents = 1 / 0.18215 * latents

	with torch.no_grad():
		image = vae.decode(latents).sample

	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
	images = (image * 255).round().astype("uint8")
	pil_images = [Image.fromarray(image) for image in images]
	
	name = make_name()
	pil_images[0].save(f"../../results/generation/{name}.png")
	print(name)


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--prompt", type=str, default="a car which does not have tires")
	parser.add_argument("--clip-embeddings-ckpt", type=str, 
		default="/workspace/clip-tests/not_problem/checkpoints/vitl14/notx1_1/notx1_prompting_clip_30.pt"
	)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = get_args()
	custom(args, ckpt=False)
