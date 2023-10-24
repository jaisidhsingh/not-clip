import torch
import clip
from collections import OrderedDict

def add_trainable_param_coop(args, model):
    # coop: trainable parameter
    ctx_vecs = torch.empty(args.n_ctx, model.transformer.d_model)
    torch.nn.init.normal_(ctx_vecs, std=0.02)  # from the paper
    trainable_param = torch.nn.Parameter(ctx_vecs)
    model.register_parameter('trainable_param', trainable_param)

def add_meta_net_cocoop(args, model):
    # cocoop: meta net
    ctx_dim = model.ln_final.weight.shape[0]
    vis_dim = model.visual.output_dim
    model.meta_net = torch.nn.Sequential(OrderedDict([
        ("linear1", torch.nn.Linear(vis_dim, vis_dim // 16)),
        ("relu", torch.nn.ReLU(inplace=True)),
        ("linear2", torch.nn.Linear(vis_dim // 16, ctx_dim))
    ]))
    model.meta_net = model.meta_net.half()  # This is okay if you want to use reduced precision (half precision).

def define_coop_encoder(args, model):
    def encode_text_coop(text):
        x = model.token_embedding(text).to(model.dtype)  # [batch_size, n_ctx, d_model]

        # ishaan: line added here for the replacement using the extra parameter stored inside the model
        context_embedding = model.trainable_param.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x[:, 1:1 + context_embedding.shape[1], :] = context_embedding  # tokenizer outputs: <start> <word1> <word2> <word3> ... <eos> <0> <0> ... 77 tokens in total

        x = x + model.positional_embedding.to(model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x).to(model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

        return x

    model.encode_text_coop = encode_text_coop

def define_cocoop_encoder(args, model):
    def encode_text_cocoop(text, image_features_normed):
        bx = model.token_embedding(text).to(model.dtype)  # [num_classes, n_tokens, d_model]
        bx = bx.unsqueeze(0).repeat(image_features_normed.shape[0], 1, 1, 1)  # [batch_size, num_classes, n_tokens, d_model] or BNLD

        # ishaan: line added here for the replacement using the extra parameter stored inside the model
        context_embedding = model.trainable_param  # [n_ctx, d_model]
        meta_embedding = model.meta_net(image_features_normed)  # [batch_size, d_model]
        context_embedding = context_embedding.unsqueeze(0)  # [1, n_ctx, d_model]
        meta_embedding = meta_embedding.unsqueeze(1)  # [batch_size, 1, d_model]
        net_embedding = context_embedding + meta_embedding  # [batch_size, n_ctx, d_model]
        net_embedding = net_embedding.unsqueeze(1).repeat(1, bx.shape[1], 1, 1)  # [batch_size, num_classes, n_ctx, d_model] or BNnD

        bx[:, :, 1:1 + context_embedding.shape[1], :] = net_embedding  # tokenizer outputs: <start> <word1> <word2> <word3> ... <eos> <0> <0> ... 77 tokens in total

        text_features = []
        for x in bx:
            x = x + model.positional_embedding.to(model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = model.ln_final(x).to(model.dtype)

            # x.shape = [num_classes, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

            text_features.append(x)
        text_features = torch.stack(text_features)

        return text_features

    model.encode_text_cocoop = encode_text_cocoop

def get_frozen_token_ids(args, model, words):
    token_ids = []
    for word in words:
        # get all the token ids for the words in the word list.
        # then check whether the EOT token is at index==2 or not. if not, raise assert - this word is not recognized as a word.
        # then check whether
        tokens = clip.tokenize([word])
        if tokens.argmax(dim=1).tolist()[0] == 2:
            token_ids.append(int(tokens[0][1]))
        else:
            assert False, f"one of the provided words: '{word}' is not recognized as a word by the clip tokenizer."

    frozen_token_ids = [id for id in range(model.token_embedding.weight.shape[0]) if id not in token_ids]
    return frozen_token_ids, token_ids

def zero_frozen_token_grads(model, frozen_token_ids): # call this every time right before optimizer.step() and you are done!
    list(model.token_embedding.parameters())[0].grad[frozen_token_ids] = list(model.token_embedding.parameters())[0].grad[frozen_token_ids] * 0

def zero_frozen_token_grads_with_check(model, frozen_token_ids, token_ids):
    zero_frozen_token_grads(model, frozen_token_ids)
    assert bool(torch.sum(list(model.token_embedding.parameters())[0].grad[frozen_token_ids]) == 0), "frozen token grads are not zero."
    assert bool(torch.sum(list(model.token_embedding.parameters())[0].grad[token_ids]) != 0), "token grads are zero."


def reinit_warm_token_embeddings(model, token_ids):
    for token_id in token_ids:
        torch.nn.init.normal_(model.token_embedding.weight[token_id], std=0.02)

def augment_clip(args, model, words, randomize_warm_token_embeddings=False, disable_coop_cocoop=False):
	if not disable_coop_cocoop:
		# augment CLIP model
		add_trainable_param_coop(args, model)
		add_meta_net_cocoop(args, model)

		# define the modified text encode methods
		define_coop_encoder(args, model)
		define_cocoop_encoder(args, model)
		model = model.to(args.device)

	# assign the embedding token ids not to retrain
	frozen_token_ids, token_ids = get_frozen_token_ids(args, model, words)

	# reinit the embeddings that will be trained
	if randomize_warm_token_embeddings:
		reinit_warm_token_embeddings(model, token_ids)

	model.zero_frozen_token_grads = lambda: zero_frozen_token_grads(model, frozen_token_ids)
	model.zero_frozen_token_grads_with_check = lambda: zero_frozen_token_grads_with_check(model, frozen_token_ids, token_ids)