import torch
import torch.nn as nn
import einops


class SemanticContrastLoss():
	def __init__(self, reduction="mean"):
		self.num_terms = 5
		self.reduction = reduction

	def reduce_self(self, loss):
		if self.reduction == "mean":
			return loss.mean(dim=0).mean(dim=0)
		
		if self.reduction == "sum":
			return loss.mean(dim=0).sum(dim=0)

	def __call__(self, i, p, n1, n2, cr1):
		"""
		formulation: L = s(i, n1) + s(p, n1) + s(n1, cr1) - s(n1, n2)
		=============================================================

		dimensions:
		------------
		i:    B x D
		n1:   B x D
		n2:   B x D
		p:    B x D
		cr1:  B x D
		L:    B x B
		
		# i and n1: (B, D) x (B*K, D) -> (B, B*K) -> (B, B, K).sum(dim=-1) -> (B, B)
		# same for p and n1
		# n1 and cr1: (B*K, D) x (B*K, D) -> (B*K, B*K) -> (B, K, K, B) -> sum till (B, B)
		# n1 and n2: (B*K, D) x (B, D) -> (B, K, B) -> sum till (B, B)

		# sim_to_minimize = (i @ n1.T) + (p @ n1.T) + (n1 @ cr1.T)
		# sim_to_maximize = n1 @ n2.T
		"""

		device = i.device

		b = i.shape[0]
		k = n1.shape[0] // b
		d = i.shape[-1]

		i = i/ i.norm(dim=-1, keepdim=True)
		p = p / p.norm(dim=-1, keepdim=True)
		n1 = n1 / n1.norm(dim=-1, keepdim=True)
		n2 = n2 / n2.norm(dim=-1, keepdim=True)
		cr1 = cr1 / cr1.norm(dim=-1, keepdim=True)

		i_and_n1 = (i @ n1.T).view(b, b, k).sum(dim=-1)
		p_and_n1 = (p @ n1.T).view(b, b, k).sum(dim=-1)

		mini_mask = 2*torch.eye(k) - 1
		repeated_mask = einops.repeat(mini_mask, 'm n -> b m n c', b=b, c=b)
		repeated_mask = repeated_mask.to(device)
	
		n1_and_cr1 = (n1 @ cr1.T).view(b, k, k, b) # !flag
		masked_n1_and_cr1 = (repeated_mask * n1_and_cr1).sum(dim=1).sum(dim=1)
		masked_n1_and_cr1 = masked_n1_and_cr1.view(b, b)

		n1_and_n2 = (n1 @ n2.T).view(b, k, b).sum(dim=1)

		sim_to_minimize = i_and_n1 + p_and_n1 + masked_n1_and_cr1
		sim_to_maximize = n1_and_n2

		loss = sim_to_minimize - sim_to_maximize
		return self.reduce_self(loss)

def test():
	b = 4
	k = 3
	d = 768

	i = torch.randn(b, d)
	p = torch.randn(b, d)
	n1 = torch.randn(b*k, d)
	n2 = torch.randn(b, d)
	cr = torch.randn(b*k, d)

	criterion = SemanticContrastLoss()
	loss = criterion(i, p, n1, n2, cr)

	print(loss.item())


if __name__ == "__main__":
	test()