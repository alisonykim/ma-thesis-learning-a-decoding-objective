#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# learned_logits_warper.py

"""
Transform token scores using learned weights.
"""

import torch
from transformers import LogitsWarper

import attributes


class LearnedLogitsWarper(LogitsWarper):
	"""
	[`LogitsWarper`] that upweights token scores using a learned mapping from LM scores to a one-hot-encoded corpus of human-generated texts.
	
	Args:
		learned_weights_path: Path to saved linear weights, i.e. the learned mapping from token attributes to one-hot encoding of human-generated (label) texts. Size is 4 x `|V|`.
		unigram_freqs_path: Path to saved corpus unigram frequencies.
		k: Number of top probability values to consider
	
	Attributes:
		device: Device onto which to load LM and tensors
		model: Learned transformation with which to modify LM scores
		unigram_freqs: Corpus unigram frequencies
		k: Number of top probability values to consider
		w: Linear weight
		b: Linear bias
	"""
	
	def __init__(self, learned_weights_path: str, unigram_freqs_path: str, top_k: int=100):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model = torch.load(learned_weights_path, map_location=torch.device(self.device))
		self.unigram_freqs = self._get_corpus_unigram_freq(unigram_freqs_path)
		self.k = top_k
		
		self.w = self.model['linear.weight'] # 1 x 4
		self.b = self.model['linear.bias'] # 1

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
		attributes = self._calc_features(scores) # 4 x |V|
		upweighted_scores = torch.matmul(self.w, attributes) + self.b # (1 x 4) x (4 x |V|)
		return upweighted_scores # 1 x |V|
	
	def _calc_features(self, scores: torch.FloatTensor) -> torch.FloatTensor:
		"""Calculate features tensor."""
		probs = attributes.calc_probs(scores)
		nlls = attributes.calc_nlls(scores)
		entropies = attributes.calc_entropies(probs, nlls)
		diff = attributes.calc_diff_nlls_ents(nlls, entropies)
		abs_diff = torch.abs(diff)
		unigram_freqs = self.unigram_freqs
		top_k = attributes.is_top_k(probs, self.k, self.device)
		try:
			features = torch.stack([probs, top_k, unigram_freqs, abs_diff], dim=-1)
		except RuntimeError:
			features = torch.stack([probs, top_k, unigram_freqs.unsqueeze(dim=0), abs_diff], dim=-1)
		return torch.transpose(features.squeeze(dim=0), dim0=0, dim1=1)
	
	def _get_corpus_unigram_freq(self, unigram_freqs_path: str):
		"""Load unigram frequency tensor."""
		return torch.load(unigram_freqs_path, map_location=torch.device(self.device))