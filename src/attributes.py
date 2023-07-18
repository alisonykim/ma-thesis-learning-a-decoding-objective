#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# attributes.py

"""
Tokenize texts, run inputs and labels through LM, calculate various token attributes.
"""

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, PreTrainedModel


def tokenize(
	texts: List[str],
	tokenizer: PreTrainedTokenizerFast,
	device: str,
	max_seq_length: Union[int, None]=None,
	truncation: bool=True,
	padding: Union[str, bool]='max_length'
) -> torch.LongTensor:
	"""
	Tokenize texts in corpus.
	
	Args:
		texts: List of texts
		tokenizer: Tokenizer to map tokens to LM vocabulary indices
		device: Device on which to load tensor
		max_seq_length: Maximum length of sequence for tokenization
		truncation: Whether to truncate texts that are longer than `max_seq_length`
		padding: Whether to pad texts that are shorter than `max_seq_length`
	
	Returns:
		Tensor of each text's token IDs. Size of the returned object is `len(texts)` x `max_seq_length` + 2 (manually added BOS and EOS tokens).
	"""
	tokenizer.pad_token_id = tokenizer.eos_token_id # Set PAD = EOS
	text_ids = tokenizer(
		text=texts, truncation=truncation, padding=padding,
		max_length=max_seq_length, return_offsets_mapping=True
	)['input_ids']
	return torch.tensor(
		[[tokenizer.bos_token_id] + text + [tokenizer.eos_token_id] for text in text_ids],
		device=device, dtype=torch.long
	)


def build_labels_for_attribs_calc(
	label_ids_model: torch.LongTensor,
	device: str
) -> torch.LongTensor:
	"""
	Construct tensors of label IDs from which to calculate various token attributes.

	Args:
		label_ids_model: Label IDs passed through the LM
		device: Device on which to load tensor
	
	Returns:
		Shifted label IDs. Size of the returned object is `label_ids_model.size(0)` (i.e. the number of tokens in `label_ids_model`) x |V|.
	"""
	return label_ids_model[..., 1:].contiguous().to(device)


def get_unigram_freqs(
	token_ids: torch.LongTensor,
	vocab_size: int,
	bos_token_id: int,
	eos_token_id: int,
	smooth: bool=False
) -> torch.FloatTensor:
	"""
	Calculate unigram frequencies of the corpus. Assumes that PAD = EOS.
	
	Args:
		token_ids: Tokenized corpus
		vocab_size: |V|
		bos_token_id: Vocabulary index of BOS token
		bos_token_id: Vocabulary index of EOS token
		smooth: Whether to apply add-one smoothing
	
	Returns:
		Unigram frequencies of each vocabulary item. Size of the returned object is 1 x |V|.

	Note:
		The counts of BOS and EOS tokens are each required to equal the number of texts. This is to prevent the inflation of EOS token counts due to padding.
	
	Example:
	>>> labels = torch.tensor([[0, 1, 2, 3, 4, 8, 9, 0, 0, 0], # A padded text
							[0, 2, 2, 3, 8, 4, 4, 7, 8, 0]]) # A truncated or normal text
	>>> vocab_size = 10
	>>> bos_token_id, eos_token_id = 0, 0
	>>> smooth = False
	>>> get_unigram_freqs(labels, vocab_size, bos_token_id, eos_token_id, smooth)
	tensor([0.2222, 0.0556, 0.1667, 0.1111, 0.1667, 0.0000, 0.0000, 0.0556, 0.1667, 0.0556])
	>>> smooth = True
	>>> get_unigram_freqs(labels, vocab_size, bos_token_id, eos_token_id, smooth)
	tensor([0.1786, 0.0714, 0.1429, 0.1071, 0.1429, 0.0357, 0.0357, 0.0714, 0.1429, 0.0714])
	"""
	# Calculate bin counts, distinguish pads from "real" tokens (could also use pad mask)
	bin_counts = torch.bincount(token_ids.view(-1), minlength=vocab_size)
	if len(token_ids.size()) == 1: # If only one text
		num_padded = bin_counts[eos_token_id] - 2
	else:
		num_padded = bin_counts[eos_token_id] - (token_ids.size(0) * 2)
	num_real_tokens = torch.numel(token_ids) - num_padded
	# Remove pad tokens from count
	if bos_token_id == eos_token_id:
		bin_counts[bos_token_id] = token_ids.size(0) * 2 # 1 BOS, 1 EOS per text
	else:
		bin_counts[bos_token_id] = token_ids.size(0)
		bin_counts[eos_token_id] = token_ids.size(0)
	# Return frequencies (optionally smoothed)
	if smooth:
		return (bin_counts + 1) / (num_real_tokens + vocab_size)
	return bin_counts / num_real_tokens


def get_model_outputs(
	texts: List[str],
	max_seq_len: int,
	tokenizer: PreTrainedTokenizerFast,
	lm: PreTrainedModel,
	device: str
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
	"""
	Calculate the model logits of each text in the corpus/corpus batch.

	Args:
		texts: List of input texts
		max_seq_len: Maximum length of sequence for tokenization
		tokenizer: Tokenizer to map tokens to vocabulary indices
		lm: LM to approximate token probabilities
		device: Device on which to load tensor
	
	Returns:
		Tuple of label texts' logits and label IDs to calculate attributes. Sizes of the returned objects are:
			1) (`num_texts` x `max_seq_len`) x |V|: total number of tokens across label texts multiplied by |V|
			2) `num_texts` x `max_seq_len: total number of tokens across label texts
	"""
	input_ids = tokenize(texts, tokenizer, device, max_seq_length=max_seq_len)
	label_ids_model = input_ids
	
	logits, label_ids_attribs = list(), list()
	for inputs, labels in zip(input_ids, label_ids_model):
		# Compute logits
		with torch.no_grad():
			output = lm(input_ids=inputs, labels=labels)
		text_logits = text_logits[..., :-1, :].contiguous()
		labels_attribs = build_labels_for_attribs_calc(labels, device)
		
		# Sanity check (mean CE loss should be close to automatic loss calculation)
		ce_loss = F.cross_entropy(
			input=text_logits.view(-1, text_logits.size(-1)),
			target=labels_attribs.view(-1),
			reduction='none'
		)
		assert torch.isclose(input=torch.mean(ce_loss), other=output['loss'])
		
		logits.append(text_logits)
		label_ids_attribs.append(labels_attribs)
	
	logits, label_ids_attribs = torch.cat(logits, dim=0), torch.cat(label_ids_attribs, dim=0)
	return logits, label_ids_attribs


def calc_probs(logits: torch.FloatTensor) -> torch.FloatTensor:
	"""
	Calculate the token probabilities under the model of each text in the corpus.

	Args:
		logits: Token logits (scores) under the model
	
	Returns:
		Token probabilities (softmaxed scores)
	"""
	return F.softmax(logits, dim=-1)


def calc_nlls(logits: torch.FloatTensor) -> torch.FloatTensor:
	"""
	Calculate the per-token negative log likelihoods (NLLs) of each token in the corpus. This is equivalent to calculating the per-token information content (IC).

	Args:
		logits: Token logits (scores) under the model
	
	Returns:
		Token NLLs
	"""
	return -1 * F.log_softmax(logits, dim=-1)


def calc_entropies(probs: torch.FloatTensor, nlls: torch.FloatTensor) -> torch.FloatTensor:
	"""
	Calculate entropies of the distribution at each time step (a.k.a. expected information content).

	The entropy of a sequence x that takes values in the alphabet X and is distributed according to P : X -> [0, 1] is defined as
		H(X) = -1 * \sum\limits_{x \in X} P(x) log P(x)
	
	Args:
		probs: Token probabilities under the model
		nlls: Token NLLs
	
	Returns:
		Entropies
	"""
	assert torch.all(nlls >= 0) and torch.all(probs >= 0)
	entropies = torch.sum(probs * nlls, dim=-1)
	return entropies.unsqueeze(dim=-1)


def calc_diff_nlls_ents(nlls: torch.FloatTensor, entropies: torch.FloatTensor) -> torch.FloatTensor:
	"""
	Calculate the difference between token NLL and distribution entropy.

	Args:
		nlls: Token NLLs
		entropies: Entropies
	
	Returns:
		Differences between NLL and entropy
	"""
	return nlls - entropies


def is_top_k(probs: torch.FloatTensor, k: int, device: str) -> torch.FloatTensor:
	"""
	Create indicator variable for whether a token is in the top `k` highest probability words.

	Args:
		probs: Token probabilities under LM
		k: Number of top probability values to consider
		device: Device onto which the tensor should be loaded
	
	Returns:
		Tensor indicating whether the label token is in the top `k` highest probability words at time step `t` (0 = False, 1 = True)
	
	References:
		https://discuss.pytorch.org/t/setting-the-values-at-specific-indices-of-a-2d-tensor/168564
	
	Example:
	>>> probs = torch.tensor([[0.1816, 0.1141, 0.1052, 0.1106, 0.1000, 0.1919, 0.0899, 0.1068],
				[0.0734, 0.1361, 0.1021, 0.1207, 0.1212, 0.1170, 0.1776, 0.1518],
				[0.0903, 0.1062, 0.1008, 0.1436, 0.0947, 0.1802, 0.1064, 0.1777],
				[0.1587, 0.1406, 0.0921, 0.1807, 0.1095, 0.0963, 0.1422, 0.0799]])
	>>> k = 3
	>>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
	>>> is_top_k(probs, k, device)
	tensor([[1, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0]])
	"""
	top_k_idx = torch.topk(probs, k).indices
	top_k_indicator = torch.zeros([probs.size(0), probs.size(-1)], device=device, dtype=torch.long)
	top_k_indicator[torch.arange(top_k_indicator.size(0)), top_k_idx.t()] = 1 # Assign 1 to top-k
	assert torch.sum(top_k_indicator) == torch.numel(top_k_idx) # Need as many 1's as top-k indices
	return top_k_indicator