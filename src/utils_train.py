#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils_train.py

"""
Utility functions for training linear regression model.
"""

import random
from math import floor
from typing import List, Tuple

import numpy as np
import torch
from transformers import (
	PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig,
	GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
)

import corpus


def get_max_text_len(
	corpus: List[str],
	percentile: float,
	lm_config: PretrainedConfig,
	separator: str=' '
) -> int:
	"""Get `percentile`-th length of texts (number of tokens)."""
	input_lens = np.array([len(text.split(separator)) for text in corpus])
	percentile_val = floor(np.percentile(input_lens, percentile))
	return min(percentile_val, lm_config().n_positions-2) # -2 because manually adding BOS/EOS


def get_corpus(corpus_path: str, max_num_texts: int, min_seq_len: int=30) -> corpus.Wikitext:
	"""Retrieve input and label texts."""
	return corpus.Wikitext(corpus_path, max_num_texts, min_seq_len)


def get_lm_tools(
	device: str,
	checkpoint: str='gpt2-large'
) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig]:
	"""Retrieve pretrained LM, tokenizer, and model configuration."""
	return (
		GPT2LMHeadModel.from_pretrained(checkpoint).to(device),
		GPT2TokenizerFast.from_pretrained(checkpoint),
		GPT2Config
	)


def seed_worker(worker_id: int) -> None:
	"""
	Ensure reproducibility for batching with ```torch.datasets.DataLoader```.
	
	Implementation from: https://pytorch.org/docs/stable/notes/randomness.html
	"""
	worker_seed = torch.initial_seed() % 2 ** 32
	np.random.seed(worker_seed)
	random.seed(worker_seed)