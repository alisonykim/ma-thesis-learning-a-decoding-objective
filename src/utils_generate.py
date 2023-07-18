#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils_generate.py

"""
Utility functions for text generation.
"""

from typing import Optional, Tuple, Union

from transformers import (
	PreTrainedTokenizerFast,
	LogitsWarper,
	TopKLogitsWarper,
	TemperatureLogitsWarper,
	TopPLogitsWarper,
	TypicalLogitsWarper
)

import constants
from learned_logits_warper import LearnedLogitsWarper


def check_warper_param_val(
	strategy: str,
	param_val: Union[int, float],
	tokenizer: PreTrainedTokenizerFast
) -> None:
	"""Ensure that `param_val` is valid for a particular decoding strategy."""
	STRATEGIES = constants.DECODING_STRATEGIES
	assert strategy in [running for strategy in STRATEGIES.values() for running in strategy]

	if strategy in STRATEGIES['learned']:
		assert isinstance(param_val, int)
	elif strategy in STRATEGIES['top_k']:
		assert isinstance(param_val, int)
		assert param_val >= 1 and param_val <= tokenizer.vocab_size
	elif strategy in STRATEGIES['temp']:
		assert param_val > 0
	elif strategy in STRATEGIES['top_p']:
		assert 0 < param_val <= 1
	elif strategy in STRATEGIES['typical']:
		assert 0 < param_val <= 1
	else:
		raise ValueError(f'The decoding strategy {strategy} is not supported at this time.')


def normalize_strategy_name(strategy: str) -> str:
	"""Normalize decoding strategy name."""
	STRATEGIES = constants.DECODING_STRATEGIES
	for normalized, running in STRATEGIES.items():
		if strategy in running:
			return normalized


def make_hyps_filename(strategy: str, param_val: Union[int, float]) -> str:
	"""Create filename for chosen strategy and parameter value."""
	if strategy in constants.DECODING_STRATEGIES['top_k']:
		return normalize_strategy_name(strategy) + f'_{str(int(param_val))}.hyps'
	elif strategy in constants.DECODING_STRATEGIES['learned']:
		if len(str(param_val)) == 1:
			return normalize_strategy_name(strategy) + f'_0{str(param_val)}.hyps'
		return normalize_strategy_name(strategy) + f'_{str(int(param_val))}.hyps'
	return normalize_strategy_name(strategy) + str(param_val) + '.hyps'


def get_logits_warper(
	strategy: str,
	param_val: Union[int, float],
	tokenizer: PreTrainedTokenizerFast,
	learned_weights_path: Optional[str],
	corpus_freq_path: Optional[str],
	k: Optional[int]
) -> Tuple[LogitsWarper, str]:
	"""Prepare the correct logits warper."""
	check_warper_param_val(strategy, param_val, tokenizer)
	hyps_filename = make_hyps_filename(strategy, param_val)
	STRATEGIES = constants.DECODING_STRATEGIES
	if strategy in STRATEGIES['learned']:
		return LearnedLogitsWarper(learned_weights_path, corpus_freq_path, k), hyps_filename
	elif strategy in STRATEGIES['top_k']:
		return TopKLogitsWarper(top_k=int(param_val)), hyps_filename
	elif strategy in STRATEGIES['temp']:
		return TemperatureLogitsWarper(temperature=param_val), hyps_filename
	elif strategy in STRATEGIES['top_p']:
		return TopPLogitsWarper(top_p=param_val), hyps_filename
	elif strategy in STRATEGIES['typical']:
		return TypicalLogitsWarper(mass=param_val), hyps_filename