#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evaluate.py

"""
Evaluate the generated texts (hypotheses) with respect to their references.

Sample program call from main directory:
	python3 src/evaluate.py --host remote \
	--decoding-strategy learned --param-val 11 \
	--mauve-scale-factor 5 --mauve-max-len 512 --max-n 4 \
	--seed 42
"""

import json
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
from argparse import ArgumentParser
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import mauve
import nltk
import numpy as np
import torch
from transformers import (
	PreTrainedModel,
	GPT2LMHeadModel, GPT2TokenizerFast,
	BartForConditionalGeneration, BartTokenizerFast
)

import utils_generate

start_eval = time.time() # Start timer


def get_argument_parser() -> ArgumentParser:
	parser = ArgumentParser(description='Specify parameters for training.')
	parser.add_argument(
		'--host',
		type=str, required=True,
		help='Where the script is being run. Choose from: local, remote.'
	)
	parser.add_argument(
		'--decoding-strategy', '-ds',
		type=str, required=True,
		help='Which decoding strategy/ies to use for generation. Choose from: \
			learned \
			top_k \
			temp \
			top_p / nucleus \
			typical'
	)
	parser.add_argument(
		'--param-val', '-tau',
		type=str, required=True,
		help='Which parameter value to use with the decoding strategy. Values must be: \
			learned: model number \
			top_k: [1, |V|] \
			temp: > 0 \
			top_p / nucleus: (0, 1] \
			typical: (0, 1]'		
	)
	parser.add_argument(
		'--mauve-scale-factor',
		type=int, default=5, required=False,
		help='MAUVE scale factor.'
	)
	parser.add_argument(
		'--mauve-max-len',
		type=int, default=512, required=False,
		help='MAUVE maximum text length to consider.'
	)
	parser.add_argument(
		'--seed',
		type=int, default=42, required=False,
		help='Seed to initialize k-means clustering for MAUVE calculations.'
	)
	parser.add_argument(
		'--max-n',
		type=int, default=4, required=False,
		help='Maximum size of n-gram with which to calculate n-gram diversity.'
	)
	return parser


def calc_mauve(
	hypotheses: List[str],
	references: List[str],
	scale_factor: int=5
) -> SimpleNamespace:
	"""
	Calculate MAUVE score.

	References:
		https://github.com/krishnap25/mauve

	Args:
		hypotheses: List of n hypothesis texts, where the i-th element is a text of len_i
		references: List of n reference texts, where the i-th element is a text of len_i
		seed: Random seed to initialize k-means cluster assignments
		kmeans_num_redo: Number of times to redo k-means clustering (best is kept)
	
	Returns:
		`types.SimpleNamespace` with fields {'mauve', 'frontier_integral', 'divergence_curve', 'p_hist, q_hist'}
	"""
	try: # If on GPU
		return mauve.compute_mauve(
			p_text=hypotheses, q_text=references,
			featurize_model_name='gpt2-large', mauve_scaling_factor=scale_factor,
			device_id=torch.cuda.current_device(), seed=args.seed
		)
	except: # If not on GPU
		return mauve.compute_mauve(
			p_text=hypotheses, q_text=references,
			featurize_model_name='gpt2-large', mauve_scaling_factor=scale_factor,
			seed=args.seed
		)


def count_ngrams(text_ids: List[torch.Tensor], n: int) -> Counter:
	"""
	Count n-grams in a text.
	
	Args:
		text_ids: List of tokenized texts
		n: Size of n-gram
	
	Returns:
		Counter of n-grams
	"""
	corpus_ngrams = list()
	for text in text_ids:
		text_ngrams = list(nltk.ngrams(text.tolist(), n))
		corpus_ngrams.extend(text_ngrams)
	return Counter(corpus_ngrams)


def calc_ngram_diversity(
	text_ids: List[torch.Tensor],
	max_n: int=4
) -> Tuple[float, Dict[int, float]]:
	"""
	Calculate corpus token n-gram diversity `D` as defined in Meister, Forster, & Cotterell (2021). Eq. 14 can be rewritten as
		D = 1/max_n * \sum_{i=1}^{max_n} (# unique i-grams in `k` texts) / (# i-grams in text)
	
	Args:
		text_ids: List of tokenized texts
		max_n: Maximum size of n-gram with which to calculate 
	
	Raises:
		ValueError: If n < 1 (must be positive integer greater than 1).
	
	Returns:
		Tuple of `D` and n-gram coverage dictionary (keys = n, values = d_n as defined in aformentioned paper)
	"""
	if max_n < 1:
		raise ValueError(f'`max_n` must be a positive integer greater than 1. Please try with an appropriate value.')
	
	ngram_coverage = dict()
	for n in range(1, max_n+1):
		ngram_counts = count_ngrams(text_ids, n)
		num_unique = len(ngram_counts.keys())
		num_all = sum(ngram_counts.values())
		ngram_coverage[n] = num_unique / num_all
	return np.mean(list(ngram_coverage.values())), ngram_coverage


def calc_text_ppl(text_ids: List[torch.Tensor], lm: PreTrainedModel) -> float:
	"""Calculate generated text perplexity."""
	nlls = list()
	with torch.no_grad():
		for text in text_ids:
			try:
				outputs = lm(text, labels=text)
			except RuntimeError:
				outputs = lm(text.unsqueeze(dim=0))
			nlls.append(outputs['loss'])
	return torch.exp(torch.stack(nlls).mean()).item()
	

if __name__ == '__main__':
	# Parse CL arguments
	parser = get_argument_parser()
	args = parser.parse_args()
	tau = eval(args.param_val)

	# Define paths
	if args.host == 'local':
		HOME_DIR = Path(__file__).resolve().parents[1]
	elif args.host == 'remote':
		HOME_DIR = os.getcwd() # /home/[username]/data
	DATA_DIR = os.path.join(HOME_DIR, 'data')
	HYPS_DIR = os.path.join(DATA_DIR, 'hypotheses')
	HYPS_SUB_DIR = os.path.join(HYPS_DIR, '42')
	REFS_DIR = os.path.join(DATA_DIR, 'references')
	LOGS_DIR = os.path.join(HOME_DIR, 'logs')
	SCORES_DIR = os.path.join(LOGS_DIR, 'scores')
	if not os.path.exists(SCORES_DIR):
		os.mkdir(SCORES_DIR)
	HYPS_FILENAME = utils_generate.make_hyps_filename(args.decoding_strategy, tau)
	HYPS_PATH = os.path.join(HYPS_SUB_DIR, HYPS_FILENAME)
	REFS_FILENAME = 'lm.refs'
	REFS_PATH = os.path.join(REFS_DIR, REFS_FILENAME)
	SCORES_FILENAME = f'{HYPS_FILENAME[:-5]}_{args.seed}.json'
	SCORES_PATH = os.path.join(SCORES_DIR, SCORES_FILENAME)

	# Select optimal device for processing
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cudnn.benchmark = True
	else:
		device = torch.device('cpu')
	
	# Get LM and tokenizer of model used to generate text
	lm = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
	tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
	tokenizer.pad_token_id = tokenizer.eos_token_id

	# Get independent LM and tokenizer
	lm_ind = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
	tokenizer_ind = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')
	tokenizer_ind.pad_token_id = tokenizer.eos_token_id
	
	# Assemble texts
	with open(HYPS_PATH, mode='r', encoding='utf-8') as hyps:
		hyp_texts = [hyp.rstrip() for hyp in hyps] # LM used to generate text
	with open(REFS_PATH, mode='r', encoding='utf-8') as refs:
		ref_texts = [ref.rstrip() for ref in refs] # LM used to generate text
	assert len(hyp_texts) == len(ref_texts)

	# Tokenize (no batch-tokenization, which will inflate EOS token counts)
	# GPT-2 (manually add BOS & EOS)
	hyp_ids = [
		torch.tensor([tokenizer.bos_token_id] + tokenizer(text)['input_ids'] + [tokenizer.eos_token_id]).to(device) for text in hyp_texts
	]
	ref_ids = [
		torch.tensor([tokenizer.bos_token_id] + tokenizer(text)['input_ids'] + [tokenizer.eos_token_id]).to(device) for text in ref_texts
	]
	# BART (BOS & EOS automatically added)
	hyp_ids_ind = [tokenizer_ind(text, return_tensors='pt')['input_ids'].to(device) for text in hyp_texts]
	ref_ids_ind = [tokenizer_ind(text, return_tensors='pt')['input_ids'].to(device) for text in ref_texts]
	
	# Print generation start
	print('=' * 89, flush=True)
	print(f'Preparing for evaluation...', flush=True)
	print(f'  Decoding strategy: {utils_generate.normalize_strategy_name(args.decoding_strategy).upper()}', flush=True)
	print(f'  Tau: {tau}', flush=True)
	print(f'  Device: {device.type.upper()}', flush=True)
	print('', flush=True)
	print(f'  Hypotheses path: {HYPS_PATH}', flush=True)
	print(f'  References path: {REFS_PATH}', flush=True)
	print(f'  Scores path: {SCORES_PATH}', flush=True)
	print('=' * 89, flush=True)
	
	# Calculate MAUVE
	print(f'Calculating MAUVE of hypotheses w.r.t. references...', flush=True)
	start_mauve = time.time()
	mauve_namespace = calc_mauve(
		hyp_texts, ref_texts,
		scale_factor=args.mauve_scale_factor
	)
	print(f'Calculated MAUVE in {str(timedelta(seconds=(time.time()-start_mauve)))}', flush=True)
	print(f'MAUVE score: {mauve_namespace.mauve}', flush=True)
	print(f'-' * 89, flush=True)

	# Calculate PPL (original LM)
	print(f'Calculating PPL of hypotheses and references using generation LM...', flush=True)
	start_ppl = time.time()
	ppl_hyps = calc_text_ppl(hyp_ids, lm)
	ppl_refs = calc_text_ppl(ref_ids, lm)
	print(f'Calculated PPL in {str(timedelta(seconds=(time.time()-start_ppl)))}', flush=True)
	print(f'PPL_hyps = {ppl_hyps}', flush=True)
	print(f'PPL_refs = {ppl_refs}', flush=True)
	print(f'-' * 89, flush=True)

	# Calculate PPL (alternative LM)
	print(f'Calculating PPL of hypotheses and references using independent LM...', flush=True)
	start_ppl_2 = time.time()
	ppl_hyps_ind = calc_text_ppl(hyp_ids_ind, lm_ind)
	ppl_refs_ind = calc_text_ppl(ref_ids_ind, lm_ind)
	print(f'Calculated PPL in {str(timedelta(seconds=(time.time()-start_ppl_2)))}', flush=True)
	print(f'PPL_hyps = {ppl_hyps_ind}', flush=True)
	print(f'PPL_refs = {ppl_refs_ind}', flush=True)
	print(f'-' * 89, flush=True)

	# Calculate n-gram diversity
	print(f'Calculating n-gram diversity of hypotheses and references...', flush=True)
	start_ngram_div = time.time()
	ngram_diversity_hyp, ngram_coverage_hyp = calc_ngram_diversity(hyp_ids, args.max_n)
	ngram_diversity_ref, ngram_coverage_ref = calc_ngram_diversity(ref_ids, args.max_n)
	print(f'Calculated n-gram diversity in {str(timedelta(seconds=(time.time()-start_ngram_div)))}', flush=True)
	print(f'D_hyps = {ngram_diversity_hyp}', flush=True)
	print(f'D_refs = {ngram_diversity_ref}', flush=True)
	print(f'-' * 89, flush=True)
	
	# Prepare JSON string
	eval_dict = {
		'host': args.host,
		'hyps_file': f'{HYPS_FILENAME}',
		'refs_file': f'{REFS_FILENAME}',
		'mauve_kmeans_seed': args.seed,
		'mauve_max_len': args.mauve_max_len,
		'mauve_scale_factor': args.mauve_scale_factor,
		'mauve_score': mauve_namespace.mauve,
		'mauve_frontier_integral': mauve_namespace.frontier_integral,
		'mauve_div_curve': mauve_namespace.divergence_curve.tolist(),
		'mauve_p_hist': mauve_namespace.p_hist.tolist(),
		'mauve_q_hist': mauve_namespace.q_hist.tolist(),
		'ppl_hyps': ppl_hyps,
		'ppl_refs': ppl_refs,
		'ppl_diff': ppl_hyps - ppl_refs,
		'ppl_hyps_ind': ppl_hyps_ind,
		'ppl_refs_ind': ppl_refs_ind,
		'ppl_diff_ind': ppl_hyps_ind - ppl_refs_ind,
		'ngram_max_n': args.max_n,
		'ngram_diversity_hyp': ngram_diversity_hyp,
		'ngram_coverage_hyp': ngram_coverage_hyp,
		'ngram_diversity_ref': ngram_diversity_ref,
		'ngram_coverage_ref': ngram_coverage_ref
	}

	# Write scores dictionary to JSON file
	with open(SCORES_PATH, mode='w', encoding='utf-8') as eval_json:
		json.dump(eval_dict, eval_json, indent=4)
	
	# Print finish
	print('=' * 89, flush=True)
	print(f'Finished evaluation in {str(timedelta(seconds=(time.time()-start_eval)))}.', flush=True)
	print('=' * 89, flush=True)