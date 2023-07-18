#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate.py

"""
Generate texts.

Sample program call from main directory:
	python3 src/generate.py --host remote \
		--decoding-strategy top-k --param-val 50 --max-gen-len 150 \
		--batch-size 5 --update-interval 50
"""

import os
import re
import time
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria

import utils_tensor
import utils_train
import utils_generate

torch.cuda.empty_cache()


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
		'--max-gen-len',
		type=int, default=512, required=True,
		help='Maximum length of text to generate.'
	)
	parser.add_argument(
		'--batch-size',
		type=int, default=1, required=False,
		help='How many prompts to batch at a time for generation.'
	)
	parser.add_argument(
		'--seed',
		type=int, default=42, required=False,
		help='Random seed for text generation.'
	)
	parser.add_argument(
		'--update-interval',
		type=int, default=100, required=False,
		help='How often to print update to console (number of batches).'
	)
	return parser


if __name__ == '__main__':
	# Parse CL arguments
	parser = get_argument_parser()
	args = parser.parse_args()
	tau = eval(args.param_val)

	# Define directory paths
	if args.host == 'local':
		HOME_DIR = Path(__file__).resolve().parents[1]
	elif args.host == 'remote':
		HOME_DIR = os.getcwd() # /home/[username]/data
	DATA_DIR = os.path.join(HOME_DIR, 'data')
	CORPUS_DIR = os.path.join(DATA_DIR, 'corpus')
	MODEL_DIR = os.path.join(HOME_DIR, 'models')
	WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')
	PROMPTS_DIR = os.path.join(DATA_DIR, 'prompts')
	HYPS_DIR = os.path.join(DATA_DIR, 'hypotheses')
	HYPS_SUB_DIR = os.path.join(HYPS_DIR, str(args.seed))
	if not os.path.exists(HYPS_DIR):
		os.makedirs(HYPS_DIR)
	if not os.path.exists(HYPS_SUB_DIR):
		os.makedirs(HYPS_SUB_DIR)

	# Select optimal device and num_workers for processing
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cudnn.benchmark = False # Deterministically select algorithm for reproducibility
		num_workers = 1
		pin_memory = True
	else:
		device = torch.device('cpu')
		map_location = torch.device('cpu')
		num_workers = 0
		pin_memory = False

	# Get generation tools
	lm, tokenizer, _, _ = utils_train.get_tools(args.task, device)
	tokenizer.pad_token_id = tokenizer.eos_token_id
	assert lm.device.type == device.type
	print(f'LM and tokenizer prepared.', flush=True)

	# Assemble the desired logits warpers and stopping criteria
	learned_model_num_str = str(tau)
	learned_model_num_str = learned_model_num_str if len(learned_model_num_str) > 1 else '0' + str(learned_model_num_str)
	WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, args.task + '_' + learned_model_num_str + '.pkl')
	CORPUS_FREQ_PATH = os.path.join(CORPUS_DIR, 'wiki200k.pt')
	if utils_generate.normalize_strategy_name(args.decoding_strategy) == 'learned':
		k = 50 if tau >= 7 else 100 # Experiments 1-6: k = 100, 7-12: k = 50
		logits_warper, hyps_filename = utils_generate.get_logits_warper(
			args.decoding_strategy, tau,
			tokenizer, WEIGHTS_PATH, CORPUS_FREQ_PATH, k
		)
	else:
		logits_warper, hyps_filename = utils_generate.get_logits_warper(
			args.decoding_strategy, tau
		)
	logits_warper_list = LogitsProcessorList([logits_warper])
	stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=args.max_gen_len)])
	print(f'Logits warper(s) and stopping criteria defined.', flush=True)

	# Define filenames
	PROMPTS_PATH = os.path.join(PROMPTS_DIR, args.task + '.prompts')
	HYPS_PATH = os.path.join(HYPS_SUB_DIR, hyps_filename)
	if os.path.exists(HYPS_PATH):
		os.remove(HYPS_PATH)
	
	# Print generation start
	start_gen = time.time()	
	print('=' * 89, flush=True)
	print(f'Preparing for generation...', flush=True)
	print('', flush=True)
	print(f'\tDecoding strategy: {utils_generate.normalize_strategy_name(args.decoding_strategy).upper()}', flush=True)
	print(f'\tTau: {tau}', flush=True)
	print(f'\tDevice: {device.type.upper()}', flush=True)
	print('', flush=True)
	print(f'\tPrompts path: {PROMPTS_PATH}')
	print(f'\tHypothesis path: {HYPS_PATH}')
	
	# Assemble prompts
	with open(PROMPTS_PATH, mode='r', encoding='utf-8') as f_prompts:
		prompts = [prompt.rstrip('\n') for prompt in f_prompts]
	assert len(prompts) > 0

	# Batch prompts reproducibly
	g = torch.Generator()
	g.manual_seed(args.seed)
	prompts_loader = DataLoader(
		prompts, batch_size=args.batch_size,
		pin_memory=pin_memory, num_workers=num_workers,
		worker_init_fn=utils_train.seed_worker(args.seed), generator=g
	)

	# Generate (sampling-based)
	print('=' * 89, flush=True)
	print(f'Beginning generation...', flush=True)
	torch.manual_seed(args.seed)
	with open(HYPS_PATH, mode='w', encoding='utf-8') as hyps:
		with tqdm(prompts_loader, unit='prompt') as prompts_batches:
			i = 1
			for batch in prompts_batches:
				# Tokenize
				input_ids = tokenizer(batch, return_tensors='pt', padding='longest')['input_ids'].to(device)
				assert input_ids.device.type == device.type

				# Generate
				start_sample = time.time()
				if utils_generate.normalize_strategy_name(args.decoding_strategy) == 'learned':
					output_ids = [
						lm.sample(
							input_ids[i].unsqueeze(dim=0), logits_warper=logits_warper,
							stopping_criteria=stopping_criteria,
							pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
						) for i in range(input_ids.size(0))
					]
					output_ids = utils_tensor.pad_list_of_tensors(
						output_ids,
						pad_length=args.max_gen_len, pad_value=tokenizer.eos_token_id,
						device=device, stack=True
					).squeeze()
				else:
					if args.batch_size == 1:
						output_ids = lm.sample(
							input_ids=input_ids[0].unsqueeze(dim=0), logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
						).squeeze()
					else:
						output_ids = lm.sample(
							input_ids=input_ids, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
						).squeeze()
				assert output_ids.device.type == device.type

				# Decode and write hypotheses to file
				start_decode = time.time()
				with open(HYPS_PATH, mode='a', encoding='utf-8') as hyps:
					if args.batch_size == 1:
						hypothesis = tokenizer.decode(output_ids, skip_special_tokens=True)
						hypothesis = re.sub(r'\s', r' ', hypothesis).rstrip()
						hypothesis = re.sub(r'\s{2,}', r' ', hypothesis).rstrip()
						hyps.write(hypothesis + '\n')
					else:
						hypotheses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
						for hypothesis in hypotheses:
							hypothesis = re.sub(r'\s', r' ', hypothesis).rstrip()
							hypothesis = re.sub(r'\s{2,}', r' ', hypothesis).rstrip()
							hyps.write(hypothesis + '\n')
				assert os.path.exists(HYPS_PATH) and os.stat(HYPS_PATH).st_size != 0

				# Print progress
				if i % args.update_interval == 0:
					print('-' * 89, flush=True)
					print(f'Generated and wrote {i * args.batch_size}/{len(prompts)} hypotheses to file so far.', flush=True)
					print('', flush=True)
					print('Sample generation:', flush=True)
					print('', flush=True)
					print(f'\tPrompt: {batch[-1]}', flush=True)
					print('', flush=True)
					print(f'\tHypothesis: {hypothesis}', flush=True)
					print('', flush=True)
					print(f'Elapsed time since start of generation: {str(timedelta(seconds=(time.time()-start_gen)))}.', flush=True)
				i += 1
				torch.cuda.empty_cache()
	
	# Print finish
	print('=' * 89, flush=True)
	print(f'Finished generation in {str(timedelta(seconds=(time.time()-start_gen)))} seconds.', flush=True)