#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py

"""
Train a linear regression model.

Sample program call from main directory:
	python3 src/train.py --host remote --model-num 3 \
	--attribute-names probs unigram_freq abs_diff top_k --top-k 100 \
	--num-texts 10000 --max-seq-percentile 85 \
	--max-num-epochs 10 --batch-size 40 \
	--val-interval 2 --val-split 0.20 \
	--max-lr 1e-2 --sched-patience 4 --sched-factor 0.1 \
	--early-stop-patience 2 --seed 42
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import attributes
import constants
import logger
import utils_train
from early_stopper import EarlyStopper
from hellinger import HellingerDistance
from regression import LinearRegression

torch.set_grad_enabled(True)
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)


def get_argument_parser() -> ArgumentParser:
	parser = ArgumentParser(description='Specify parameters for training.')
	parser.add_argument(
		'--host',
		type=str, required=True,
		help='Where the script is being run. Choose from: local, remote.'
	)
	parser.add_argument(
		'--model-num',
		type=int, required=True,
		help='Model number.'
	)
	parser.add_argument(
		'--attribute-names',
		nargs='+', required=True,
		help='Attributes to be mapped to one-hot encoded corpus. Choose from: \
			probs (probability of token under model) \
			unigram_freq (unigram frequency) \
			diff (difference between token IC and entropy)	\
			abs_diff (absolute difference between token IC and entropy)	\
			sq_diff (squared difference between token IC and entropy) \
			top_k (whether token is in top k probability tokens)'
	)
	parser.add_argument(
		'--top-k',
		type=int, default=None, required=False,
		help='Number of top probability values to consider.'
	)
	parser.add_argument(
		'--num-texts',
		type=int, required=True,
		help='Number of texts to use for learning weights.'
	)
	parser.add_argument(
		'--max-seq-percentile',
		type=int, default=95, required=False,
		help='Percentile of text lengths to set as maximum tokenizer input length.'
	)
	parser.add_argument(
		'--max-num-epochs',
		type=int, required=True,
		help='Maximum number of epochs for training.'
	)
	parser.add_argument(
		'--val-interval',
		type=int, default=1, required=False,
		help='After how many epochs to calculate validation loss.'
	)
	parser.add_argument(
		'--val-split',
		type=float, default=0.15, required=False,
		help='Size of validation set relative to training corpus.'
	)
	parser.add_argument(
		'--batch-size',
		type=int, default=10, required=False,
		help='Size of training batch.'
	)
	parser.add_argument(
		'--max-lr',
		type=float, required=True,
		help='Initial learning rate.'
	)
	parser.add_argument(
		'--sched-patience',
		type=float, default=4, required=False,
		help='Number of epochs with non-decreasing losses to ignore before annealing learning rate.'
	)
	parser.add_argument(
		'--sched-factor',
		type=float, default=0.1, required=False,
		help='Multiplicative factor by which to reduce learning rate.'
	)
	parser.add_argument(
		'--early-stop-patience',
		type=int, default=0, required=False,
		help='Tolerated number of validation periods during which validation loss does not decrease, after which early stopping will occur.'
	)
	parser.add_argument(
		'--seed',
		type=int, default=42, required=False,
		help='Random seed.'
	)
	return parser


def build_regression_input(texts: List[str]) -> Tuple[torch.FloatTensor, torch.LongTensor]:
	"""
	Construct attributes tensor that will be used to learn mapping to one-hot encoded corpus.

	Args:
		texts: List of texts

	Returns:
		Attributes tensor. Size of returned object is `|V|` x `m` x `k`, where `|V|` = LM vocabulary size, `m` = number of tokens across label texts, and `k` = number of attributes.
	"""
	# Run texts through LM, calculate attributes
	logits, label_ids_attribs = attributes.get_model_outputs(
		texts, max_seq_len, tokenizer, lm, device
	)

	# Calculate probabilities
	probs = attributes.calc_probs(logits)
	nlls = attributes.calc_nlls(logits)
	entropies = attributes.calc_entropies(probs, nlls)
	diff_nlls_ents = attributes.calc_diff_nlls_ents(nlls, entropies)

	# Build regression input
	token_attributes = list()
	if 'probs' in args.attribute_names:
		token_attributes.append(probs)
	if 'top_k' in args.attribute_names:
		token_attributes.append(attributes.is_top_k(probs, args.top_k, device))
	if 'unigram_freq' in args.attribute_names:
		CORPUS_FREQ_PATH = os.path.join(CORPUS_DIR, 'wiki200k.pt')
		unigram_freqs = torch.load(CORPUS_FREQ_PATH, map_location=torch.device(device))
		unigram_freqs = unigram_freqs.repeat(probs.size(0), 1)
		token_attributes.append(unigram_freqs)
	if 'diff' in args.attribute_names:
		token_attributes.append(diff_nlls_ents)
	if 'abs_diff' in args.attribute_names:
		token_attributes.append(torch.abs(diff_nlls_ents))
	if 'sq_diff' in args.attribute_names:
		token_attributes.append(torch.square(diff_nlls_ents))
	del probs, nlls, entropies, diff_nlls_ents
	regression_input = torch.stack(token_attributes, dim=-1).to(device)
	return regression_input, label_ids_attribs


def _print_params() -> None:
	"""Print training parameters to console."""
	print('=' * 89)
	print('')
	print(f'-- LEARNING WEIGHTS: MODEL {args.model_num}')
	print('')
	print('')
	print('-- CORPUS PARAMETERS')
	print(f'\tTrain/Validation split: {1-args.val_split}/{args.val_split}')
	print(f'\t  Training corpus size: {len(train_texts)}')
	print(f'\t  Validation corpus size: {len(valid_texts)}')
	print(f'\tMaximum text length percentile: {args.max_seq_percentile}')
	print(f'\tMaximum model text length: {max_seq_len}')
	print('')
	print('-- TRAINING HYPERPARAMETERS')
	print(f'\tPredictors: {args.attribute_names}')
	if args.top_k:
		print(f'\t  Top `k` = {args.top_k}')
	print(f'\tMaximum epochs: {args.max_num_epochs}')
	print(f'\tBatch size: {args.batch_size}')
	print(f'\tInitial learning rate: {args.max_lr}')
	print(f'\tValidation interval (epochs): {args.val_interval}')
	print(f'\tSchedule factor: {args.sched_factor}')
	print(f'\tSchedule patience (epochs): {int(args.sched_patience)}')
	print(f'\tEarly stopping: {True if args.early_stop_patience else False} (patience = {int(args.early_stop_patience) if args.early_stop_patience else "N/A"} validation periods)')
	print(f'\tSeed: {args.seed}')
	print('')
	print('-- DEVICE PARAMETERS')
	print(f'\tDevice: {str(device).upper()}')
	print(f'\tPin memory: {pin_memory}')
	print(f'\tNumber of workers: {num_workers}')
	print('')
	print('-- LOG AND MODEL PATHS')
	print(f'\tLogs: {LOGS_DIR}')
	print(f'\tModels: {WEIGHTS_DIR}')
	print('')
	print('=' * 89)
	print('')


def _print_epoch_loss(epoch: int, lr: float, learn_time: float, loss: float) -> None:
	"""Print per-epoch loss to console."""
	print('-' * 89)
	print(f'epoch {epoch} / {args.max_num_epochs} | lr {lr} | epoch training duration {time.strftime("%H:%M:%S", time.gmtime(learn_time))} | loss {loss.item()}')
	print('-' * 89)


def _print_finish(train_time: float) -> None:
	"""Print end-of-training message to console."""
	print('')
	print('=' * 89)
	print('|\t\t\t\t\t\t\t\t\t  END OF TRAINING  \t\t\t\t\t\t\t\t\t|')
	print(f'Completed training in {time.strftime("%H:%M:%S", time.gmtime(train_time))} ')
	print('=' * 89)


def save_model() -> None:
	"""Save model to file."""
	torch.save(train_model.state_dict(), MODEL_PATH)


def validate() -> float:
	"""Perform model validation."""
	print('-' * 89)
	print(f'Computing validation loss...')
	torch.cuda.empty_cache()
	val_batch_losses = list()
	with torch.no_grad():
		with tqdm(valid_loader, unit='batch', desc='Batches') as valid_batches:
			for batch in valid_batches:
				val_attributes, label_ids_attribs = build_regression_input(batch[0], batch[1])
				val_predictions = train_model(val_attributes)
				val_batch_loss = criterion(val_predictions, label_ids_attribs)
				val_batch_losses.append(val_batch_loss.item())
	return np.mean(val_batch_losses)


def train() -> None:
	"""Train the linear regression model."""
	print('=' * 89)
	print('|\t\t\t\t\t\t\t\t\tSTART OF TRAINING\t\t\t\t\t\t\t\t\t|')
	print('=' * 89)
	start_train = time.time()

	if args.early_stop_patience:
		early_stopper = EarlyStopper(args.early_stop_patience)
		best_val_loss = np.inf
	
	for epoch in range(1, args.max_num_epochs+1):
		start_epoch = time.time()
		
		with tqdm(train_loader, unit='batch', desc='Batches') as train_batches:
			for batch in train_batches:
				# Build regression input tensor, compute forward pass (with softmax)
				optimizer.zero_grad()
				train_attributes, label_ids_attribs = build_regression_input(list(batch))
				p_hat = train_model(train_attributes)
				del train_attributes
				torch.cuda.empty_cache()

				# Calculate loss, backpropagate
				train_loss = criterion(p_hat, label_ids_attribs)
				train_loss.backward()
				del p_hat, label_ids_attribs
				torch.cuda.empty_cache()

				# Update parameters
				optimizer.step()
			
			# Print and log losses
			end_epoch = time.time() - start_epoch

			_print_epoch_loss(epoch=epoch, lr=optimizer.param_groups[0]['lr'], learn_time=end_epoch, loss=train_loss)
			tb_logger.add_scalar('Train Loss/Epoch', train_loss.item(), epoch-1)
			scheduler.step(train_loss)

			# Validation
			try:
				if epoch % args.val_interval == 0:
					curr_val_loss = validate()
					if curr_val_loss < best_val_loss: # Keep model with lowest val loss thus far
						best_val_loss = curr_val_loss
						save_model()
						print(f'New best model saved.')
					tb_logger.add_scalar('Validation Loss/Epoch', curr_val_loss, epoch)
					print('')
					print(f'training {int(epoch/args.max_num_epochs * 100)}% complete | validation loss {curr_val_loss}')

					# Early stopping
					if args.early_stop_patience:
						if early_stopper.early_stop(curr_val_loss):
							print(f'Early stopping after epoch {epoch} / {args.max_num_epochs}: best validation error {early_stopper.early_stop(curr_val_loss)[1]} is {args.early_stop_patience} validation periods past.')
							break
					print('-' * 89)
			except ZeroDivisionError: # If not performing validation
				pass
	
	tb_logger.flush()
	if not args.early_stop_patience:
		save_model()
	end_train = time.time() - start_train
	_print_finish(train_time=end_train)


if __name__ == '__main__':
	# Parse CL arguments
	parser = get_argument_parser()
	args = parser.parse_args()

	# Ensure that arguments are valid
	for attribute_name in args.attribute_names: # Whether `attribute`` is valid
		assert attribute_name in constants.ATTRIBUTES
	assert 'top_k' in args.attribute_names and args.top_k # Must assign `k``

	# Define directory paths
	if args.host == 'local':
		HOME_DIR = Path(__file__).resolve().parents[1]
	elif args.host == 'remote':
		HOME_DIR = os.getcwd()
	DATA_DIR = os.path.join(HOME_DIR, 'data')
	CORPUS_DIR = os.path.join(DATA_DIR, 'corpus')
	MODEL_DIR = os.path.join(HOME_DIR, 'models')
	WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')
	LOGS_DIR = os.path.join(HOME_DIR, 'logs')
	TRAIN_LOGS_DIR = os.path.join(LOGS_DIR, 'train')
	TB_LOGS_DIR = os.path.join(LOGS_DIR, 'tb')

	# Define log path and direct output to both console and log file
	LOG_NUM_STR = str(args.model_num) if len(str(args.model_num)) > 1 else '0' + str(args.model_num)
	TRAIN_LOG_PATH = os.path.join(TRAIN_LOGS_DIR, LOG_NUM_STR + '_' + str(args.seed) + '.log')
	sys.stdout = logger.Logger(TRAIN_LOG_PATH)
	sys.stderr = sys.stdout

	# Define model path
	MODEL_NAME = LOG_NUM_STR + '.pkl'
	MODEL_PATH = os.path.join(WEIGHTS_DIR, MODEL_NAME)

	# Select optimal device for processing
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cudnn.benchmark = False # Reproducibility
		num_workers = 2
		pin_memory = True
		worker_id = torch.cuda.current_device()
	else:
		device = torch.device('cpu')
		num_workers = 0
		pin_memory = False
		worker_id = 13
	
	# Load corpus
	CORPUS_TEXTS_PATH = os.path.join(CORPUS_DIR, 'wiki16k.txt')
	input_texts = utils_train.get_corpus(CORPUS_TEXTS_PATH, max_num_texts=args.num_texts).texts
	label_texts = input_texts

	# Load model, tokenizer, config; determine vocab size
	lm, tokenizer, lm_config = utils_train.get_lm_tools(device)
	assert lm.device.type == device.type # Make sure LM has been moved to the proper device

	# Calculate maximum sequence lengths for tokenizer
	max_seq_len = utils_train.get_max_text_len(input_texts, args.max_seq_percentile, lm_config)
	
	# Split texts intro training and validation groups
	train_texts, valid_texts, _, _ = train_test_split(
		input_texts, label_texts, test_size=args.val_split, random_state=args.seed
	)
	
	# Batch texts reproducibly
	g = torch.Generator()
	g.manual_seed(args.seed)
	train_loader = DataLoader(
		train_texts, batch_size=args.batch_size,
		pin_memory=pin_memory, num_workers=num_workers,
		worker_init_fn=utils_train.seed_worker(args.seed), generator=g
	)
	valid_loader = DataLoader(
		valid_texts, batch_size=args.batch_size,
		pin_memory=pin_memory, num_workers=num_workers,
		worker_init_fn=utils_train.seed_worker(args.seed), generator=g
	)

	# Print parameters to console
	_print_params()
	del input_texts, label_texts, train_texts, valid_texts

	# Instantiate linear regression class
	dim_in = len(args.attribute_names)
	dim_out = 1
	train_model = LinearRegression(dim_in, dim_out, device=device)

	# Instantiate TB logger
	tb_logger = SummaryWriter(log_dir=TB_LOGS_DIR, comment=LOG_NUM_STR)

	# Instantiate loss criterion, optimizer, LR scheduler
	criterion = HellingerDistance(device=device)
	optimizer = torch.optim.Adam(params=train_model.parameters(), lr=args.max_lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.sched_factor, patience=args.sched_patience, threshold=1e-3, min_lr=1e-8)

	# Train, record and print losses, save model
	train()