#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# early_stopper.py

"""
Implement early stopping.

Inspired by: https://stackoverflow.com/a/73704579
"""

from typing import Tuple, Union


class EarlyStopper:
	"""
	Trigger early stopping when the best validation loss is `patience` validation periods past.
	
	Args:
		patience: Tolerated number of validation periods during which validation loss does not decrease, after which early stopping will occur.
		min_delta: Value by which validation loss must improve in order to count as an improvement.
	
	Attributes:
		patience: Tolerated number of validation periods during which validation loss does not decrease, after which early stopping will occur.
		min_delta: Value by which validation loss must improve in order to count as an improvement.
		best_val_loss: Lowest validation loss at the validation period.
		periods_past: Number of validation periods passed since the best validation loss.
	"""
	
	def __init__(self, patience: int, min_delta: float=5e-2):
		self.patience = patience
		self.min_delta = min_delta
		self.best_val_loss = None
		self.periods_past = 0
	
	def early_stop(self, curr_val_loss: float) -> Union[Tuple[bool, float], bool]:
		if self.best_val_loss is None or curr_val_loss < self.best_val_loss:
			print(f'New best validation loss: {self.best_val_loss} -> {curr_val_loss}')
			self.best_val_loss = curr_val_loss
			self.periods_past = 0
		elif curr_val_loss >= self.best_val_loss + self.min_delta:
			self.periods_past += 1
			if self.periods_past >= self.patience:
				return True, self.best_val_loss
		return False