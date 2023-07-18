#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hellinger.py

"""
Implement Hellinger distance as a loss criterion for learned and labels probability distributions
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class HellingerDistance(nn.Module):
	"""
	Create a loss criterion that measures the similarity between the output and one-hot distributions.

	The Hellinger distance between two discrete probability distributions P = (p_1, ..., p_k) and Q = (q_1, ..., q_k) is defined as
		HD(P, Q) = 1 / \sqrt(2) * \sqrt(\sum_{i=1}^{k} (\sqrt(p_i) - \sqrt(q_i)) ** 2)
	
	Args:
		device: Device on which to load the tensor
	"""

	def __init__(self, device: str):
		super(HellingerDistance, self).__init__()
		self.device = device
	
	def forward(self, output: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
		"""
		Args:
			output: Predicted probability distributions
			labels: Label token IDs

		Returns:
			Hellinger distance between distributions `output` and `labels`
		
		Raises:
			IndexError: If `output` and `labels` have inconsistent dimensions
			ValueError: If `output` does not sum to 1
		
		Examples:
		>>> device = 'cpu'
		>>> criterion = HellingerDistance(device)
		>>> output = torch.tensor([[0.25, 0., 0., 0.7500],
								[0.02, 0., 0.98, 0.],
								[0.8, 0.15, 0., 0.05],
								[0., 0., 0., 1.]])
		>>> labels = torch.tensor([[3, 2], [0, 0]]) # one-hot [[0, 0, 0, 1],
															[0, 0, 1, 0],
															[1, 0, 0, 0],
															[1, 0, 0, 0]]
		>>> criterion(output, labels)
		tensor(1.1179)
		>>> output2 = torch.tensor([[0.25, 0., 0., .7500],
								[0.02, 0., 0.98, 0.],
								[0.8, 0.15, 0., 0.05],
								[1., 1., 1., 1.]]) # not a valid probability distribution
		>>> criterion(output2, labels)
		ValueError: At least 1 row of `output` does not represent a valid probability distribution(s). Please ensure that every row sums to 1.

		Specific row(s) that triggered ValueError were recorded in 'path/to/logs'.
		>>> labels2 = torch.tensor([1, 0, 0])
		>>> criterion(output, labels2)
		IndexError: Expected number of elements of `labels` to equal number of rows of `output` (one ground truth label per row).
			`output`: 4 rows
			`labels`: 3 labels
		"""
		self._has_valid_dimensions(output, labels)
		
		is_prob_dist = self._is_prob_dist(output, dim=-1)
		if not is_prob_dist[0]:
			error_msg = f'At least 1 row of `output` does not represent a valid probability distribution(s). Please ensure that every row sums to 1.\n\nSpecific row(s) that triggered ValueError:\n{is_prob_dist[1]}'
			raise ValueError(f'{error_msg}')
		
		# Expand summand: (\sqrt(p_i) - \sqrt(q_i)) ** 2) = p_i - 2 \sqrt(p_i * q_i) + q_i
		sum_p = torch.sum(output.clone())
		sum_q = torch.numel(labels)
		sum_pq = -2 * torch.sum(torch.sqrt(torch.gather(output, 1, labels.view(-1).unsqueeze(-1))))
		hellinger_distance = (1 / torch.sqrt(torch.tensor(2.))) * torch.sqrt(sum_p + sum_pq + sum_q)
		return hellinger_distance.to(self.device)
	
	def _is_prob_dist(
		self,
		tensor: torch.FloatTensor,
		dim: int=-1,
		rtol: float=1e-3
	) -> Tuple[bool, Union[None, Dict[int, float]]]:
		"""
		Check whether the rows of `tensor` each sum to 1.

		Args:
			tensor: Input tensor to check
			dim: Dimension along which to sum
			rtol: Permitted relative tolerance when comparing sum of tensor elements to 1
		
		Returns:
			Whether `tensor` is a valid probability distribution, optionally which indices of `tensor` do not sum to 1 and what the sums are
		"""
		invalid = dict()
		sums = torch.sum(tensor, dim=dim, dtype=torch.float32).to(self.device)
		if torch.all(torch.isclose(sums, torch.tensor([1.], device=self.device), rtol=rtol)):
			return (True, None)
		for idx, sum in enumerate(sums):
			invalid[idx] = sum
		return (False, invalid)
	
	@staticmethod
	def _has_valid_dimensions(output: torch.FloatTensor, labels: torch.LongTensor) -> None:
		"""
		Ensure that `output` and `labels` have valid dimensions.

		Args:
			output: Model prediction
			labels: Ground truth distribution
		
		Raises:
			IndexError: If `labels` and `output` do not have consistent dimensions
		"""
		if output.size(0) != labels.view(-1).size(0): # expect `output` # rows = `labels` # elements
			raise IndexError(f'Expected number of labels (`labels` elements) to equal number of rows of `output`.\n\t`output`: {output.size(0)} rows\n\t`labels`: {labels.size(0)} elements')