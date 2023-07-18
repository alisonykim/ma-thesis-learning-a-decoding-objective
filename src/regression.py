#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# regression.py

"""
Learn a linear transformation of token attributes to a one-hot encoded corpus.
"""

import torch.nn as nn


class LinearRegression(nn.Module):
	"""
	Learn weights W^{T} for a linear mapping p_dec(.|y_{<t}) of token attributes f(.|y_{<t}) to a one-hot encoded corpus.

	The regression problem is defined as:
		p_dec(.|y_{<t}) = f(.|y_{<t}) W^{T} + b
	
	Args:
		num_attributes: Number of input features
		dim_out: Number of output features
		device: Device on which to load tensors
	
	Attributes:
		device: Device on which to load tensors
		linear: Linear layer
		softmax: Softmax to project output onto probability simplex
	"""

	def __init__(self, num_attributes: int, dim_out: int, device: str):
		super(LinearRegression, self).__init__()
		
		self.device = device

		self.linear = nn.Linear(num_attributes, dim_out, device=self.device)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		"""Compute forward pass."""
		output = self.linear(x)
		if len(output.size()) > 2:
			output = output.squeeze()
		p_dec = self.softmax(output)
		return p_dec.to(self.device)