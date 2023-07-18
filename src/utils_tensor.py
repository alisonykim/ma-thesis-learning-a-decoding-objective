#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils_tensor.py

"""
Utility functions for tensor manipulation
"""

from typing import List, Union

import torch
import torch.nn.functional as F


def pad_list_of_tensors(
	tensor_list: List[torch.Tensor],
	pad_length: int,
	pad_value: Union[float, int],
	device: Union[torch.device, str],
	dim: int=0,
	long: bool=True,
	stack: bool=False
) -> Union[List[torch.Tensor], torch.Tensor]:
	"""
	Pad list of 1-D tensors to same length.

	Args:
		tensor_list: List of tensors
		pad_length: Desired length of tensor
		pad_value: Value of padding element
		device: Device onto which to load tensor
		dim: Dimension along which to stack tensors, if `stack`
		long: Whether padded tensors should be of dtype `long`
		stack: Whether to stack tensors
	
	Returns:
		Original tensors with padding, either as a list or a stacked tensor
	
	Example:
	>>> a = [torch.rand(5), torch.rand(4), torch.rand(2), torch.rand(4)]
	>>> a
	[tensor([0.6147, 0.6905, 0.3839, 0.4012, 0.0659]), tensor([0.2989, 0.0632, 0.7404, 0.8137]), tensor([0.9474, 0.3757]), tensor([0.7105, 0.8811, 0.3346, 0.9999])]
	>>> pad_length = 8
	>>> pad_value = 2.0
	>>> device = 'cpu'
	>>> long = False
	>>> stack = True
	>>> pad_list_of_tensors(a, pad_value, device, long=long, stack=stack)
	tensor([[0.6147, 0.6905, 0.3839, 0.4012, 0.0659, 2.0000, 2.0000, 2.0000],
		[0.2989, 0.0632, 0.7404, 0.8137, 2.0000, 2.0000, 2.0000, 2.0000],
		[0.9474, 0.3757, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000],
		[0.7105, 0.8811, 0.3346, 0.9999, 2.0000, 2.0000, 2.0000, 2.0000]])
	"""
	assert len(tensor_list) >= 1 # At least 1 tensor to pad
	tensor_list = [tensor_list[i].squeeze() for i in range(len(tensor_list))] # Must be 1-D
	
	for i in range(len(tensor_list)):
		num_to_fill = pad_length - len(tensor_list[i])
		assert num_to_fill >= 0 # Can't pad longer than `pad_value` positions
		tensor_list[i] = F.pad(tensor_list[i], (0, num_to_fill), value=pad_value).to(device)
		if long:
			tensor_list[i] = tensor_list[i].long()
		assert tensor_list[i].size(0) == pad_length
	
	if stack:
		return torch.stack(tensor_list, dim=dim)
	return tensor_list