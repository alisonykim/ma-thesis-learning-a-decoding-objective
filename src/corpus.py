#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# corpus.py

"""
Create corpus class to download/assemble, preprocess, and prepare texts for batching.
"""

import re
from typing import Dict, List

from torch.utils.data import Dataset


class Wikitext(Dataset):
	"""
	Compile corpus of input and label texts to be used for learning weights.
	
	Args:
		corpus_path: Path to saved corpus text file
		max_num_texts: Maximum number of texts to comprise corpus
		min_seq_len: Minimum length of text to be included in corpus
	
	Attributes:
		dataset: Corpus of texts from which to calculate token attributes
		max_num_texts: Maximum number of texts to comprise corpus
		min_seq_len: Minimum length of text
	"""

	def __init__(
		self,
		corpus_path: str,
		max_num_texts: int=6000,
		min_seq_len: int=30
	):
		try:
			with open(corpus_path, mode='r', encoding='utf-8') as f_corpus:
				corpus = [text.rstrip() for text in f_corpus]
			self.input_texts = self.filter_texts(corpus, min_seq_len)[:max_num_texts]
			self.label_texts = self.input_texts
		except FileNotFoundError:
			raise FileNotFoundError(f'You entered a nonexistent corpus filepath. Please try again.')
		except IndexError:
			raise IndexError(f'You entered too high a value for `max_num_texts`. Please try again with a smaller integer value.')
	
	@staticmethod
	def filter_texts(corpus: List[str], min_seq_len: int, separator: str=' ') -> List[str]:
		"""Filter out texts that are too short and/or contain too few alphabetic characters."""
		filtered_corpus = list()
		for i in range(len(corpus)):
			if len(corpus[i].split(' ')) < min_seq_len: # Filter out short texts
				continue
			if re.search(r'( =)+', corpus[i]): # Filter out titles and section headers
				continue
			if len(re.findall(r'<unk>', corpus[i], flags=re.IGNORECASE)) > 1: # Filter out texts with >1 <unk>
				continue
			if len([token for token in corpus[i].split(separator) if token.isalpha()]) > 15: # Not just numbers and symbols
				filtered_corpus.append(corpus[i])
		return filtered_corpus
		
	def __len__(self) -> int:
		"""Return length of array."""
		return self.max_num_texts
	
	def __getitem__(self, idx: int ) -> Dict[str, List[str]]:
		"""Get input and label IDs."""
		return {'input_text': self.input_texts[idx], 'label_text': self.label_texts[idx]}