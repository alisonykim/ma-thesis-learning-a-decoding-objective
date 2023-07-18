#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

"""Constants."""


ATTRIBUTES = [
	'probs',
	'unigram_freq',
	'diff',
	'abs_diff',
	'sq_diff',
	'top_k'
]


DECODING_STRATEGIES = {
	'learned': ['learned', 'learned weights'],
	'top_k': ['top_k', 'top-k', 'top k'],
	'temp': ['temperature', 'temp'],
    'top_p': ['top_p', 'top-p', 'top p', 'nucleus', 'nuc'],
    'typical': ['typical', 'typical-p', 'typical_p', 'typical p'],
    'beam': ['beam', 'beam search', 'beam_search']
}