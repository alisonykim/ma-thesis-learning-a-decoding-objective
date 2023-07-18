#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# logger.py

"""
Print stdout and exceptions to console and copy them to a log file.

Inspired by: https://stackoverflow.com/a/14906787
"""

import sys
import traceback


class Logger(object):
	"""Context manager that copies stdout and any exceptions to a log file."""
	def __init__(self, log_filename):
		self.file = open(log_filename, 'w', encoding='utf-8')
		self.stdout = sys.stdout

	def __enter__(self):
		sys.stdout = self

	def __exit__(self, exc_type):
		sys.stdout = self.stdout
		if exc_type is not None:
			self.file.write(traceback.format_exc())
		self.file.close()

	def write(self, data):
		self.file.write(data)
		self.stdout.write(data)

	def flush(self):
		self.file.flush()
		self.stdout.flush()