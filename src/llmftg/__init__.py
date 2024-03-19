'''
Python package showing the usage of Large Language Models (LLMs) for text/code generative purposes
using PyTorch and Hugging Face.

Assignment 1.3 for CS 6263: Natural Language Processing
'''

import logging

from rich.logging import RichHandler

# Configure logger
logging.basicConfig(
    level=logging.NOTSET, format='%(message)s', datefmt='[%X]',
    handlers=[RichHandler()]
)
logger = logging.getLogger('rich')
