"""
A python library for generating different permutations of audible segments from audio files. 
"""

__author__ = "Zabir Al Nazi"
__copyright__ = "Copyright 2021"
__credits__ = []
__license__ = "MIT"
__version__ = "0.0.3"
__maintainer__ = "https://github.com/zabir-nabil"
__status__ = "Production"

from .audioperm import read_audio, word_segments, permutations, fixed_len_segments
from .audioperm import AudioPerm