# Set GPU
from typing import List, Tuple

import numpy as np
import pickle

import sys

from random import choice


class Logger(object):
    def __init__(self, filename: str, mode: str = "a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
