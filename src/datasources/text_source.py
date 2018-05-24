from threading import Lock
from typing import List

import pickle
import os
import csv
import numpy as np
import tensorflow as tf
from numpy.random import permutation

import logging
logger = logging.getLogger(__name__)

from core import BaseDataSource

class TextSource(object):
    def __init__(self,
                 batch_size: int,
                 file_path: str,
                 testing=False,
                 override_file=False):
        """Initializes data source and loads sentences from disk"""
        clean_file = ''.join(os.path.splitext(file_path)[0]) + '.neat'
        # If clean pickled file doesn't exit we create it
        if not os.path.exists(clean_file) or override_file:
            logger.info('Clean file {} generated'.format(clean_file))
            self._preprocess_file(file_path, clean_file)
        # Open clean data file
        with open(clean_file, 'rb') as f:
            logger.info('Loading data from {}'.format(clean_file))
            data = pickle.load(f)
 
        self._len = len(data)
        self._batch_size = batch_size
        self._data = data

        self._batched_data = self._generate_batches()

        
    def _separate_sentences(self, data):
        inputs  = [[s[k] for k in range(4)]  for s in data]  # First 4 sentences as input
        outputs = [s[4] for s in data]                       # Last (5th) sentence as output
        return inputs, outputs

    def _shuffle(self):
        return [self._data[i] for i in permutation(self._len)]

    def _generate_batches(self):
        shuffled_data = self._shuffle()
        self._batched_data = []
        for i in range(self._len//self._batch_size):
            self._batched_data.append(shuffled_data[self._batch_size*i:self._batch_size*(i+1)])
        
        if self._len%self._batch_size != 0:
            logger.info("Number of entries is not multiple of batch size. {} hisotries have been not included in this epoch".format(self._len%self._batch_size))

    def get_batch(self):
        if not self._batched_data:
            self._generate_batches()
        logging.debug("Current numeber of batches: {}".format(len(self._batched_data)))
        return self._separate_sentences(self._batched_data.pop())

    def _preprocess_file(self, file_path, clean_file):
            with open(file_path, 'r') as f:
                data = list(csv.DictReader(f))
            # Just grab sentences from 1 to 5 and save them in a new file
            data = [[s[k] for k in ['sentence{}'.format(i+1) for i in range(5)]]  for s in data] 
            with open(clean_file, 'wb') as f:
                data = pickle.dump(data, f)

    def preprocess_data(self):
        # Whatever preprocess we may do to the data
        pass

    def shape(self):
        # In principle our placeholders are [None, None], so just leavin' this here
        pass