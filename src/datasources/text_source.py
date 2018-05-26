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

import util.preprocessor as coree

class TextSource(object):
    def __init__(self,
                 batch_size: int,
                 file_path: int ='../datasets/train_stories.csv',
                 testing=False,
                 override_file=False,
                 vocab_size=2e5):
        """Initializes data source and loads sentences from disk"""
        self._sentences_file = ''.join(os.path.splitext(file_path)[0]) + '.clean'
        self._data_file = ''.join(os.path.splitext(file_path)[0]) + '.processed'
        
        self._vocab_size = vocab_size

        # If preprocessed file doesn't exit we create it
        if not os.path.exists(self._data_file) or override_file:
            self.preprocess(file_path, self._sentences_file)

        # Open clean data file
        logger.info('Loading data from {}'.format(self._data_file))
        data = coree.load_preprocessed_data(self._data_file)
        
        self.len_data = len(data)  # Number of total data     
        self._batch_size = batch_size
        self._data = data
        logger.debug('Loaded data : [{}, {}, ...]'.format(self._data[0],self._data[0]))

        self._generate_batches()
        self.num_batches = len(self._batched_data)  # Number of batches
        logger.info('Text source initialize complete')
        
    def _separate_sentences(self, data):
        inputs = []
        for i in range(4):
            inputs.append(np.array([s[i] for s in data]))
        inputs = tuple(inputs)
        # inputs  = [[s[k] for k in range(4)]  for s in data]  # First 4 sentences as input
        outputs = [s[4] for s in data]                       # Last (5th) sentence as output
        return inputs, outputs

    def _shuffle(self):
        return [self._data[i] for i in permutation(self.len_data)]

    def _generate_batches(self):
        logger.debug('Generating new data batches')
        shuffled_data = self._shuffle()
        self._batched_data = []
        for i in range(self.len_data//self._batch_size):
            self._batched_data.append(shuffled_data[self._batch_size*i:self._batch_size*(i+1)])
        if self.len_data%self._batch_size != 0:
            logger.info("Number of entries is not multiple of batch size. {} hisotries have been not included in this epoch".format(self.len_data%self._batch_size))

    def get_batch(self):
        if not self._batched_data:
            self._generate_batches()
        logging.debug("Current numeber of batches: {}".format(len(self._batched_data)))
        return self._separate_sentences(self._batched_data.pop())

    def preprocess(self, file_path, clean_file):
        """ Whole preprocess pipeline data"""
        # Preprocess file
        coree.preprocess_file(file_path, clean_file)
        logger.info('Clean file {} created'.format(clean_file))
        logger.info('Started preprocessing data...')
        # Preprocess data
        coree.preprocess_data(clean_file, self._vocab_size)
        logger.info('Data preprocessing and vocabulary creation complete.')

    def shape(self):
        # In principle our placeholders are [None, None], so just leavin' this here
        pass