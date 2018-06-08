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
                 submission=False,
                 override_file=False,
                 vocab_size=20000):
        """Initializes data source and loads sentences from disk"""
        self._sentences_file = ''.join(os.path.splitext(file_path)[0]) + '.clean'
        self._data_file = ''.join(os.path.splitext(file_path)[0]) + '.processed'
        self.testing = testing
        self.submission = submission
        self.preprocessed_vocab = True if testing or submission else False
        self.vocab_size = vocab_size

        # If preprocessed file doesn't exit we create it
        if not os.path.exists(self._data_file) or override_file:
            self.preprocess(file_path, self._sentences_file)

        a_vocab_dict, _ = coree.load_vocab()
        if not len(a_vocab_dict) == self.vocab_size:
            logger.warning('Vocab size set to {} instead of {}'.format(len(a_vocab_dict), self.vocab_size))
            self.vocab_size = len(a_vocab_dict)
        # Open clean data file
        logger.info('Loading data from {}'.format(self._data_file))
        data = coree.load_preprocessed_data(self._data_file)
        
        self.len_data = len(data)  # Number of total data     
        self.batch_size = batch_size
        self._data = data
        logger.debug('Loaded data : [{}, {}, ...]'.format(self._data[0],self._data[0]))

        self._generate_batches()
        self.num_batches = len(self._batched_data)  # Number of batches
        logger.info('Text source initialize complete')
    
    def _pad_and_measure_sentences(self, data, max_sentence_length=50):
        batch_size = len(data)
        sentence_n = len(data[0])
        padded_sentences = np.zeros([batch_size, sentence_n, max_sentence_length])
        sentence_lengths = np.zeros([batch_size, sentence_n])
        for i in range(batch_size):
            for j in range(sentence_n):
                sentence = data[i][j]
                sentence_length = min(max_sentence_length, len(sentence))
                pad_symbol = 2 #TODO
                padded_sentences[i,j] = sentence[:sentence_length] + [pad_symbol]*(max_sentence_length-sentence_length)
                if len(sentence) > max_sentence_length:
                    sentence[max_sentence_length] = 1 # 1 for 'eos' symbol
                sentence_lengths[i,j] = sentence_length
        return padded_sentences, sentence_lengths

    def _shuffle(self):
        return [self._data[i] for i in permutation(self.len_data)]

    def _generate_batches(self):
        logger.debug('Generating new data batches')
        if self.testing or self.submission:
            shuffled_data = self._data[:]
            self._batched_data = []
            for i in range(self.len_data//self.batch_size):
                self._batched_data.append(shuffled_data[self.batch_size*i:self.batch_size*(i+1)])
            difference = self.batch_size - (self.len_data % self.batch_size)
            last_elem = shuffled_data[self.batch_size*(i+1):] + [[[0] for _ in range(len(shuffled_data[0]))]]*difference
            self._batched_data.append(last_elem)
            logger.debug('Returning test batch')
        else:
            shuffled_data = self._shuffle()
            self._batched_data = []
            for i in range(self.len_data//self.batch_size):
                self._batched_data.append(shuffled_data[self.batch_size*i:self.batch_size*(i+1)])
            if self.len_data % self.batch_size != 0:
                logger.info("Number of entries is not multiple of batch size. {} stories have been not included in this epoch".format(self.len_data%self.batch_size))

    def get_batch(self):
        if not self._batched_data:
            self._generate_batches()
        logging.debug("Current numeber of batches: {}".format(len(self._batched_data)))
        data = self._batched_data.pop(0)
        return self._pad_and_measure_sentences(data)

    def preprocess(self, file_path, clean_file):
        """ Whole preprocess pipeline data"""
        # Preprocess file
        coree.preprocess_file(file_path, clean_file, self.testing, self.submission)
        logger.info('Clean file {} created'.format(clean_file))
        logger.info('Started preprocessing data...')
        # Preprocess data
        coree.preprocess_data(clean_file, self._data_file, self.vocab_size, use_prevocab=self.preprocessed_vocab)
        logger.info('Data preprocessing and vocabulary creation complete.')

    def shape(self):
        # In principle our placeholders are [None, None], so just leavin' this here
        pass
