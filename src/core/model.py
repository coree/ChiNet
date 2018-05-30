"""Base model class for Tensorflow-based model construction."""
from datasources import TextSource
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from .live_tester import LiveTester
from .time_manager import TimeManager
from .summary_manager import SummaryManager
from .checkpoint_manager import CheckpointManager
import logging
logger = logging.getLogger(__name__)


class BaseModel(object):
    """Base model class for Tensorflow-based model construction.

    This class assumes that there exist no other Tensorflow models defined.
    That is, any variable that exists in the Python session will be grabbed by the class.
    """

    def __init__(self,
                 tensorflow_session: tf.Session,
                 learning_schedule: List[Dict[str, Any]],
                 train_data: Dict[str, TextSource],
                 test_data: Dict[str, TextSource] = {},
                 test_losses_or_metrics: str = None):
        """Initialize model with data sources and parameters."""
        assert len(train_data) > 0
        self._tensorflow_session = tensorflow_session
        self._train_data = train_data
        self._test_data = test_data
        self._test_losses_or_metrics = test_losses_or_metrics
        self._initialized = False

        # Extract and keep known prefixes/scopes
        self._learning_schedule = learning_schedule
        self._known_prefixes = [schedule for schedule in learning_schedule]

        # Check consistency of given data sources
        train_data_sources = list(train_data.values())
        logger.debug('Data sources: {}'.format(train_data_sources))
        test_data_sources = list(test_data.values())
        self._batch_size = train_data_sources.pop().batch_size
        for data_source in train_data_sources + test_data_sources:
            if data_source.batch_size != self._batch_size:
                raise ValueError(('Data source "%s" has anomalous batch size of %d ' +
                                  'when detected batch size is %d.') % (data_source.short_name,
                                                                        data_source.batch_size,
                                                                        self._batch_size))

        # Register a manager for tf.Summary
        self.summary = SummaryManager(self)

        # Register a manager for checkpoints
        self.checkpoint = CheckpointManager(self)

        # Register a manager for timing related operations
        self.time = TimeManager(self)

        # Prepare for live (concurrent) validation/testing during training, on the CPU
        self._enable_live_testing = len(self._test_data) > 0
        self._tester = LiveTester(self, self._test_data)

        # Run-time parameters
        self.is_training = tf.placeholder(tf.bool)
        self.use_batch_statistics = tf.placeholder(tf.bool)

        self._build_all_models()

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        return self.__class__.__name__

    @property
    def output_path(self):
        """Path to store logs and model weights into."""
        return '%s/%s' % (os.path.abspath(os.path.dirname(__file__) + '/../../outputs'),
                          self.identifier)

    def _build_all_models(self):
        """Build training (GPU/CPU) and testing (CPU) streams."""
        self.output_tensors = {}
        self.loss_terms = {}
        self.metrics = {}
        self.inputs = {}

        def _build_datasource_summaries(data_sources, mode):
            """Register summary operations for input data from given data sources."""
            # Used only for images

        def _build_train_or_test(mode):
            data_sources = self._train_data if mode == 'train' else self._test_data

            # Build model
            output_tensors, loss_terms, metrics, inputs = self.build_model(data_sources, mode=mode)

            # Record important tensors
            self.output_tensors[mode] = output_tensors
            self.loss_terms[mode] = loss_terms
            self.metrics[mode] = metrics
            self.inputs[mode] = inputs
            # logger.debug(' [*] Model interfacers: \n - output : {}\ - loss_terms : {}\n - metrics : {}\n - inputs : {}'.format(
            #                                     self.output_tensors, self.loss_terms, self.metrics, self.inputs ))

            # Create summaries for scalars
            if mode == 'train':
                for name, loss_term in loss_terms.items():
                    self.summary.scalar('loss/%s/%s' % (mode, name), tf.squeeze(loss_term))  # Use tf.squeeze to solve dim error
                for name, metric in metrics.items():
                    self.summary.scalar('metric/%s/%s' % (mode, name), metric)

        # Build the main model
        _build_datasource_summaries(self._train_data, mode='train')
        _build_train_or_test(mode='train')
        logger.info('Built model for training.')

        # If there are any test data streams, build same model with different scope
        # Trainable parameters will be copied at test time
        if self._enable_live_testing:
            _build_datasource_summaries(self._test_data, mode='test')
            with tf.variable_scope('test'):
                _build_train_or_test(mode='test')
            logger.info('Built model for testing.')

            self._tester._post_model_build()  # Create copy ops to be run before every test run
        self.summary._post_model_build()  # Merge registered summary operations

    
    def build_model(self, data_sources: Dict[str, TextSource], mode: str):
        """Build model."""
        raise NotImplementedError('BaseModel::build_model is not yet implemented.')

    def initialize_if_not(self, training=False):
        """Initialize variables and begin preprocessing threads."""
        if self._initialized:
            return

        # Build supporting operations
        self.checkpoint.build_savers()  # Create savers
        if training:
            self._build_optimizers()
            # TODO Maybe put the batcher shuffle here

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        self._initialized = True

    #TODO will this be used?
    def _build_optimizers(self):
        """Based on learning schedule, create optimizer instances."""
        self._optimize_ops = []
        all_trainable_variables = tf.trainable_variables()
        logger.debug('All trainable variables : {}'.format(all_trainable_variables))
        for spec in self._learning_schedule:
            optimize_ops = []
            loss_terms = spec['loss_terms_to_optimize']
            assert isinstance(loss_terms, dict)
            for loss_term_key, prefixes in loss_terms.items():
                assert loss_term_key in self.loss_terms['train'].keys()
                logger.debug('Prefixes : {}'.format(prefixes))
                if prefixes == 'ALL':
                    logger.info('Training all trainable variables')
                    variables_to_train = [v for v in all_trainable_variables]
                else:
                    variables_to_train = []
                    for prefix in prefixes:
                        variables_to_train += [
                            v for v in all_trainable_variables
                            if v.name.startswith(prefix)
                        ]
                logger.debug(' Variables to train : {}'.format(variables_to_train))
                optimize_op = tf.train.AdamOptimizer(
                    learning_rate=spec['learning_rate'],
                    # beta1=0.9,
                    # beta2=0.999,
                ).minimize(
                    loss=self.loss_terms['train'][loss_term_key],
                    var_list=variables_to_train,
                    name='optimize_%s' % loss_term_key,
                )
                optimize_ops.append(optimize_op)
            self._optimize_ops.append(optimize_ops)
            logger.info('Built optimizer for: %s' % ', '.join(loss_terms.keys()))

    def load_embeddings(self):
        #load embeddings from file and call embedding matrix assign op
        pass

    #[GAN HACK pretrain generator to output target sentences from only noise TODO use also input sentences?] 
    def pretrain(self, num_epochs=None, num_steps=None):
        #Pretrain generator (repeat n_epochs)
        #generate sentence from random noise
        #sample target sentence
        #optimize generator loss using distance between target sentence and generated sentence (TODO what distance metric?)
        
        pass

    def train(self, num_epochs=None, num_steps=None):
        #Train discriminator and generator (repeat n_epochs)

        #Discriminator training (repeat n_step_d)
        #sample input sentences and target sentence
        #generate sentence from random noise and input sentences
        #[GAN HACK add noise to generated or target sentence later in training to increase performance]
        #compute discrimator scores for generated and target sentence
        #optimize discriminator loss using discriminator scores

        #Generator training (repeat n_step_g)
        #sample input sentences and target sentence
        #generate sentence from random noise and input sentences
        #compute discriminator score for generated sentence 
        #optimize generator loss using discriminator score [GAN HACK also use similarity to target sentence in loss]
        
        #[GAN HACK update n_step_d and n_step_g using discriminator and generator losses]

        pass

    def evaluate(self, data_source):
        #Predict correct story endings (repeat for every test data entry)

        #sample inputs sentences and two target sentences
        #compute discriminator scores for both target sentences
        #predict whether first or second target sentence was right sentence using discriminator scores
        #write scores to file #TODO how much of this to do here?

        pass
