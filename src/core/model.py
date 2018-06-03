"""Base model class for Tensorflow-based model construction."""
from datasources import TextSource
from util.preprocessor import load_vocab
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from gensim import models

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
        self.embedding_assign_op = {}

        def _build_datasource_summaries(data_sources, mode):
            """Register summary operations for input data from given data sources."""
            # Used only for images

        def _build_train_or_test(mode):
            data_sources = self._train_data if mode == 'train' else self._test_data

            # Build model
            raw_outputs, loss_terms, metrics, inputs = self.build_model(data_sources, mode=mode)
            output_tensors = raw_outputs['predicted_ending']  # TODO Cahnge to all 

            # Record important tensors
            self.output_tensors[mode] = output_tensors
            self.embedding_assign_op[mode] = raw_outputs['embedding_assign_op']
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

    #load embeddings from file and call embedding matrix assign op
    def load_embeddings(self, path, vocab_size=25000, binary=True):  #TODO get vocab_size and embedding_size params from elsewhere?

        logger.info("Loading external embeddings from %s" % path)


        _, vocab = load_vocab()  # Retrieve just word id
        if not len(vocab) == vocab_size:
            logger.warning(' Asked for a vocabulary size of {0} but vocabulary file has {1} entries. Setting vocab_size to {1}'.format(vocab_size, len(vocab)))
            vocab_size = len(vocab)
        model = models.KeyedVectors.load_word2vec_format(path, binary=binary)  

        embedding_size = model.vector_size  # Embedding size given by the loaded module
        logger.info("Embedding size of {}".format)

        external_embedding = np.zeros(shape=(vocab_size, embedding_size))

        non_included = []
        for tok, idx in vocab.items():
            if tok in model.vocab:
                external_embedding[idx] = model[tok]
            else:
                non_included.append(tok)
                external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=embedding_size)

        if not non_included:
            logger.debug("Tokens not included: {}".format(non_included))
            logger.warning('{} of {} tokens were not found in embedding fike'.format(len(non_included), vocab_size))

        # TODO also assign embedding start and stop words, perhaps also stop word error bound 

        fetches = {}
        fetches['output_tensors'] = self.embedding_assign_op['train']  # ["embeding_assign_op"]

        feed_dict = {self.inputs['train']['word2vec_weights']: external_embedding}

        outcome = self._tensorflow_session.run(
            fetches=fetches,
            feed_dict=feed_dict
        )

    #[GAN HACK pretrain generator to output target sentences from only noise TODO use also input sentences?] 
    def pretrain(self, num_epochs=None, num_steps=None):
        #TODO how does weight saving work for pretraining?

        #Pseudocode

        #Pretrain generator (repeat n_epochs)
        #generate sentence from random noise
        #sample target sentence
        #optimize generator loss using distance between target sentence and generated sentence (TODO what distance metric?)
        
        if num_steps is None:
            num_batches = [s.num_batches for s in list(self._train_data.values())][0]  # TODO : make it cleaner
            num_steps = int(num_epochs * num_batches)
        self.initialize_if_not(training=True)

        initial_step = self.checkpoint.load_all()
        current_step = initial_step
        
        logger.info(' * Number of steps: {}'.format(num_steps)) 
        for current_step in range(initial_step, num_steps):

            sentences, sentence_lengths = self._train_data['real'].get_batch()
            feed_dict = dict()
            feed_dict[self.inputs['train']['extra_sentence']] = sentences[:,-1] #only target sentence
            feed_dict[self.inputs['train']['extra_sentence_length']] = sentence_lengths[:,-1]
            feed_dict[self.is_training] = True
            feed_dict[self.use_batch_statistics] = True

            logger.debug(feed_dict)

            fetches = {}
            fetches['loss_terms'] = self.output_tensors['train']["pretrain_generator_loss"] #TODO correct?

            summary_ops = self.summary.get_ops(mode='train')
            if len(summary_ops) > 0:
                fetches['summaries'] = summary_ops

            outcome = self._tensorflow_session.run(
                fetches=self.output_tensors['train']["pretrain_generator_loss"],
                feed_dict=feed_dict
            )
            
            self.time.end('train_iteration')

            # Print progress
            to_print = '%07d> ' % current_step
            # to_print += ', '.join(['%s = %f' % (k, v)
            #                        for k, v in zip(loss_term_keys, outcome['loss_terms'])])
            self.time.log_every('train_iteration', to_print, seconds=2)

            # Trigger copy weights & concurrent testing (if not already running)
            if self._enable_live_testing:
                self._tester.trigger_test_if_not_testing(current_step)

            # Write summaries
            if 'summaries' in outcome:
                self.summary.write_summaries(outcome['summaries'], current_step)

            # Save model weights
            if self.time.has_been_n_seconds_since_last('save_weights', 600):
                self.checkpoint.save_all(current_step)

        # Save final weights
        self.checkpoint.save_all(current_step)

    def train(self, num_epochs=None, num_steps=None):
        #Psudocode

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

        if num_steps is None:
            num_batches = [s.num_batches for s in list(self._train_data.values())][0]  # TODO : make it cleaner
            num_steps = int(num_epochs * num_batches)
        self.initialize_if_not(training=True)

        initial_step = self.checkpoint.load_all()
        current_step = initial_step

        num_steps_discriminator = 20
        num_steps_generator = 20
        
        logger.info(' * Number of steps: {}'.format(num_steps)) 
        for current_step in range(initial_step, num_steps):

            for substep in range(num_steps_discriminator):
                fetches = {}
                fetches['loss_terms'] = self.loss_terms['train']['discriminator_loss']
                summary_ops = self.summary.get_ops(mode='train')  # TODO Temporal fix
                if len(summary_ops) > 0:
                    fetches['summaries'] = summary_ops
                
                sentences, sentence_lengths = self._train_data['real'].get_batch()
                feed_dict = dict()
                feed_dict[self.inputs['train']['sentences']] = sentences
                feed_dict[self.inputs['train']['sentence_lengths']] = sentence_lengths
                feed_dict[self.is_training] = True
                feed_dict[self.use_batch_statistics] = True

                outcome = self._tensorflow_session.run(
                    fetches=fetches,
                    feed_dict=feed_dict
                )
            
            for substep in range(num_steps_generator):
                fetches = {}
                fetches['loss_terms'] = self.loss_terms['train']['generator_loss']
                summary_ops = self.summary.get_ops(mode='train')  # TODO Temporal fix
                if len(summary_ops) > 0:
                    fetches['summaries'] = summary_ops

                sentences, sentence_lengths = self._train_data['real'].get_batch()
                feed_dict = dict()
                feed_dict[self.inputs['train']['sentences']] = sentences
                feed_dict[self.inputs['train']['sentence_lengths']] = sentence_lengths
                feed_dict[self.is_training] = True
                feed_dict[self.use_batch_statistics] = True

                outcome = self._tensorflow_session.run(
                    fetches=fetches,
                    feed_dict=feed_dict
                )

            #TODO how do I get losses here?
            #update num_steps_discriminator and num_steps_generator
 
            self.time.end('train_iteration')

            # Print progress
            to_print = '%07d> ' % current_step
            to_print += ', '.join(['%s = %f' % (k, v)
                                   for k, v in zip(loss_term_keys, outcome['loss_terms'])])
            self.time.log_every('train_iteration', to_print, seconds=2)

            # Trigger copy weights & concurrent testing (if not already running)
            if self._enable_live_testing:
                self._tester.trigger_test_if_not_testing(current_step)

            # Write summaries
            if 'summaries' in outcome:
                self.summary.write_summaries(outcome['summaries'], current_step)

            # Save model weights
            if self.time.has_been_n_seconds_since_last('save_weights', 600):
                self.checkpoint.save_all(current_step)

        # Save final weights
        self.checkpoint.save_all(current_step)

    def evaluate(self, data_source):
        #Pseudocode

        #Predict correct story endings (repeat for every test data entry)

        #sample inputs sentences and two target sentences
        #compute discriminator scores for both target sentences
        #predict whether first or second target sentence was right sentence using discriminator scores
        #write scores to file #TODO how much of this to do here?

        #fetches['output_tensors'] = ['predicted_ending']
        #feed_dict[self.inputs['test']['sentences']]
        #feed_dict[self.inputs['test']['sentence_lengths']]
        #feed_dict[self.inputs['test']['extra_sentence']]
        #feed_dict[self.inputs['test']['extra_sentence_length']]

        pass
