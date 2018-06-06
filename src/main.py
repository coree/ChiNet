#!/usr/bin/env python3
import argparse

import coloredlogs
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)

from tensorflow.python import debug as tf_debug

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a 2D joint estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )
    logging.basicConfig(level=args.v.upper())

    # Initialize Tensorflow session
    tf.logging.set_verbosity(args.v.upper())
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess) # -> For debugging

    with sess as session:  # tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        logger.info('Initialize tensorflow session')
        # Declare some parameters
        batch_size = 32

        # Define model
        from datasources import TextSource
        from models import CGAN
        model = CGAN(
            # Tensorflow session
            # Note: The same session must be used for the model and the data sources.
            session,

            # The learning schedule describes in which order which part of the network should be
            # trained and with which learning rate.
            #
            # A standard network would have one entry (dict) in this argument where all model
            # parameters are optimized. To do this, you must specify which variables must be
            # optimized and this is done by specifying which prefixes to look for.
            # The prefixes are defined by using `tf.variable_scope`.
            #
            # The loss terms which can be specified depends on model specifications, specifically
            # the `loss_terms` output of `BaseModel::build_model`.
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'pretrain_loss': {'generator'},
                        'generator_loss': {'generator'},
                        'discriminator_loss': {'document', 'sentence', 'discriminator', 'attention'}
                    },
                    'metrics': ['kp_2D_mse'],
                    'learning_rate': 1e-3,
                }
            ],

            test_losses_or_metrics=['kp_2D_mse'],

            # Data sources for training and testing.
            train_data={
                'real': TextSource(
                    batch_size,
                    file_path='../datasets/train_stories.csv',
                ),
            },
            # test_data={
            #     'real': TextSource(
            #         1,
            #         file_path='../datasets/train_stories.csv',
            #     ),
            # },
        )

        model.load_embeddings(path="../datasets/word2vec/GoogleNews-vectors-negative300.bin", 
                              binary=True)

        #TODO pretrain
        model.pretrain(
            num_steps=0,
        )

        model.train(
            num_epochs=0,
        )

        model.evaluate(
            TextSource(
                    batch_size,
                    file_path='../datasets/cloze_test_val.csv',
                    testing=True,
                )
        )
