#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)


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
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        logger.info('Initialize tensorflow session')
        # Declare some parameters
        batch_size = 1

        # Define model
        from datasources import TextSource
        from models import GAN
        model = GAN(
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
            learning_schedule=[   # TODO: Implemet this in a nice and global way
                {
                    'loss_terms_to_optimize': {
                        'sigmoid_cross_entropy' : 'ALL'
                    },
                    'metrics': ['kp_2D_mse'],
                    'learning_rate': 1e-4,
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
            # If you want to validate your model, split the training set into
            # training and validation and uncomment this line
            # test_data={
            #     'real': HDF5Source(
            #         session,
            #         batch_size,
            #         hdf_path='../datasets/new_data.h5',  # TODO -> Do we need a different file?
            #         keys_to_use=['test'],
            #         testing=True,
            #     ),
            # },
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=50,
        )
