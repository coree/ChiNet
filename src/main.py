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
        batch_size = 32

        # Define model
        from datasources import HDF5Source
        from models import StackedHourglass
        model = StackedHourglass(
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
                        'map_mse_0': ['pre', 'hourglass_0'],
                        'map_mse_1': ['pre', 'hourglass_0', 'hourglass_1'],
                        'map_mse_2': ['pre', 'hourglass_0', 'hourglass_1', 'hourglass_2'],
                        'map_mse_3': ['pre', 'hourglass_0', 'hourglass_1', 'hourglass_2', 'hourglass_3'],
                        'map_mse_4': ['pre', 'hourglass_0', 'hourglass_1', 'hourglass_2', 'hourglass_3', 'hourglass_4'],
                        'map_mse_5': ['pre', 'hourglass_0', 'hourglass_1', 'hourglass_2', 'hourglass_3', 'hourglass_4', 'hourglass_5'],
                        'map_mse_6': ['pre', 'hourglass_0', 'hourglass_1', 'hourglass_2', 'hourglass_3', 'hourglass_4', 'hourglass_5', 'hourglass_6'],
                        'map_mse_7': ['pre', 'hourglass_0', 'hourglass_1', 'hourglass_2', 'hourglass_3', 'hourglass_4', 'hourglass_5', 'hourglass_6', 'hourglass_7'],
                        # TODO: I hate this thing up here, have to automate this -> Edit: Not sure yet it works. Edit 2: No, I'm sure it's wrong.
                        #       That must not be right, I mean, yeah, sure, intermediate supervision or whatever, but no. That's not how the hell
                        #       this is intendet to work, right? I mean, sure by steps maybe, but all just throuwn there I'm sure crashes any
                        #       GPU. @Francesco we have to check this for sure (I'll open an issue as soon as I finis MoC (i.e. never))
                        # TODO: NOT GET BANNED FROM THE CLUSTER/ NOT CRASH ALL EUROPE AZURE SERVICES
                    },
                    'metrics': ['kp_2D_mse'],
                    'learning_rate': 1e-4,
                }
            ],
            
            test_losses_or_metrics=['kp_2D_mse'],

            # Data sources for training and testing.
            train_data={
                'real': HDF5Source(
                    session,
                    batch_size,
                    hdf_path='../datasets/new_data.h5',
                    keys_to_use=['train'],
                    min_after_dequeue=2000,
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

        # Test for Kaggle submission
        model.evaluate_for_kaggle(
            HDF5Source(
                session,
                batch_size,
                hdf_path='../datasets/testing.h5',
                keys_to_use=['test'],
                testing=True,
            )
        )
