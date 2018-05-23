"""MnistNet architecture."""

from typing import Dict

import tensorflow as tf
from core import BaseDataSource, BaseModel
from .hourglass_module import hourglass, residual # I deleted my helper files (ping me if you need anything (tho you shouldn't))
from util.gaussian import gaussian_maps


import logging
logger = logging.getLogger(__name__)

#TODO modify into GAN, for now just discriminator
class GAN(BaseModel):
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        logger.info('Start building model {}'.format(__name__))
        
        #logging.debug(targets)

        #parameters TODO config from higher level
        sentence_hidden_size = 64
        document_hidden_size = 128
        document_n_hidden = 4
        embedding_size = 100
        vocab_size = 20000
        batch_size = 32
        initializer = tf.contrib.layers.xavier_initializer

        #get inputs
        inputs = input_tensors['input']    # Inputs
        target = input_tensors['target']  # Targets
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors  # Data source automatically handles the datafiles

        #embedding
        with tf.variable_scope('embed'):
            embedding_weights = tf.get_variable("weights", shape=[vocab_size, embedding_size], initializer=tf.constant_initializer([vocab_size, embedding_size]), trainable=False)
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            embedding_init = embedding_weights.assign(embedding_placeholder)
            
            #TODO fix sentence length? how to deal with variable size sentence length?
            embedded_inputs = []
            for input_ in inputs:
                embedded_inputs += [tf.nn.embedding_lookup(embedding_weights, input_)]
            embedded_target = tf.nn.embedding_lookup(embedding_weights, target)

        #sentence RNN
        with tf.variable_scope('sentence'):
            #stack inputs and targets
            embedded_inputs_target = tf.concat([embedded_inputs, embedded_target], axis=1)

            sentence_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=sentence_hidden_size, activation=tf.nn.relu) #TODO use tanh (default) instead of relu?
            sentence_output, sentence_state = tf.nn.dynamic_rnn(sentence_rnn_cell, embedded_inputs_target)

            rnn_state_inputs = sentence_state[:document_n_hidden]
            rnn_state_target = sentence_state[document_n_hidden]

        #TODO attention
        
        #document RNN
        with tf.variable_scope('document'):
            document_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=document_hidden_size, activation=tf.nn.relu) #TODO activation: same as sentence
            document_output, document_state = tf.nn.dynamic_rnn(document_rnn_cell, rnn_state_inputs)
            #document_to_sentence_weights = tf.get_variable("document_to_sentence_weigths", shape=[document_hidden_size, sentence_hidden_size], initializer=initializer, trainable=True)
            document_sentence_space = tf.layers_dense(document_state, document_hidden_size) #TODO activation? no?
            score = tf.reduce_sum(tf.multiply(document_sentence_space, rnn_state_target))
            score_probability = tf.sigmoid(score)
    
        #TODO loss

        #BELOW SHOULD BE DELETED, KEPT FOR REFERENCE
        filters = 256
        num_joints = 21  # TODO: Why can't I use "tf.shape(y[2])"?
        hourglass_stacks = 8
        decrease_factor = 2
        logger.debug('Inputs - {} \nTargets - {}'.format(x,y))
        y = y / decrease_factor  # TODO: Change the way we rescale things
        outs = []

        with tf.variable_scope('pre'):
            x = tf.layers.conv2d(x, filters=filters/4, kernel_size=7, strides=decrease_factor,
                                 padding='same', data_format='channels_first')
            x = tf.nn.relu(tf.layers.batch_normalization(x))

            x = residual(x, filters/2)
            x = residual(x, filters)

        for i in range(hourglass_stacks):
            with tf.variable_scope('hourglass_{}'.format(i)):
                x = hourglass(x)

                r = residual(x, filters)
                logger.debug(r)
                r = tf.layers.conv2d(r, filters, 1, data_format='channels_first')
                logger.debug(r)
                r = tf.nn.relu(tf.layers.batch_normalization(x))
                logger.debug(r)
                # compute block loss with o
                o = tf.layers.conv2d(r, num_joints, 1, data_format='channels_first')
                logger.debug('{} - Stack output'.format(o))
                outs.append(o)
                
                if i < hourglass_stacks:
                    join1 = tf.layers.conv2d(o, filters, 1, data_format='channels_first')
                    logger.debug('{} - Output branch'.format(join1))
                    join2 = tf.layers.conv2d(r, filters, 1, data_format='channels_first')
                    logger.debug('{} - Main branck'.format(join2))
                    x = join1 + join2
                    logger.debug('{} - Juntion (Add branches)'.format(x))        
        
        # Convert y (target) coords (x_T, y_T) to 2D distribution map
        y = tf.map_fn(gaussian_maps, tf.cast(y, tf.float32))  # Are they ints or floats?
        logger.debug(' ---- Loss ----\n Targets - {}\n Preds - {}'.format(y, outs))
        
        # Loss
        loss_terms = {}
        for idx, o in enumerate(outs):
            loss_terms['map_mse_{}'.format(idx)] = tf.losses.mean_squared_error(o, y) 
            
        # Define outputs
        # loss_terms = {  # To optimize  --> Original snippet. 
            # 'kp_2D_mse': (tf.reduce_mean(outs))
                            # #tf.reduce_mean(tf.squared_difference(x, y)),
        # }
        logger.info('Model {} building exiting.'.format(__name__))
        return {'kp_2D': outs[-1]}, loss_terms, {}
