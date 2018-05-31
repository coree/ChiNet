#GAN architecture

import logging
import tensorflow as tf
from core import BaseModel
from datasources import TextSource

from typing import Dict

logger = logging.getLogger(__name__)

# outputs: dict with score 'score', dict with loss 'sigmoid_cross_entropy', empty dict

# required placeholder tensors in input_tensors dict (except 'embedding_weights', all should have batch dimension first): 
# 'input_sentence': 4 input sentences in vocab index format
# 'target_sentence': 1 target sentence in vocab index format
# 'target_label': 1 target label in int format
# 'embedding_weights': word2vec embedding matrix #TODO pass in other way

#TODO modify into GAN, for now just discriminator
class GAN(BaseModel):
    def build_model(self, data_sources: Dict[str, TextSource], mode: str):
        
        logger.info('Start building model {}'.format(__name__))

        #parameters TODO config from higher level
        sentence_hidden_size = 64
        document_hidden_size = 128
        embedding_size = 100
        max_sentence_length = 50
        input_sentence_n = 4
        document_n_hidden = input_sentence_n
        vocab_size = data_sources['real'].vocab_size 
        batch_size = data_sources['real'].batch_size 
        initializer = tf.contrib.layers.xavier_initializer
        rnn_activation = tf.nn.relu #TODO use tanh (default)?

        #inputs
        sentences = tf.placeholder(shape=[batch_size, input_sentence_n+1, max_sentence_length], dtype=tf.int64, name='sentences')
        sentence_lengths = tf.placeholder(shape=[batch_size, input_sentence_n+1], dtype=tf.int32, name='sentence_lengths')
        #for pretraining generator and evaluation where we have either only a target sentence or an extra target sentence
        extra_target_sentence = tf.placeholder(shape=[batch_size, max_sentence_length], dtype=tf.int64, name='extra_target_sentence')
        extra_target_sentence_length = tf.placeholder(shape=[batch_size], dtype=tf.int32, name="extra_target_sentence_length")

        word2vec_weights = tf.placeholder(shape=[vocab_size, embedding_size], dtype=tf.float32)  # input_tensors['embedding_weights'] #loaded word2vec embedding weights

        #embedding
        with tf.variable_scope('embedding'):
             embedding_weights = tf.get_variable("weights", shape=[vocab_size, embedding_size], initializer=tf.constant_initializer([vocab_size, embedding_size]), trainable=False)
            assign_embedding_op = embedding_weights.assign(word2vec_weights) #embedding_init must be called to load embedding weights
            
        #     embedded_inputs = []
        #     for sentence in input_sentences:
        #         embedded_inputs += [tf.nn.embedding_lookup(embedding_weights, sentence)]
        #     embedded_target = tf.nn.embedding_lookup(embedding_weights, target_sentence)
        #     #embedded sentence shape [batch_size, embedding_size]

        embedded_sentences = tf.one_hot(sentences, vocab_size)
        embedded_extra_target_sentence = tf.one_hot(sentences, vocab_size)

        #sentence rnn
        with tf.variable_scope('sentence'):
            #define rnn cell type
            sentence_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=sentence_hidden_size, activation=rnn_activation) 



            #apply rnn to each sentence
            sentence_states = []
            for i in range(input_sentence_n+1):
                sentence_output, sentence_state = tf.nn.dynamic_rnn(cell=sentence_rnn_cell, inputs=embedded_sentences[:,i], sequence_length=sentence_lengths[:,i], dtype=tf.float32)
                sentence_states += [sentence_state]
            
            #separate sentence rnn final hidden states for inputs and target
            rnn_state_inputs = tf.stack(sentence_states[:input_sentence_n], axis=1) #shape [batch_size, input_sentence_n, document_hidden_size]
            rnn_state_target = sentence_states[input_sentence_n] # shape [batch_size, document_hidden_size]
            #extra target sentence
            rnn_state_extra_target = tf.nn.dynamic_rnn(cell=sentence_rnn_cell, inputs=embedded_extra_target_sentence, sequence_length=extra_target_sentence_length, dtype=tf.float32)

        #TODO attention
        rnn_state_inputs_attention = rnn_state_inputs
        
        #document RNN
        with tf.variable_scope('document'):
            document_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=document_hidden_size, activation=rnn_activation)

            document_output, document_state = tf.nn.dynamic_rnn(document_rnn_cell, rnn_state_inputs_attention, dtype=tf.float32)

            #transform document final hidden state from document space to sentence space
            document_state_sentence_space = tf.layers.dense(document_state, sentence_hidden_size) #TODO activation? no?

            #determine target sentence score
            score = tf.reduce_sum(tf.multiply(document_state_sentence_space, rnn_state_target), axis=1) #dot product
            score_logit = tf.sigmoid(score) #sigmoid to get probability
            #extra target
            score_extra = tf.reduce_sum(tf.multiply(document_state_sentence_space, rnn_state_extra_target), axis=1) 
            score_logit_extra = tf.sigmoid(score_extra) 
        

        with tf.variable_scope("pretrain_generator_loss"):
            pretrain_generator_loss = -similarity(embedded_extra_target_sentence, embedded_generated_sentence)

        with tf.variable_scope("generator_loss"):
            generator_loss = (1-score_generated) - similarity(embedded_target_sentence, embedded

        with tf.variable_scope("discriminator_loss"):
            discriminator_loss = (1-score(rnn_state_extra_target))

        logger.info('Model {} building exiting.'.format(__name__))

        return ({"predicted_ending": predicted_ending},          # Output
                {"pretrain_generator_loss": pretrain_generator_loss, "generator_loss": generator_loss, "discriminator_loss": discriminator_loss},   # Loss
                {},
                {"sentences": sentences, "sentence_lengths": sentence_lengths, "target_label": target_label}
                )         # Any metric we may wanna monitor (I guess)
        #TODO remove target_label input without breaking things
