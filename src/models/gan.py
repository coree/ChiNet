#GAN architecture

import logging
import tensorflow as tf
from core import BaseDataSource, BaseModel
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
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        
        logger.info('Start building model {}'.format(__name__))


        #parameters TODO config from higher level
        sentence_hidden_size = 64
        document_hidden_size = 128
        document_n_hidden = 4
        embedding_size = 100
        vocab_size = 20000
        batch_size = 32
        initializer = tf.contrib.layers.xavier_initializer
        rnn_activation = tf.nn.relu #TODO use tanh (default)?

        #inputs
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors  # Data source automatically handles the datafiles
        
        input_sentences = input_tensors['input_sentence']   
        target_sentence = input_tensors['target_sentence']
        target_label = input_tensors['target_label'] #0=false ending 1=true ending
        word2vec_weights = input_tensors['embedding_weights'] #loaded word2vec embedding weights

        #embedding
        with tf.variable_scope('embed'):
            #load embedding weights
            embedding_weights = tf.get_variable("weights", shape=[vocab_size, embedding_size], initializer=tf.constant_initializer([vocab_size, embedding_size]), trainable=False)
            embedding_init = embedding_weights.assign(word2vec_weights) #embedding_init must be called to load embedding weights
            
            embedded_inputs = []
            for sentence in input_sentences:
                embedded_inputs += [tf.nn.embedding_lookup(embedding_weights, sentence)]
            embedded_target = tf.nn.embedding_lookup(embedding_weights, target_sentence)
            #embedded sentence shape [batch_size, embedding_size]

        #sentence rnn
        with tf.variable_scope('sentence'):
            #collect inputs and targets in list
            embedded_inputs_target_list = embedded_inputs + [embedded_target]
            
            #define rnn cell type
            sentence_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=sentence_hidden_size, activation=rnn_activation) 

            #apply rnn to each sentence
            sentence_states = []
            for embedded_sentence in embedded_inputs_target_list:
                sentence_output, sentence_state = tf.nn.dynamic_rnn(sentence_rnn_cell, embedded_sentence)
                sentence_states += [sentence_state]

            #separate sentence rnn final hidden states for inputs and target
            rnn_state_inputs = tf.stack(sentence_states[:document_n_hidden], axis=1) #shape [batch_size, 4, document_hidden_size]
            rnn_state_target = sentence_states[document_n_hidden] # shape [batch_size, document_hidden_size]

        #TODO attention
        
        #document RNN
        with tf.variable_scope('document'):
            document_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=document_hidden_size, activation=rnn_activation)

            document_output, document_state = tf.nn.dynamic_rnn(document_rnn_cell, rnn_state_inputs)

            #transform document final hidden state from document space to sentence space
            document_state_sentence_space = tf.layers_dense(document_state, sentence_hidden_size) #TODO activation? no?

            #determine target sentence score
            score = tf.reduce_sum(tf.multiply(document_state_sentence_space, rnn_state_target), axis=1) #dot product
            score_logit = tf.sigmoid(score) #sigmoid to get probability
    
        with tf.variable_scope('loss'):
            #sigmoid cross entropy loss TODO simpler loss function for binary prediction?
            sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label, logits=score_logit)


        logger.info('Model {} building exiting.'.format(__name__))

        return {"score": score}, {"sigmoid_cross_entropy": sigmoid_cross_entropy_loss}, {} #TODO last dict not used?
