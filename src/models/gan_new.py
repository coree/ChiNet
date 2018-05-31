#GAN architecture

import logging
import tensorflow as tf
from core import BaseModel
from datasources import TextSource

from typing import Dict

logger = logging.getLogger(__name__)

#cosine similarity
def similarity(x, y):
    return 0

#discrimiantor score from input document state and target sentence state
def score(document_state, target_sentence_state, document_to_sentence_weights) 
    document_state_sentence_space = tf.matmul(document_state, document_to_sentence_weights)
    #dimensions are expanded by 1 to make dot product correct: batch_size x [(1 x sentence_hidden) x (sentence_hidden x 1)] => batch_size x [1]
    score = tf.matmul(tf.expand_dims(document_state_sentence_space, axis=1), tf.expand_dims(target_sentence_state, axis=2))
    score_sigmoid = tf.sigmoid(score)
    return score_sigmoid

#TODO matrix operation version of Gumbel could be breaking everything
def gumbel_softmax(generator_state, generator_to_embedding_weights, embedding_weights, temperature_weights, temperature_epsilon, batch_size):
    gumbel_pi = tf.nn.softmax(tf.matmul(tf.matmul(generator_state, generator_to_embedding_weights), tf.transpose(embedding_weights)))
    gumbel_t = tf.nn.relu(tf.matmul(tf.expand_dims(generator_state, axis=1), temperature_weights)) + temperature_epsilon
    gumbel_u = tf.random.normal([batch_size])
    gumbel_g = -tf.log(-tf.log(gumbel_u))
    gumbel_p = tf.softmax((tf.log(gumbel_pi) + gumbel_g) / gumbel_t)
    gumbel_y = tf.matmul(gumbel_p, embedding_weights)
    return gumbel_y

#generator conditional sentence generation
def generate_sentence(document_state, generator_to_embedding_weights, embedding_weights, temperature_weights, temperature_epsilon, batch_size, embedding_size, embedded_stop_word, embedded_start_word, max_sentence_length, stop_word_error_bound, conditional=True):
    stop_word_error_bound = tf.constant(value=stop_word_error_bound)

    generated_sentence = tf.fill([batch_size, max_sentence_length, embeddeding_size], embedded_stop_word)
    generated_sentence_length = tf.fill([batch_size], max_sentence_length)
            
    random_seed = tf.random_normal([batch_size]) #TODO change random seed every step? instead distort document state?

    #condition on input document state depending on parameter
    if conditional:
        generator_conditioners = tf.concat([document_state, random_seed], axis=1)
    else:
        generator_conditioners = random_seed

    initial_words = tf.constant(value=embedded_start_word, shape=[batch_size], dtype=tf.float32)
    generator_input = tf.concat([initial_words, generator_conditioners], axis=1)
    generator_state = generator_rnn_cell.zero_state(batch_size, dtype=tf.float32)
    for i in range(max_sentence_length):
        _, generator_state = generator_rnn_cell(generator_input, generator_state)
        #determine embedded generated word from generator state 
        generated_word = gumbel_softmax(generator_state=generator_state, generator_to_embedding_weights=generator_to_embedding_weights, embedding_weights=embedding_weights, temperature_weights=temperature_weights, temperature_epsilon=temperature_epsilon, batch_size=batch_size)
        for j in range(batch_size):
            #set sequence length to min(sequence_length, i) if stop word was generated
            stop_word_generated_condition = (tf.reduce_sum(generated_word[j] - embedded_stop_word) < stop_word_error_bound) #TODO right way to do condition? TODO better way to check if embedding is "close enough"/equal to stop word?
            generated_sentence_length[j] = tf.cond(stop_word_generated_condition, tf.minimum(tf.constant(value=i), generated_sentence_length[j]), generated_sentence_length[j])
        #TODO terminate generation loop if stop words have been generated for every sentence in batch, sub-TODO: possible to only generate for batch elements where stop words not yet generated?
        generated_sentence[:,i] = generated_word
        generator_input = tf.concat([generated_words, generator_conditioners], axis=1)

    return generated_sentence, generated_sentence_length

#given sentences with one-hot vocabulary indices, apply embeddings and sentence rnn to get sentence representations
def embed_and_sentence_rnn(sentences, sentence_lengths, sentence_n, max_sentence_length, batch_size):
    #TODO semiflat trick could be breaking everything, idea behind is to embed and sentence-rnn process every sentence in every batch in parallel and then reshape to restore batches
    semiflat_sentences = tf.reshape(sentences, [batch_size*(sentence_n), max_sentence_length])
    semiflat_sentence_lengths = tf.reshape(sentence_lengths, [batch_size*sentence_n])      
    semiflat_sentences_embedded = tf.nn.embedding_lookup(embedding_weights, semiflat_sentences)
    _, semiflat_sentence_states = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=semiflat_sentences_embedded, sequence_length=semiflat_sentence_lengths, dtype=tf.float32)
    sentence_states = tf.reshape(semiflat_sentence_states, [batch_size, sentence_n, sentence_hidden_size])
    return sentence_states

#TODO modify into GAN, for now just discriminator
class GAN(BaseModel):
    def build_model(self, data_sources: Dict[str, TextSource], mode: str):
        logger.info('Start building model {}'.format(__name__))

        #PARAMETERS TODO config from higher level
        sentence_hidden_size = 64
        document_hidden_size = 128
        generator_hidden_size = 256
        embedding_size = 100
        max_sentence_length = 50
        input_sentence_n = 4
        document_n_hidden = input_sentence_n
        vocab_size = data_sources['real'].vocab_size 
        batch_size = data_sources['real'].batch_size 
        initializer = tf.contrib.layers.xavier_initializer
        rnn_activation = tf.nn.relu #TODO use tanh (default)?
        embedded_start_word = 0 #TODO
        embedded_stop_word = 0 #TODO
        stop_word_error_bound = 10 #TODO


        #SHARED VARIABLES
        with tf.variable_scope('inputs'):
            #sentences contains both input and target sentences
            sentences = tf.placeholder(shape=[batch_size, input_sentence_n+1, max_sentence_length], dtype=tf.int64, name='sentences')
            sentence_lengths = tf.placeholder(shape=[batch_size, input_sentence_n+1], dtype=tf.int32, name='sentence_lengths') 
            #extra target sentence is used for generated  evaluation
            extra_target_sentence = tf.placeholder(shape=[batch_size, max_sentence_length], dtype=tf.int64, name='extra_target_sentence')
            extra_target_sentence_length = tf.placeholder(shape=[batch_size], dtype=tf.int32, name="extra_target_sentence_length")

        with tf.variable_scope('embedding'):
            word2vec_weights = tf.placeholder(shape=[vocab_size, embedding_size], dtype=tf.float32) 
            embedding_weights = tf.get_variable("weights", shape=[vocab_size, embedding_size], initializer=tf.constant_initializer([vocab_size, embedding_size]), trainable=False)
            embedding_assign_op = embedding_weights.assign(word2vec_weights) #must be called to load embedding weights

        with tf.variable_scope('sentence'):
            sentence_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=sentence_hidden_size, activation=rnn_activation) 

        #TODO attention
        
        with tf.variable_scope('document'):
            document_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=document_hidden_size, activation=rnn_activation)
            document_to_sentence_weights = tf.get_variable(name="document_to_sentence_weights", shape=[document_hidden_size, sentence_hidden_size], dtype=tf.float32, initializer=initializer)

        with tf.variable_scope('generator'):
            generator_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=generator_hidden_size, activation=rnn_activation)
            generator_to_embedding_weights = tf.get_variable(name="generator_to_embedding_weights", shape=[generator_hidden_size, embedding_size], dtype=tf.float32, initializer=initializer)
            temperature_weights = tf.get_variable(name="temperature_weights", shape=[generator_hidden_size,1], dtype=tf.float32) 
            temperature_epsilon = tf.constant(1e-20)


        #GRAPHS
        with tf.variable_scope("pretrain_generator_loss"):
            target_state = embed_and_sentence_rnn(sentences=extra_target_sentence, sentence_lengths=extra_target_sentence_length, sentence_n=1, max_sentence_length=max_sentence_length, batch_size=batch_size)
            
            generated_sentence, generated_sentence_length = generate_sentence(document_state=None, generator_to_embedding_weights=generator_to_embedding_weights, embedding_weights=embedding_weights, temperature_weights=temperature_weights, temperature_epsilon=temperature_epsilon, batch_size=batch_size, embedding_size=embedding_size, embedded_stop_word=embedded_stop_word, embedded_start_word=embedded_start_word, max_sentence_length=max_sentence_length, conditional=False)

            _, generated_state = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=generated_sentence, sentence_length=generated_sentence_lengths, dtype=tf.float32)

            pretrain_generator_loss = -similarity(target_state, generated_state)

        with tf.variable_scope("generator_loss"):
            sentence_states = embed_and_sentence_rnn(sentences=sentences, sentence_lengths=sentence_lengths, sentence_n=input_sentence_n+1, max_sentence_length=max_sentence_length, batch_size=batch_size)
            input_states = sentence_states[:,:input_sentence_n]
            target_state = sentence_states[:,input_sentence_n]
            
            input_states_attention = input_states #TODO attention

            _, document_state = tf.nn.dynamic_rnn(document_rnn_cell, input_states_attention, dtype=tf.float32)

            generated_sentence, generated_sentence_length = generate_sentence(document_state=document_state, generator_to_embedding_weights=generator_to_embedding_weights, embedding_weights=embedding_weights, temperature_weights=temperature_weights, temperature_epsilon=temperature_epsilon, batch_size=batch_size, embedding_size=embedding_size, embedded_stop_word=embedded_stop_word, embedded_start_word=embedded_start_word, max_sentence_length=max_sentence_length, conditional=True)

            _, generated_sentence_state = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=generated_sentence, sequence_length=generated_sentence_lengths, dtype=tf.float32)

            #loss
            score_generated = score(document_state, generated_sentence_state, document_to_sentence_weights) 
            generator_loss = -np.log(score_generated) - similarity(target_state, generated_state) #TODO minus similarity? (paper says otherwise but I think it's typo)

        #TODO namespace issues?
        with tf.variable_scope("discriminator_loss"):
            sentence_states = embed_and_sentence_rnn(sentences=sentences, sentence_lengths=sentence_lengths, sentence_n=input_sentence_n+1, max_sentence_length=max_sentence_length, batch_size=batch_size)
            input_states = sentence_states[:,:input_sentence_n]
            target_state = sentence_states[:,input_sentence_n]
            
            input_states_attention = input_states #TODO attention
            
            _, document_state = tf.nn.dynamic_rnn(document_rnn_cell, input_states_attention, dtype=tf.float32)

            generated_sentence, generated_sentence_length = generate_sentence(document_state=document_state, generator_to_embedding_weights=generator_to_embedding_weights, embedding_weights=embedding_weights, temperature_weights=temperature_weights, temperature_epsilon=temperature_epsilon, batch_size=batch_size, embedding_size=embedding_size, embedded_stop_word=embedded_stop_word, embedded_start_word=embedded_start_word, max_sentence_length=max_sentence_length, conditional=True)

            _, generated_state = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=generated_sentence, sequence_length=generated_sentence_lengths, dtype=tf.float32)
            
            score_generated = score(document_state, generated_state, document_to_sentence_weights)
            score_target = score(document_state, target_state, document_to_sentence_weigths)

            discriminator_loss = -tf.log(score_target) - tf.log(1-score_generated)

        with tf.variable_scope("prediction"):
            sentences_with_extra = tf.concat([sentences, extra_target_sentence], axis=1)
            sentence_lengths_with_extra = tf.concat([sentence_lengths, extra_target_sentence_length], axis=1)

            sentence_states = embed_and_sentence_rnn(sentences=sentences_with_extra, sentence_lengths=sentence_lengths_with_extra, sentence_n=input_sentence_n+2, max_sentence_length=max_sentence_length, batch_size=batch_size)
            input_states = sentence_states[:,:input_sentence_n]
            target_1_state = sentence_states[:,input_sentence_n]
            target_2_state = sentence_states[:,input_sentence_n+1]
            
            _, document_state = tf.nn.dynamic_rnn(document_rnn_cell, input_states_attention, dtype=tf.float32)
            
            score_target_1 = score(document_state, target_1_state, document_to_sentence_weights)
            score_target_2 = score(document_state, target_2_state, document_to_sentence_weights)
            
            #predicted ending is 0 for ending_1 and 1 for ending_2 (extra target)
            predicted_ending = tf.cond(score_target_1 > score_target_2, 0, 1) 

        logger.info('Model {} building exiting.'.format(__name__))

        return ({"predicted_ending": predicted_ending},          #output
                {"pretrain_generator_loss": pretrain_generator_loss, "generator_loss": generator_loss, "discriminator_loss": discriminator_loss},   #loss
                {}, #whatever
                {"sentences": sentences, "sentence_lengths": sentence_lengths}  #input
                )         
