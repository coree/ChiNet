#GAN architecture

import logging
import tensorflow as tf
from core import BaseModel
from datasources import TextSource

from typing import Dict

logger = logging.getLogger(__name__)


#cosine similarity
def cosine_similarity(x, y):
    """Computes the cosine similarity between x and y"""
    x = tf.nn.l2_normalize(x, 1)  
    y = tf.nn.l2_normalize(y, 1)
    return 1 - tf.losses.cosine_distance(x, y, axis=1)

#discrimiantor score from input document state and target sentence state
def score(document_state, target_sentence_state):
    with tf.variable_scope("document", reuse=True):
        document_to_sentence_weights = tf.get_variable("document_to_sentence_weights")

    document_state_sentence_space = tf.matmul(document_state, document_to_sentence_weights)
    #dimensions are expanded by 1 to make dot product correct: config['batch_size'] x [(1 x sentence_hidden) x (sentence_hidden x 1)] => config['batch_size'] x [1]
    score = tf.matmul(tf.expand_dims(document_state_sentence_space, axis=1), tf.expand_dims(target_sentence_state, axis=2))
    score_sigmoid = tf.sigmoid(score)
    return tf.squeeze(score_sigmoid) #squeeze removes 1x1 dimension at end

#TODO matrix operation version of Gumbel softmax could be breaking everything
def gumbel_softmax(generator_state, config):
    with tf.variable_scope("generator", reuse=True):
        generator_to_embedding_weights = tf.get_variable("generator_to_embedding_weights")
        temperature_weights = tf.clip_by_value(  # KIDS DON'T DO THIS AT HOME
                                        tf.get_variable("temperature_weights"),
                                        clip_value_min=1e-10,
                                        clip_value_max=0.999)
        temperature_epsilon = tf.clip_by_value(
                                        tf.get_variable("temperature_epsilon"),
                                        clip_value_min=1e-10,
                                        clip_value_max=0.999)
        temperature_epsilon = tf.Print(temperature_epsilon, [temperature_epsilon, temperature_weights])
    with tf.variable_scope("embedding", reuse=True):
        embedding_weights = tf.get_variable("embedding_weights")

    gumbel_pi = tf.nn.softmax(tf.matmul(tf.matmul(generator_state, generator_to_embedding_weights), tf.transpose(embedding_weights)))
    gumbel_t = tf.nn.relu(tf.matmul(generator_state, temperature_weights)) + temperature_epsilon
    gumbel_u = tf.random_uniform(shape=[config['batch_size'], config['vocab_size']], minval=0.0, maxval=1.0)
    gumbel_g = -tf.log(-tf.log(gumbel_u))
    #TODO clip t
    gumbel_p = tf.nn.softmax((tf.log(gumbel_pi) + gumbel_g) / gumbel_t)
    #TODO not sure if gumbel_y is correct...
    gumbel_y = tf.matmul(gumbel_p, embedding_weights) 
    
    return gumbel_y, gumbel_pi[:,config['stop_word_index']]

#generator conditional sentence generation
def generate_sentence(document_state, generator_rnn_cell, embedded_start_word, config, conditional=True):
    with tf.variable_scope("generator", reuse=True):
        generator_to_embedding_weights = tf.get_variable("generator_to_embedding_weights")
    with tf.variable_scope("document", reuse=True):
        document_to_generator_weights = tf.get_variable("document_to_generator_weights")

    #condition on input document state depending on parameter
    random_seed = tf.random_normal([config['batch_size'], config['generator_hidden_size']])
    if conditional:
        document_state_generator_space = tf.matmul(document_state, document_to_generator_weights)
        generator_conditioners = document_state_generator_space + random_seed
    else:
        generator_conditioners = random_seed
    
    initial_word = tf.stack([embedded_start_word]*config['batch_size'], axis=0) #first word seen by generator 

    #generate max_sentence_length words
    generated_sentence_list = []
    generated_sentence_length_list = [tf.fill(dims=[], value=config['max_sentence_length'])]*config['batch_size']

    generated_word = initial_word
    generator_state = generator_conditioners  # TODO tuple?
    for i in range(config['max_sentence_length']):
        _, generator_state = generator_rnn_cell(generated_word, generator_state, scope="generator")
        generated_word, stop_word_probability = gumbel_softmax(generator_state=generator_state, config=config)

        #set sequence length to min(sequence_length, i) if stop word was generated
        for j in range(config['batch_size']):
            stop_word_generated_condition = stop_word_probability[j] > tf.constant(value=config['stop_word_bound'])
            generated_sentence_length_list[j] = tf.cond(stop_word_generated_condition, lambda: tf.minimum(tf.constant(value=i), generated_sentence_length_list[j]), lambda: generated_sentence_length_list[j])

        generated_sentence_list += [generated_word]

    generated_sentence = tf.stack(generated_sentence_list, axis=1)
    generated_sentence_length = tf.stack(generated_sentence_length_list, axis=0)

    return generated_sentence, generated_sentence_length

#given sentences with one-hot vocabulary indices, apply embeddings and sentence rnn to get sentence representations
def apply_embedding_and_sentence_rnn(sentences, sentence_lengths, sentence_rnn_cell, sentence_n, config): #TODO sentence rnn cell through get_variables?
    with tf.variable_scope("embedding", reuse=True):
        embedding_weights = tf.get_variable("embedding_weights")
    
    sentence_states_list = []
    for i in range(sentence_n):
        embedded_sentence = tf.nn.embedding_lookup(embedding_weights, sentences[:,i])
        initial_state = sentence_rnn_cell.zero_state(config['batch_size'], dtype=tf.float32)
        _, sentence_state = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=embedded_sentence, sequence_length=sentence_lengths[:,i], initial_state=initial_state, dtype=tf.float32, scope="sentence")
        sentence_states_list += [sentence_state]

    sentence_states = tf.stack(sentence_states_list, axis=1)

    return sentence_states

class CGAN(BaseModel):
    def build_model(self, data_sources: Dict[str, TextSource], mode: str):
        logger.info('Start building model {}'.format(__name__))

        #PARAMETERS TODO config from higher level
        config = dict()
        config['sentence_hidden_size'] = 64
        config['document_hidden_size'] = 128
        config['generator_hidden_size'] = 256
        config['embedding_size'] = 300
        config['max_sentence_length'] = 50
        config['input_sentence_n'] = 4
        config['vocab_size'] = data_sources['real'].vocab_size 
        config['batch_size'] = data_sources['real'].batch_size
        config['initializer'] = tf.contrib.layers.xavier_initializer()
        config['rnn_activation'] = tf.nn.tanh 
        config['start_word_index'] = 0
        config['stop_word_index'] = 1
        config['stop_word_bound'] = 0.9

        #SHARED VARIABLES
        with tf.variable_scope('inputs'):
            #sentences contains both input and target sentences
            sentences = tf.placeholder(shape=[config['batch_size'], config['input_sentence_n']+1, config['max_sentence_length']], dtype=tf.int64, name='sentences')
            sentence_lengths = tf.placeholder(shape=[config['batch_size'], config['input_sentence_n']+1], dtype=tf.int32, name='sentence_lengths') 
            #extra sentence is used for pretraining and evaluation
            extra_sentence = tf.placeholder(shape=[config['batch_size'], 1, config['max_sentence_length']], dtype=tf.int64, name='extra_sentence')
            extra_sentence_length = tf.placeholder(shape=[config['batch_size'], 1], dtype=tf.int32, name="extra_sentence_length")

        with tf.variable_scope('embedding'):
            word2vec_weights = tf.placeholder(shape=[config['vocab_size'], config['embedding_size']], dtype=tf.float32, name='word2vec_weights') 
            embedding_weights = tf.get_variable("embedding_weights", shape=[config['vocab_size'], config['embedding_size']], trainable=False)
            embedding_assign_op = embedding_weights.assign(word2vec_weights) #must be called to load embedding weights
            embedded_start_word = tf.nn.embedding_lookup(embedding_weights, tf.constant(value=config['start_word_index']))

        with tf.variable_scope('sentence'):
            sentence_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=config['sentence_hidden_size'], activation=config['rnn_activation'], name="sentence_cell")
        
        #TODO attention
        
        with tf.variable_scope('document'):
            document_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=config['document_hidden_size'], activation=config['rnn_activation'], name="document_cell")
            document_to_sentence_weights = tf.get_variable(name="document_to_sentence_weights", shape=[config['document_hidden_size'], config['sentence_hidden_size']], dtype=tf.float32, initializer=config['initializer'])
            document_to_generator_weights = tf.get_variable(name="document_to_generator_weights", shape=[config['document_hidden_size'], config['generator_hidden_size']], dtype=tf.float32, initializer=config['initializer'])

        with tf.variable_scope('generator'):
            generator_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=config['generator_hidden_size'], activation=config['rnn_activation'], name="generator_cell")
            generator_to_embedding_weights = tf.get_variable(name="generator_to_embedding_weights", shape=[config['generator_hidden_size'], config['embedding_size']], dtype=tf.float32, initializer=config['initializer'])
            temperature_weights = tf.get_variable(name="temperature_weights", shape=[config['generator_hidden_size'],1], dtype=tf.float32, initializer=config['initializer']) 
            temperature_epsilon = tf.get_variable(name="temperature_epsilon", shape=[], initializer=tf.constant_initializer(0.01), dtype=tf.float32) #TODO can float32 contain 1e-20? 

        #GRAPHS 
        #Generator pretraining
        target_state = apply_embedding_and_sentence_rnn(sentences=extra_sentence, sentence_lengths=extra_sentence_length, sentence_rnn_cell=sentence_rnn_cell, sentence_n=1, config=config)[:,0]
            
        generated_sentence, generated_sentence_length = generate_sentence(document_state=None, conditional=False, generator_rnn_cell=generator_rnn_cell, embedded_start_word=embedded_start_word, config=config)

        initial_state = sentence_rnn_cell.zero_state(config['batch_size'], dtype=tf.float32)
        _, generated_state = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=generated_sentence, sequence_length=generated_sentence_length, initial_state=initial_state, dtype=tf.float32, scope="sentence")

        pretrain_generator_loss = tf.reduce_sum(-cosine_similarity(target_state, generated_state))

        #Generator
        with tf.name_scope('generator'):
            sentence_states = apply_embedding_and_sentence_rnn(sentences=sentences, sentence_lengths=sentence_lengths, sentence_rnn_cell=sentence_rnn_cell, sentence_n=config['input_sentence_n']+1, config=config)
            input_states = sentence_states[:,:config['input_sentence_n']]
            target_state = sentence_states[:,config['input_sentence_n']]
            
            input_states_attention = input_states #TODO attention; use attention (introduces information about target) when generating sentence?

            initial_state = document_rnn_cell.zero_state(config['batch_size'], dtype=tf.float32)
            _, document_state = tf.nn.dynamic_rnn(document_rnn_cell, input_states_attention, initial_state=initial_state, dtype=tf.float32, scope="document")

            generated_sentence, generated_sentence_length = generate_sentence(document_state=document_state, conditional=True, generator_rnn_cell=generator_rnn_cell, embedded_start_word=embedded_start_word, config=config)

            initial_state = sentence_rnn_cell.zero_state(config['batch_size'], dtype=tf.float32)
            _, generated_state = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=generated_sentence, sequence_length=generated_sentence_length, initial_state=initial_state, dtype=tf.float32, scope="sentence")
        
            score_generated = score(document_state, generated_state) 

            generator_loss = tf.reduce_sum(-tf.log(score_generated) - cosine_similarity(target_state, generated_state)) #TODO minus similarity? (paper says otherwise but I think it's typo)

        #Discriminator
        with tf.name_scope('discriminator'):
            sentence_states = apply_embedding_and_sentence_rnn(sentences=sentences, sentence_lengths=sentence_lengths, sentence_rnn_cell=sentence_rnn_cell, sentence_n=config['input_sentence_n']+1, config=config)
            input_states = sentence_states[:,:config['input_sentence_n']]
            target_state = sentence_states[:,config['input_sentence_n']]
                
            input_states_attention = input_states #TODO attention; repeat document state twice for target and generated attention?
            
            initial_state = document_rnn_cell.zero_state(config['batch_size'], dtype=tf.float32)
            _, document_state = tf.nn.dynamic_rnn(document_rnn_cell, input_states_attention, initial_state=initial_state, dtype=tf.float32, scope="document")

            generated_sentence, generated_sentence_length = generate_sentence(document_state=document_state, conditional=True, generator_rnn_cell=generator_rnn_cell, embedded_start_word=embedded_start_word, config=config)

            initial_state = sentence_rnn_cell.zero_state(config['batch_size'], dtype=tf.float32)
            _, generated_state = tf.nn.dynamic_rnn(sentence_rnn_cell, inputs=generated_sentence, sequence_length=generated_sentence_length, initial_state=initial_state, dtype=tf.float32, scope="sentence")
                
            score_generated = score(document_state, generated_state)
            score_target = score(document_state, target_state)

            discriminator_loss = tf.reduce_sum(-tf.log(score_target) - tf.log(1-score_generated))

        #Prediction
        sentences_with_extra = tf.concat([sentences, extra_sentence], axis=1)
        sentence_lengths_with_extra = tf.concat([sentence_lengths, extra_sentence_length], axis=1)

        sentence_states = apply_embedding_and_sentence_rnn(sentences=sentences_with_extra, sentence_lengths=sentence_lengths_with_extra, sentence_rnn_cell=sentence_rnn_cell, sentence_n=config['input_sentence_n']+2, config=config)
        input_states = sentence_states[:,:config['input_sentence_n']]
        target_1_state = sentence_states[:,config['input_sentence_n']]
        target_2_state = sentence_states[:,config['input_sentence_n']+1]
           
        #TODO attention; 2 separate document states based on attention?
           
        initial_state = document_rnn_cell.zero_state(config['batch_size'], dtype=tf.float32)
        _, document_state = tf.nn.dynamic_rnn(document_rnn_cell, input_states_attention, initial_state=initial_state, dtype=tf.float32, scope="document")
            
        score_target_1 = score(document_state, target_1_state)
        score_target_2 = score(document_state, target_2_state)
            
        #predicted ending is 0 for ending_1 and 1 for ending_2 (extra target)
        predicted_ending_list = []
        for i in range(config['batch_size']):
            predicted_ending_list += tf.cond(score_target_1[i] > score_target_2[i], lambda: 0, lambda: 1)
        predicted_ending = tf.stack(predicted_ending_list, axis=0)

        logger.info('Model {} building exiting.'.format(__name__))

        return ({"predicted_ending": predicted_ending, "embedding_assign_op": embedding_assign_op},          #output
                {"pretrain_loss": pretrain_generator_loss, "generator_loss": generator_loss, "discriminator_loss": discriminator_loss},   #loss
                {},  #metrics
                {"sentences": sentences, "sentence_lengths": sentence_lengths, "extra_sentence": extra_sentence, "extra_sentence_length": extra_sentence_length,
                    "word2vec_weights": word2vec_weights}  #input
                )         
