import csv
import numpy as np
import nltk
from nltk.corpus import names
import os
import pickle
import time
import logging
import random
logger = logging.getLogger(__name__)

MALE_NAMES = names.words('male.txt')
FEMALE_NAMES = names.words('female.txt')
MALES = ['John','Adrian','Shane','Michael','Philip', 'Ethan', 'Xavier', 'Evan', 'Ralph','Frank']
FEMALES = ['Charlotte','Vicki','Monica','Jana','Elisa', 'Madonna','Sophia','Lauren','Jackie','Viola']

#write vocabulary to file
def write_vocab(id_word, word_id):
    if not os.path.exists("../datasets/vocab"):
        os.makedirs("../datasets/vocab")
    with open("../datasets/vocab/WordID.pkl", "wb") as f:
        pickle.dump(word_id, f)
    with open("../datasets/vocab/IDWord.pkl", "wb") as f:
        pickle.dump(id_word, f)

#write data to file
def write_processed_data(processed_data, write_file):
    if not os.path.exists("../datasets"):
        os.makedirs("../datasets")
    with open(write_file, "wb") as f:
        pickle.dump(processed_data, f)


#load vocabulary and inverse vocabulary from file
def load_vocab():
    with open('../datasets/vocab/IDWord.pkl', 'rb') as IDWord_file:
        id_word = pickle.load(IDWord_file)
    with open('../datasets/vocab/WordID.pkl', 'rb') as WordID_file:
        word_id = pickle.load(WordID_file)
    return id_word, word_id


def load_preprocessed_data(filename='../datasets/train_stories.processed'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def make_vocab(vocab, vocab_size):
    # add most frequent words in vocabulary to known words
    known_words = []
    for word in sorted(vocab, key=vocab.get, reverse=True):
        if len(known_words) < vocab_size - 1:
            known_words += [word]

    # build dictonary of word to ids
    word_id = {word: i for i, word in enumerate(known_words, 4)}
    word_id["BOS"] = 0
    word_id["EOS"] = 1
    word_id["PAD"] = 2
    word_id["UNK"] = 3

    id_word = dict(zip(word_id.values(), word_id.keys()))  # reverse dict to get word from id
    known_words = list(word_id.keys())
    # write to pickle
    write_vocab(id_word, word_id)
    return id_word, known_words, word_id


# TODO: Dehumanyfy function is extremely time inefficient (could be much more optimized)
# TODO: Check if sex may give any source of extra information (which I doubt, tho)
def dehumanify(story):
    """
    Just as it says, people are not names, they are just another numbered blorg! 
            (culture yourselves [ https://preview.tinyurl.com/greatBlorg ] [ /watch?v=-FwnxRiozyU ])

    Args:
        sentence (list): Human named sentence.
    Retunrs:
        sentence (list) with glorious meaningful names.

    (In a more serious context, we replace proper names with a stereotypical string in order to reduce 
     the vocabulary size, delete unnecesary information and (probably) improve the model. ) 
    """
    random.shuffle(MALES)
    random.shuffle(FEMALES)
    male_counter = 0
    female_counter = 0
    names_dict = {}
    sentences = []

    for sent in story:
        sentence = nltk.word_tokenize(str(sent))
        for idx, word in enumerate(sentence):
            if word in MALE_NAMES:
                if word not in list(names_dict.keys()):
                    names_dict[word] = male_counter
                    sentence[idx] = MALES[male_counter]
                    male_counter += 1
                else:
                    sentence[idx] = MALES[names_dict[word]]

            elif word in FEMALE_NAMES:
                if word not in list(names_dict.keys()):
                    names_dict[word] = female_counter
                    sentence[idx] = FEMALES[female_counter]
                    female_counter += 1
                else:
                    sentence[idx] = FEMALES[names_dict[word]]
                    
        sentences.append(sentence)
    return sentences

def preprocess_file(file_path='../datasets/train_stories.csv', 
                         clean_file='../datasets/train_stories.clean', testing=False):
    """ Preprocess the csv into a pickled python list"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    # Just grab sentences from 1 to 5 and save them in a new file
    if testing:
        keys = ['InputSentence{}'.format(i+1) for i in range(4)] + ['RandomFifthSentenceQuiz{}'.format(i+1) for i in range(2)]
        data = [[s[k] for k in keys]  for s in data] 
    else:
        data = [[s[k] for k in ['sentence{}'.format(i+1) for i in range(5)]]  for s in data] 
        
    with open(clean_file, 'wb') as f:
        pickle.dump(data, f)


def preprocess_results(file_path='../datasets/cloze_test_val.csv', 
                         clean_file='../datasets/results.clean'):
    """ Preprocess the csv into a pickled python list"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    # Grab answers and process them
    data = [int(s['AnswerRightEnding'])-1 for s in data] 

    with open(clean_file, 'wb') as f:
        pickle.dump(data, f)


def load_results(clean_file='../datasets/results.clean', overwrite=False):
    if not os.path.exists(clean_file) or overwrite:
        preprocess_results(clean_file=clean_file)
        logger.info('Results file preprocessed')
    with open(clean_file, 'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_data(read_file='../datasets/train_stories.clean', write_file='../datasets/train_stories.preprocessed', vocab_size=20000, write=True, use_prevocab=False):
    """ 
    Preprocess, tokenise and encode the data
    
    Args:
        read_file (str): Clean data file (generated by preprocess_file)
        vocab_size (int): Size of vocabulary (Deafault 20k)
        write (bool): Write output to file (w/ wirte_preprocessed_data)

    Returns:
        (data, word_id, id_word)
    """
    with open(read_file, 'rb') as f:
        raw = pickle.load(f)
    data =[]
    vocab = dict()
    start_time = time.time()
    for progress, block in enumerate(raw):
        story =[]
        dehumanized_block = dehumanify(block)
        for tokens in dehumanized_block:
            #tokens = nltk.word_tokenize(str(line))
            sentence = []
            #dehumanify(tokens)
            for token in tokens:

                token = token.lower()
                if token in vocab.keys():
                    vocab[token] += 1
                else:
                    vocab[token] = 1

                sentence.append(token)
            story.append(sentence)
        if not (progress+1) % 1000:
            t = (len(raw)-progress)*(time.time()-start_time)/progress
            logger.info('Pt.1/2 Estimated completion in {0:.0f}:{1:02.0f} min  - Preprocessing 1/2 completion {2}/{3} stories)'.format(
                                                        t//60,  t%60,  # min, sec 
                                                        progress + 1, 
                                                        len(raw)))
        data.append(story)

    if not use_prevocab:
        # generate vocabulary and write it to a pickle
        id_word, known_words, word_id = make_vocab(vocab, vocab_size)
    else:
        id_word, word_id = load_vocab()
        known_words = list(word_id.keys())
    #print('known_words', known_words)
    #print('id_word', id_word)

    # exchanges words with ids and replaces words that are not in vocab with the id of unk
    new_data = []
    start_time = time.time()
    for progress, story in enumerate(data):
        new_story = []
        for sentence in story:
            sentence = ['BOS'] + sentence
            sentence += ['EOS']
            for idx, word in enumerate(sentence):
                if word not in known_words:
                    sentence[idx] = word_id['UNK']
                else:
                    sentence[idx] = word_id[word]
            new_story.append(sentence)
        new_data.append(new_story)
        if not (progress+1) % 1000:
            t = (len(raw)-progress)*(time.time()-start_time)/progress
            logger.info('Pt.2/2 Estimated completion in {0:.0f}:{1:02.0f} min  - Preprocessing 2/2 completion {2}/{3} stories)'.format(
                                                        t//60,  t%60,  # min, sec 
                                                        progress + 1, 
                                                        len(data)))
    if write:
        write_processed_data(new_data, write_file)
    return new_data, word_id, id_word
