import csv
import numpy as np
import nltk
import os
import pickle
# import pandas as pd

#write vocabulary to file
def write_vocab(id_word, word_id):
    if not os.path.exists("./vocab"):
        os.makedirs("./vocab")
    with open("./vocab/WordID.pkl", "wb") as f:
        pickle.dump(word_id, f)
    with open("./vocab/IDWord.pkl", "wb") as f:
        pickle.dump(id_word, f)

#write data to file
def write_processed_data(processed_data):
    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")
    with open("./dataset/processed_data.pkl", "wb") as f:
        pickle.dump(processed_data, f)


#load vocabulary and inverse vocabulary from file
def load_vocab():
    with open('./vocab/IDWord.pkl', 'rb') as IDWord_file:
        id_word = pickle.load(IDWord_file)
    with open('./vocab/WordID.pkl', 'rb') as WordID_file:
        word_id = pickle.load(WordID_file)



def make_vocab(vocab, vocab_size):
    # add most frequent words in vocabulary to known words
    known_words = []
    for word in sorted(vocab, key=vocab.get, reverse=True):
        if len(known_words) < vocab_size - 1:
            known_words += [word]

    # build dictonary of word to ids
    word_id = {word: i for i, word in enumerate(known_words, 1)}
    word_id["<unk>"] = 0
    id_word = dict(zip(word_id.values(), word_id.keys()))  # reverse dict to get word from id
    # write to pickle
    write_vocab(id_word, word_id)
    return id_word, known_words, word_id



def preprocess_data(read_file='../datasets/train_stories.csv', vocab_size=20000, write=True):
    with open(read_file, 'r') as f:
        reader = csv.reader(f)
        data =[]
        vocab = dict()
        for row in reader:
            story =[]
            for element in row[1:]:  # clip IDs column
                tokens = nltk.word_tokenize(element)
                sentence = []
                for token in tokens:

                    token = token.lower()
                    if token in vocab.keys():
                        vocab[token] += 1
                    else:
                        vocab[token] = 1

                    sentence.append(token)
                story.append(sentence)
            data.append(story)

        data.pop(0) # clip description row


        # generate vocabulary and write it to a pickle
        id_word, known_words, word_id = make_vocab(vocab, vocab_size)

        #print('known_words', known_words)
        #print('id_word', id_word)

        # exchanges words with ids and replaces words that are not in vocab with the id of unk
        for story in data:
            for sentence in story:
                for idx, word in enumerate(sentence):
                    if word not in known_words:
                        sentence[idx] = word_id['<unk>']
                    else:
                        sentence[idx] = word_id[word]

        np_data = np.array(data)

        if write:
            write_processed_data(np_data)

        return np_data, word_id, id_word