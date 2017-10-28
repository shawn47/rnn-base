from utils import load_model_parameters_theano, save_model_parameters_theano
from rnn_theano import RNNTheano
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime

vocabulary_size = 8000
num_sentences = 10
senten_min_length = 7
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
index_to_word = []
word_to_index = []


def loadModel():
	model = RNNTheano(vocabulary_size, hidden_dim=50)
	# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
	# save_model_parameters_theano('./data/trained-model-theano.npz', model)
	# load_model_parameters_theano('./data/trained-model-theano.npz', model)
	load_model_parameters_theano('./data/pretrained.npz', model)
	return model


def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

if __name__ == '__main__':
    print "Reading CSV file..."
    with open('./data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace = True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))

    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    model = loadModel()
    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model)
        print " ".join(sent)
