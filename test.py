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
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def loadData(file):
    print "Reading CSV file..."
    with open('./data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace = True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
    return sentences

def setTrainData(sentences):
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
     
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
     
    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
     
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train, y_train

class RNNNumpy:     
    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def softmax(self, values):
        targetE = []
        for item in values:
            targetE.append(np.exp(item))
        sumTargetE = np.sum(targetE, axis = 0)    
        return [x / sumTargetE for x in targetE]

    def forward_propagation(self, x):
	    T = len(x)
	    s = np.zeros((T + 1, self.hidden_dim))
	    s[-1] = np.zeros(self.hidden_dim)
	    o = np.zeros((T, self.word_dim))
	    for t in np.arange(T):
	        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t - 1]))
	        o[t] = self.softmax(self.V.dot(s[t]))
	    return [o, s]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis = 1)

    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
 
    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y) / N

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])              
                dLdU[:,x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h = 0.001, error_threshold = 0.01):
        bptt_gradients = self.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            it = np.nditer(parameter, flags = ['multi_index'], op_flags = ['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus) / (2*h)
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    def numpy_sdg_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
     
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        for i in range(len(y_train)):
            model.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

if __name__ == '__main__':
    X_train, Y_train = setTrainData(loadData("asd"))
    
    np.random.seed(10)
    model = RNNNumpy(vocabulary_size)
    # o, s = model.forward_propagation(X_train[10])
    # print o.shape
    # print o
    # predictions = model.predict(X_train[10])
    # print predictions.shape
    # print predictions

    # grad_check_vocab_size = 100
    # np.random.seed(10)
    # model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
    # model.gradient_check([0,1,2,3], [1,2,3,4])
    
    time_start = time.time()
    # model.numpy_sdg_step(X_train[10], Y_train[10], 0.005)
    
    losses = train_with_sgd(model, X_train[:100], Y_train[:100], nepoch = 10, evaluate_loss_after = 1)
    time_end = time.time()
    print ("%s -s" % (time_end - time_start))
