# Natural Language Toolkit: RTE Classifier
#
# Copyright (C) 2001-2017 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Simple classifier for RTE corpus.

It calculates the overlap in words and named entities between text and
hypothesis, and also whether there are words / named entities in the
hypothesis which fail to occur in the text, since this is an indicator that
the hypothesis is more informative than (i.e not entailed by) the text.

TO DO: better Named Entity classification
TO DO: add lemmatization
"""
from __future__ import print_function

import nltk
from nltk.classify.util import accuracy
from flask import Flask, request, jsonify
import json
import pickle
from sklearn import metrics
import numpy as np

app = Flask(__name__)
classifier = None

def ne(token):
    """
    This just assumes that words in all caps or titles are
    named entities.

    :type token: str
    """
    if token.istitle() or token.isupper():
        return True
    return False

def lemmatize(word):
    """
    Use morphy from WordNet to find the base form of verbs.
    """
    lemma = nltk.corpus.wordnet.morphy(word, pos=nltk.corpus.wordnet.VERB)
    if lemma is not None:
        return lemma
    return word

class RTEFeatureExtractor(object):
    """
    This builds a bag of words for both the text and the hypothesis after
    throwing away some stopwords, then calculates overlap and difference.
    """
    def __init__(self, rtepair, stop=True, use_lemmatize=False):
        """
        :param rtepair: a ``RTEPair`` from which features should be extracted
        :param stop: if ``True``, stopwords are thrown away.
        :type stop: bool
        """
        self.stop = stop
        self.stopwords = set(['a', 'the', 'it', 'they', 'of', 'in', 'to', 'is',
                              'have', 'are', 'were', 'and', 'very', '.', ','])

        self.negwords = set(['no', 'not', 'never', 'failed', 'rejected',
                             'denied'])
        # Try to tokenize so that abbreviations, monetary amounts, email
        # addresses, URLs are single tokens.
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer('[\w.@:/]+|\w+|\$[\d.]+')

        #Get the set of word types for text and hypothesis
        self.text_tokens = tokenizer.tokenize(rtepair.text)
        self.hyp_tokens = tokenizer.tokenize(rtepair.hyp)
        self.text_words = set(self.text_tokens)
        self.hyp_words = set(self.hyp_tokens)

        if use_lemmatize:
            self.text_words = set(lemmatize(token) for token in self.text_tokens)
            self.hyp_words = set(lemmatize(token) for token in self.hyp_tokens)

        if self.stop:
            self.text_words = self.text_words - self.stopwords
            self.hyp_words = self.hyp_words - self.stopwords

        self._overlap = self.hyp_words & self.text_words
        self._hyp_extra = self.hyp_words - self.text_words
        self._txt_extra = self.text_words - self.hyp_words


    def overlap(self, toktype, debug=False):
        """
        Compute the overlap between text and hypothesis.

        :param toktype: distinguish Named Entities from ordinary words
        :type toktype: 'ne' or 'word'
        """
        ne_overlap = set(token for token in self._overlap if ne(token))
        if toktype == 'ne':
            if debug:
                print("ne overlap", ne_overlap)
            return ne_overlap
        elif toktype == 'word':
            if debug:
                print("word overlap", self._overlap - ne_overlap)
            return self._overlap - ne_overlap
        else:
            raise ValueError("Type not recognized:'%s'" % toktype)

    def hyp_extra(self, toktype, debug=True):
        """
        Compute the extraneous material in the hypothesis.

        :param toktype: distinguish Named Entities from ordinary words
        :type toktype: 'ne' or 'word'
        """
        ne_extra = set(token for token in self._hyp_extra if ne(token))
        if toktype == 'ne':
            return ne_extra
        elif toktype == 'word':
            return self._hyp_extra - ne_extra
        else:
            raise ValueError("Type not recognized: '%s'" % toktype)


def rte_features(rtepair):
    extractor = RTEFeatureExtractor(rtepair)
    features = {}
    features['alwayson'] = True
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    features['neg_txt'] = len(extractor.negwords & extractor.text_words)
    features['neg_hyp'] = len(extractor.negwords & extractor.hyp_words)
    return features


def rte_features_vector(rtepair):
    extractor = RTEFeatureExtractor(rtepair)
    num_features = 7
    features = np.zeros((num_features,),dtype="float32")
    features[0] = 1
    features[1] = len(extractor.overlap('word'))
    features[2] = len(extractor.hyp_extra('word'))
    features[3] = len(extractor.overlap('ne'))
    features[4] = len(extractor.hyp_extra('ne'))
    features[5] = len(extractor.negwords & extractor.text_words)
    features[6] = len(extractor.negwords & extractor.hyp_words)
    return features


def rte_classifier(trainer, features=rte_features):
    """
    Classify RTEPairs
    """
    train = [(pair, pair.value) for pair in
             nltk.corpus.rte.pairs(['rte1_dev.xml', 'rte2_dev.xml',
                                    'rte3_dev.xml'])]
    test = [(pair, pair.value) for pair in
            nltk.corpus.rte.pairs(['rte1_test.xml', 'rte2_test.xml',
                                   'rte3_test.xml'])]

    # Train up a classifier.
    print('Training classifier...')
    classifier = trainer([(features(pair), label) for (pair, label) in train])

    # Run the classifier on the test data.
    print('Testing classifier...')
    acc = accuracy(classifier, [(features(pair), label)
                                for (pair, label) in test])
    print('Accuracy: %6.4f' % acc)

    test_label = [label for (pair, label) in test]
    result = [classifier.classify(features(pair)) for (pair, label) in test]

    print(metrics.accuracy_score(test_label, result))
    print(metrics.classification_report(test_label, result))

    # Return the classifier
    return classifier


def demo_features():
    pairs = nltk.corpus.rte.pairs(['rte1_dev.xml'])[:6]
    for pair in pairs:
        print()
        for key in sorted(rte_features(pair)):
            print("%-15s => %s" % (key, rte_features(pair)[key]))


def demo_feature_extractor():
    rtepair = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
    extractor = RTEFeatureExtractor(rtepair)
    print(extractor.hyp_words)
    print(extractor.overlap('word'))
    print(extractor.overlap('ne'))
    print(extractor.hyp_extra('word'))


def demo():
    import nltk
    # import pdb; pdb.set_trace()

    # try:
    #     nltk.config_megam('/usr/local/bin/megam')
    #     trainer = lambda x: nltk.MaxentClassifier.train(x, 'megam')
    # except ValueError:
    #     try:
    #         trainer = lambda x: nltk.MaxentClassifier.train(x, 'BFGS')
    #     except ValueError:
    trainer = nltk.MaxentClassifier.train
    return rte_classifier(trainer)


@app.route("/", methods = ['POST', 'GET'])
def rte_classify():

    print(request.method)

    input = request.json
    print(input)

    rte = input['rte']
    index = int(input['index'])

    pair = nltk.corpus.rte.pairs(['rte' + rte + '_test.xml'])[index]
    print(pair.value)
    print(rte_features(pair))
    print(pair.text)
    print(pair.hyp)

    return json.dumps({ "text": pair.text,
                        "hyp": pair.hyp,
                        "entail": classifier.classify(rte_features(pair))})


if __name__ == '__main__':
    demo_features()
    demo_feature_extractor()

    if 1:
        classifier = demo()

        output = open('classifier.model', 'wb')
        pickle.dump(classifier, output)
    else:
        output = open('classifier.model', 'rb')
        classifier = pickle.load(output)

    output.close()

    # test = ((pair, pair.value) for pair in
    #         nltk.corpus.rte.pairs(['rte1_test.xml', 'rte2_test.xml',
    #                                'rte3_test.xml']))
    #
    # test1 = [(rte_features(pair), label) for (pair, label) in test]
    #
    # for i in test1:
    #     print(classifier.classify(i[0])==i[1])

    app.run()







