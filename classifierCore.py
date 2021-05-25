from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from core import *

import csv
import pickle
import time
import re, string, random

class ClassifierCore(Core):

    def trainClassifier(self):
        dataset = self.getCleanDataset()
        finalDataset = []
        for sentDataset in dataset:
            finalDataset.append(sentDataset)
        random.shuffle(finalDataset)
        train_data = finalDataset
        #~~~~~~-> Training the model...
        classifier = NaiveBayesClassifier.train(train_data)
        self.saveClassifier('classifier.pickle')

    def loadClassifier(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def saveClassifier(self, classifier, path):
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

    def trainDataValidator(self):
        try:
            return self.loadClassifier('classifier.pickle')
        except:
            self.trainClassifier()
            return self.loadClassifier('classifier.pickle')

    def processCustomText(self, custom_text):
        custom_tokens = self.removeNoise(word_tokenize(custom_text))
        custom_dict = dict([token, True] for token in custom_tokens)
        return custom_dict

    def textToSentiment(self, custom_text):
        classifier = self.trainDataValidator()

        custom_dict = self.processCustomText(custom_text)

        prediction = classifier.prob_classify(custom_dict)

        probs = {}
        samples = prediction.samples()
        for sample in samples:
            probs[str(sample)] = prediction.prob(sample)

        return probs
