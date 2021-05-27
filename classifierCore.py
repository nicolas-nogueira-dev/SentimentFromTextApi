from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from core import *

import pickle
import random

class ClassifierCore(Core):

    def __init__(self, folderpath, sentiments, classifierPath):
        super().__init__(folderpath, sentiments)
        self.classifierPath = classifierPath

    def trainClassifier(self):
        dataset = self.getCleanDataset()
        finalDataset = []
        for key in dataset:
            finalDataset += [dataset[key]]
        train_data = []
        for i in finalDataset:
            for j in i:
                train_data.append(j)
        random.shuffle(train_data)
        classifier = NaiveBayesClassifier.train(train_data)
        self.saveClassifier(classifier,self.classifierPath)

    def loadClassifier(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def saveClassifier(self, classifier, path):
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

    def trainDataValidator(self):
        try:
            return self.loadClassifier(self.classifierPath)
        except:
            self.trainClassifier()
            return self.loadClassifier(self.classifierPath)

    def processCustomText(self, custom_text):
        custom_tokens = self.removeNoise(word_tokenize(custom_text))
        custom_dict = dict([token, True] for token in custom_tokens)
        return custom_dict

    def textToSentiment(self, classifier, custom_text):
        custom_dict = self.processCustomText(custom_text)
        prediction = classifier.prob_classify(custom_dict)
        probs = {}
        samples = prediction.samples()
        for sample in samples:
            probs[str(sample)] = prediction.prob(sample)
        return probs