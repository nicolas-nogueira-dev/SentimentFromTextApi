from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier
from core import *

import pickle
import random

class ClassifierCore(Core):

    def __init__(self, folderpath, sentiments):
        super().__init__(folderpath, sentiments)

    def trainClassifier(self):
        dataset = self.getCleanDataset()
        finalDataset = []
        for key in dataset:
            finalDataset += [dataset[key]]
        random.shuffle(finalDataset)
        train_data = finalDataset
        print(len(finalDataset))
        #~~~~~~-> Training the model...
        classifier = NaiveBayesClassifier.train(train_data)
        self.saveClassifier(classifier,'classifier.pickle')

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

###############################################

folderPath = 'dataset5'
sentiments = ['positive','negative']

main = ClassifierCore(folderPath, sentiments)

main.trainClassifier()

print(main.textToSentiment('I love chocolat'))