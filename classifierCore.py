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
        
        train_data = []
        for i in finalDataset:
            for j in i:
                train_data.append(j)
        random.shuffle(train_data)
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

    def textToSentiment(self, classifier, custom_text):

        custom_dict = self.processCustomText(custom_text)

        prediction = classifier.prob_classify(custom_dict)

        probs = {}
        samples = prediction.samples()
        for sample in samples:
            probs[str(sample)] = prediction.prob(sample)

        return probs

###############################################
'''
folderPath = 'dataset5'
sentiments = ['positive','negative']

main = ClassifierCore(folderPath, sentiments)

main.trainClassifier()

text = "Hi, I last heard from Cameron from Cigna on Wednesday 28th asking for documentation on Ping Federate’s OIDC implementation, which is the flow ConnectNow supports. I supplied a link to the documentation that we have been following from Ping here hoping that this would be useful to your team. We have still not yet been granted to Cigna’s test systems and are relying on our own private instance of Ping Federate. I have configured this such that it issues ConnectNow with a JWT containing email, first name, and last name from AD. However this is all an assumption as to the behaviour and configuration as again – we have received no documentation or instruction from Cigna as to how your implementation is configured, so there is a distinct chance that ours is dramatically different. I would gently remind you that ConnectNow is fully functional without SSO. Users must simply fill out a few fields and activate via email prior to using the system as you would with many other systems. Once logged in, users will only be asked to log in again if they do not re-visit ConnectNow within 5 days. As discussed, this 5 day limit can be increased to any length at your discretion, effectively providing a workaround to this issue whilst we wait for Cigna’s test instance and any final development that must be completed as a result of any differences. On our previous call I stated that due to delays in even getting access to a Ping Federate instance, combined with having to skill-up on a brand-new proprietary system, we could only have development completed by the end of May. The good news is - internally - we’re actually around a week ahead of this target – with our internal instance of Ping Federate. At the time, we were discussing extending the logon expiration for as long as necessary so that this was no long a time-sensitive issue and could be implemented once completed. I can have this configured for you in a few minutes. Kind regards,"

print(main.textToSentiment(text))
'''