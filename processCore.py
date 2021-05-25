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

class ProcessCore(Core):

    def __init__(self, folderpath, sentiments):
        super().__init__(folderpath, sentiments)

    def testSuper(self):
        print(self.folderPath)

    def getRawDataset(self, path, settings):
        if settings['type'] == 'csv':
            list = []
            with open(path, newline='', encoding=settings['encoding']) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=settings['delimiter'], quotechar=settings['quotechar'])
                for row in spamreader:
                    list.append(row)
            return list

    def processCleanTokens(self, settings):
        print('-> Getting the dataset...')
        start_time = time.time()
        dataset = self.getRawDataset(self.folderPath+settings['filePath'], settings['type'], settings)
        print('-> %s seconds for get the dataset' % (time.time() - start_time))
        print('')
        print('-> Getting the tokens...')
        tokens = {}
        start_time = time.time()
        for sentiment in self.sentiments:
            print(f'-> Getting the {sentiment} tokens...')
            start_time_tmp = time.time()
            tokens[sentiment] = self.getSentimentItems(dataset, settings['sentimentText'][sentiment])
            print(f'-> {time.time() - start_time_tmp} seconds for get the {sentiment} tokens')
        print('-> %s seconds for get the positive/negative tokens' % (time.time() - start_time))

        # Define the stop words
        stop_words = stopwords.words(settings['stopWord'])


main = ProcessCore('dataset3/', ['positive','negative','neutral'])

main.processCleanTokens({'filePath':'dataset4.csv',
                         'type':'csv',
                         'encoding':'utf8',
                         'delimiter':',',
                         'quotechar':'"',
                         'sentimentText':{'positive':'POSITIVE',
                                          'negative':'NEGATIVE',
                                          'neutral':'NEUTRAL'
                                         },
                         'stopWord':'english',
                         })