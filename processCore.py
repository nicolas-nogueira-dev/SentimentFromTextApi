from nltk.corpus import stopwords
from core import *

import csv

class ProcessCore(Core):

    def __init__(self, folderpath, sentiments):
        super().__init__(folderpath, sentiments)

    def getRawDataset(self, path, settings):
        if settings['type'] == 'csv':
            list = []
            with open(path, newline='', encoding=settings['encoding']) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=settings['delimiter'], quotechar=settings['quotechar'], quoting=csv.QUOTE_MINIMAL)
                for row in spamreader:
                    list.append(row)
            return list

    def processCleanTokens(self, settings):
        filePath = self.folderPath + '/' + settings['filePath']
        dataset = self.getRawDataset(filePath, settings)
        rawText = {}
        for sentiment in self.sentiments:
            rawText[sentiment] = self.getSentimentItems(dataset, settings['indexExtract'],settings['sentimentText'][sentiment])
        tokens = {}
        for sentiment in self.sentiments:
            tokens[sentiment] = self.getTokens(rawText[sentiment])
        stop_words = stopwords.words(settings['stopWord'])
        cleaned_tokens = {}
        for sentiment in self.sentiments:
            cleaned_tokens[sentiment] = self.getCleanTokens(tokens[sentiment], stop_words)
        for sentiment in self.sentiments:
            self.saveCleanData(cleaned_tokens[sentiment], sentiment, str(sentiment) + '_cleaned_tokens')

    def makeDatasetViaSentiment(self, list, settings):
        finalList =[]
        for i in list:
            for key in settings['sentimentText']:
                if str(i[settings['indexExtract']['sentiment']]) == str(key) :
                    finalList += [[str(i[settings['indexExtract']['text']]),str(settings['sentimentText'][key])]]
        return finalList

    def saveProcessedDataset(self, finalList, settings):
        filePath = self.folderPath + '/' + settings['newFilePath']
        with open(filePath, 'w', encoding=settings['encoding']) as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=settings['delimiterSave'], quotechar=settings['quotecharSave'], quoting=csv.QUOTE_MINIMAL)
            for i in finalList:
                spamwriter.writerow([i[settings['indexSaving']['text']],i[settings['indexSaving']['sentiment']]])


    def preProcessDataset(self, settings):
        filePath = self.folderPath + '/' + settings['filePath']
        dataset = self.getRawDataset(filePath, settings)
        processedDataset = self.makeDatasetViaSentiment(dataset, settings)
        self.saveProcessedDataset(processedDataset,settings)
