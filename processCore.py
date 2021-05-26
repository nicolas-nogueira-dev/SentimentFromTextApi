from nltk.corpus import stopwords
from core import *

import csv
import time

class ProcessCore(Core):

    def __init__(self, folderpath, sentiments):
        super().__init__(folderpath, sentiments)

    def testSuper(self):
        print(self.folderPath)

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
        print(f'-> Getting the dataset ({filePath})...')
        start_time = time.time()
        dataset = self.getRawDataset(filePath, settings)
        print('-> %s seconds for get the dataset' % (time.time() - start_time))

        print('')
        print('-> Getting the raw text...')
        rawText = {}
        start_time = time.time()
        for sentiment in self.sentiments:
            print(f'-> Getting the {sentiment} raw text...')
            start_time_tmp = time.time()
            rawText[sentiment] = self.getSentimentItems(dataset, settings['indexExtract'],settings['sentimentText'][sentiment])
            print(f'-> {time.time() - start_time_tmp} seconds for get the {sentiment} raw text')
        print('-> %s seconds for get the raw text' % (time.time() - start_time))

        print('')
        print('-> Getting the tokens...')
        tokens = {}
        start_time = time.time()
        for sentiment in self.sentiments:
            print(f'-> Getting the {sentiment} tokens...')
            start_time_tmp = time.time()
            tokens[sentiment] = self.getTokens(rawText[sentiment])
            print(f'-> {time.time() - start_time_tmp} seconds for get the {sentiment} tokens')
        print('-> %s seconds for get the raw text' % (time.time() - start_time))

        stop_words = stopwords.words(settings['stopWord'])

        print('')
        print('-> Cleaning the tokens...')
        cleaned_tokens = {}
        start_time = time.time()
        for sentiment in self.sentiments:
            print(f'-> Cleaning the {sentiment} tokens...')
            start_time_tmp = time.time()
            cleaned_tokens[sentiment] = self.getCleanTokens(tokens[sentiment], stop_words)
            print(f'-> {time.time() - start_time_tmp} seconds for clean the {sentiment} tokens')
        print('-> %s seconds for clean the tokens' % (time.time() - start_time))

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

######################################################################

folderPath = 'dataset5'
sentiments = ['positive','negative']

main = ProcessCore(folderPath, sentiments)

main.preProcessDataset({'filePath':'TranslatedDigikalaDataset.csv',
                        'newFilePath':str(folderPath+'.csv'),
                        'type':'csv',
                        'encoding':'utf8',
                        'delimiter':',',
                        'quotechar':'"',
                        'delimiterSave':',',
                        'quotecharSave':'"',
                        'sentimentText':{'1':'POSITIVE',
                                         '0':'NEGATIVE',
                                        },
                        'indexExtract': {'text':0,'sentiment':1},
                        'indexSaving': {'text':0,'sentiment':1},
                        })


main.processCleanTokens({'filePath':str(folderPath+'.csv'),
                         'type':'csv',
                         'encoding':'utf8',
                         'delimiter':',',
                         'quotechar':'"',
                         'sentimentText':{'positive':'POSITIVE',
                                          'negative':'NEGATIVE',
                                          'neutral':'NEUTRAL',
                                         },
                         'stopWord':'english',
                         'indexExtract': {'text':0,'sentiment':1},
                         })
