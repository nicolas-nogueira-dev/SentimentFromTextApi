from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import time
import re, string

class Core():
    def __init__(self, folderpath, sentiments):
        self.folderPath = folderpath
        self.sentiments = sentiments

    def changeFolderPath(self, new):
        self.folderPath = new

    def changeSentiments(self, new):
        self.sentiments = new

    def removeNoise(self, tweet_tokens, stop_words = ()):
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    def getSentimentItems(self, dataset, index, sentiment):
        list = []
        for i in dataset:
            try:
                if i[index['sentiment']] == sentiment:
                    list.append(str(i[index['text']]))
            except:
                pass
        return list

    def getTokens(self, listText):
        finalList = []
        for i in listText:
            finalList += [word_tokenize(i)]
        return finalList

    def getCleanTokens(self, listTokens, stop_words):
        finalText = []
        for tokens in listTokens:
            finalText.append(self.removeNoise(tokens, stop_words))
        return finalText

    def saveCleanData(self, cleanData, sentiment, fileName):
        print("")
        print(f"-> Saving the {sentiment} data...")
        start_time = time.time()
        textData = ""
        for i in cleanData:
            for j in range(len(i)):
                if j == len(i)-1:
                    textData += str(i[j])
                else:
                    textData += str(i[j]) + ","
            textData += ";" + str(sentiment) + "\n"
        with open(self.folderPath + '/' + fileName+".txt", "w", encoding="utf8") as file:
            file.write(textData)
        print(f"-> %s seconds for saving the {sentiment} data" % (time.time() - start_time))

    def getCleanDatasetTokens(self, dataSetPath):
        # Extract the raw data
        with open(dataSetPath, "r", encoding="utf8") as file:
            data = file.read()
        # Convert the raw data into 1d list
        list = data.split("\n")
        del list[-1]
        # Convert the 1d list into 2d list
        process = []
        for i in list:
            process.append(i.split(";"))
        # Convert the 2d list into 3d list
        listSize = len(process)
        for i in range(listSize):
            process[i][0] = process[i][0].split(",")
        return process

    def getCleanDataset(self):
        #~~~~~~-> Getting the dataset...
        rawDataset = {}
        for sentiment in self.sentiments:
            rawDataset[sentiment] = self.getCleanDatasetTokens(self.folderPath + '/' + str(sentiment) +'_cleaned_tokens.txt')
        #~~~~~~-> Making the cleaned tokens list...
        clean_tokens_dict = {}
        for sentiment in self.sentiments:
            clean_tokens_dict[sentiment] = self.getTokensFromDataset(rawDataset[sentiment])
        #~~~~~~-> Making the dataset...
        tokens_for_model = {}
        for sentiment in self.sentiments:
            tokens_for_model[sentiment] = self.getTextsForModel(clean_tokens_dict[sentiment])
        dataset = {}
        for sentiment in self.sentiments:
            dataset[sentiment] = [(tweet_dict, sentiment) for tweet_dict in tokens_for_model[sentiment]]
        return dataset

    def getTokensFromDataset(self, dataset):
        list = []
        for i in dataset:
            list.append(i[0])
        return list

    def getTextsForModel(self, cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    def getDatasetInfos(self):
        #~~~~~~-> Getting the dataset...
        dataset = {}
        for sentiment in self.sentiments:
            filename = self.folderPath + '/' + sentiment + '_cleaned_tokens.txt'
            dataset[sentiment] = self.getCleanDataset()

        infos = {}
        infos['datasetSize'] = 0
        for sentiment in self.sentiments:
            infos[sentiment] = len(dataset[sentiment])
            infos['datasetSize'] += len(dataset[sentiment])
        return infos
