from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier

import csv
import pickle
import time
import re, string, random

class core():
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

    def getSentimentItems(self, dataset, sentiment):
        Text = []
        for i in dataset:
            try:
                if i[1] == sentiment:
                    Text.append(str(i[0]))
            except:
                pass
        return Text

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

    def saveCleanData(self, cleanData, sentiment, folderPath, fileName):
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
        with open(folderPath + fileName+".txt", "w", encoding="utf8") as file:
            file.write(textData)
        print(f"-> %s seconds for saving the {sentiment} data" % (time.time() - start_time))

    def getCleanDataset(dataSetPath):
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

    def getTokensFromDataset(self, dataset):
        list = []
        for i in dataset:
            list.append(i[0])
        return list

    def getTextsForModel(self, cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    def trainClassifier(self, sentiments, folderPath):
        #~~~~~~-> Getting the dataset...
        dataset = {}
        for sentiment in sentiments:
            dataset[sentiment] = self.getCleanDataset(folderPath + str(sentiment) +'_cleaned_tokens.txt')
        #~~~~~~-> Making the cleaned tokens list...
        clean_tokens_list
        for sentiment in sentiments:

        positive_cleaned_tokens_list = getTokensFromDataset(dataset_positive)
        negative_cleaned_tokens_list = getTokensFromDataset(dataset_negative)
        #~~~~~~-> Making the dataset...
        positive_tokens_for_model = getTextsForModel(positive_cleaned_tokens_list)
        negative_tokens_for_model = getTextsForModel(negative_cleaned_tokens_list)
        positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
        negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]
        dataset = positive_dataset + negative_dataset
        random.shuffle(dataset)
        train_data = dataset
        #~~~~~~-> Training the model...
        classifier = NaiveBayesClassifier.train(train_data)
        with open('classifier.pickle', 'wb') as f:
            pickle.dump(classifier, f)

    def loadClassifier(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def trainDataValidator(self):
        try:
            return self.loadClassifier('classifier.pickle')
        except:
            self.trainClassifier()
            return self.loadClassifier('classifier.pickle')

    def textToSentiment(custom_text):
        classifier = trainDataValidator()

        custom_tokens = removeNoise(word_tokenize(custom_text))
        custom_dict = dict([token, True] for token in custom_tokens)

        prediction = classifier.prob_classify(custom_dict)

        probs = {}
        samples = prediction.samples()
        print(samples)
        for sample in samples:
            probs[str(sample)] = prediction.prob(sample)

        return probs