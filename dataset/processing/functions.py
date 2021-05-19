##############################
#     Functions use in :     #
#     processCleanTokens.py  #
#     TextToSentiment.py     #
##############################
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier

from functions import *

import csv
import pickle
import time
import re, string, random

# Global functions :
def removeNoise(tweet_tokens, stop_words = ()):

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

# processCleanTokens.py functions :
def getRawDataset(path):
    list = []
    with open(path, newline='', encoding='utf8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            list.append(row)
    return list
def getTextFromSentiment(dataset, sentiment):
    Text = []
    for i in dataset:
        try:
            if i[1] == sentiment:
                Text.append(str(i[0]))
        except:
            pass
    return Text
def getTokens(listText):
    finalList = []
    for i in listText:
        finalList += [word_tokenize(i)]
    return finalList
def getCleanTokens(listTokens, stop_words):
    finalText = []
    for tokens in listTokens:
        finalText.append(removeNoise(tokens, stop_words))
    return finalText
def saveCleanData(cleanData, sentiment, fileName):
    print("")
    print(f"-> Saving the {sentiment} data...")
    start_time = time.time()
    folderPath = ""
    textData = ""
    for i in cleanData:
        for j in range(len(i)):
            if j == len(i)-1:
                textData += str(i[j])
            else:
                textData += str(i[j]) + ","
        textData += ";" + "positive" + "\n"
    with open(folderPath + fileName+".txt", "w", encoding="utf8") as file:
        file.write(textData)
    print(f"-> %s seconds for saving the {sentiment} data" % (time.time() - start_time))

# TextToSentiment.py functions :
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
def getTokensFromDataset(dataset):
    list = []
    for i in dataset:
        list.append(i[0])
    return list
def getTextsForModel(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
def trainClassifier():
    #~~~~~~-> Getting the dataset...
    dataset_positive = getCleanDataset("dataset/positive_cleaned_tokens.txt")
    dataset_negative = getCleanDataset("dataset/negative_cleaned_tokens.txt")
    #~~~~~~-> Making the cleaned tokens list...
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
def trainDataValidator():
    try:
        with open('classifier.pickle', 'rb') as f:
            return pickle.load(f)
    except:
        trainClassifier()
        with open('classifier.pickle', 'rb') as f:
            return pickle.load(f)
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
