from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize as wordTokenize
from nltk import classify, NaiveBayesClassifier

import pickle
import re, string, random

def removeNoise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)
        if tag.startswith('NN'):
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
def getCleanDataset(dataSetPath):
    # Extract the raw data
    with open(dataSetPath, 'r', encoding='utf8') as file:
        data = file.read()
    # Convert the raw data into 1d list
    list = data.split('\n')
    del list[-1]
    # Convert the 1d list into 2d list
    process = []
    for i in list:
        process.append(i.split(';'))
    # Convert the 2d list into 3d list
    listSize = len(process)
    for i in range(listSize):
        process[i][0] = process[i][0].split(',')
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
    dataset_positive = getCleanDataset('dataset/positive_cleaned_tokens.txt')
    dataset_negative = getCleanDataset('dataset/negative_cleaned_tokens.txt')
    dataset_neutral = getCleanDataset('dataset/neutral_cleaned_tokens.txt')
    #~~~~~~-> Making the cleaned tokens list...
    positive_cleaned_tokens_list = getTokensFromDataset(dataset_positive)
    negative_cleaned_tokens_list = getTokensFromDataset(dataset_negative)
    neutral_cleaned_tokens_list = getTokensFromDataset(dataset_neutral)
    #~~~~~~-> Making the dataset...
    positive_tokens_for_model = getTextsForModel(positive_cleaned_tokens_list)
    negative_tokens_for_model = getTextsForModel(negative_cleaned_tokens_list)
    neutral_tokens_for_model = getTextsForModel(neutral_cleaned_tokens_list)
    positive_dataset = [(tweet_dict, 'Positive') for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, 'Negative') for tweet_dict in negative_tokens_for_model]
    neutral_dataset = [(tweet_dict, 'Neutral') for tweet_dict in neutral_tokens_for_model]
    dataset = positive_dataset + negative_dataset + neutral_dataset
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
def textToSentiment(classifier, custom_text):

    custom_tokens = removeNoise(wordTokenize(custom_text))
    custom_dict = dict([token, True] for token in custom_tokens)

    prediction = classifier.prob_classify(custom_dict)

    probs = {}
    samples = prediction.samples()
    for sample in samples:
        probs[str(sample)] = round(prediction.prob(sample),4)

    return {'original_text':custom_text,'prediction':probs}
def getDatasetInfos():
    #~~~~~~-> Getting the dataset...
    dataset_positive = getCleanDataset('mixed-dataset/positive_cleaned_tokens.txt')
    dataset_negative = getCleanDataset('mixed-dataset/negative_cleaned_tokens.txt')
    #~~~~~~-> Making the cleaned tokens list...
    positive_cleaned_tokens_list = getTokensFromDataset(dataset_positive)
    negative_cleaned_tokens_list = getTokensFromDataset(dataset_negative)
    #~~~~~~-> Making the dataset...
    positive_tokens_for_model = getTextsForModel(positive_cleaned_tokens_list)
    negative_tokens_for_model = getTextsForModel(negative_cleaned_tokens_list)
    positive_dataset = [(tweet_dict, 'Positive') for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, 'Negative') for tweet_dict in negative_tokens_for_model]
    dataset =dataset_positive + dataset_negative

    infos = {}
    infos['datasetSize'] = len(dataset)
    infos['posDatasetSize'] = len(dataset_positive)
    infos['negDatasetSize'] = len(dataset_negative)

    return infos
