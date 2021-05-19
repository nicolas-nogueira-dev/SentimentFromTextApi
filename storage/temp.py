from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import time
import re, string, random

def getCleanTokens(listTokens, stop_words):
    finalText = []
    for tokens in listTokens:
        finalText.append(remove_noise(tokens, stop_words))
    return finalText
def remove_noise(tweet_tokens, stop_words = ()):

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
def get_texts_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
def getDataset(dataSetPath):
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

if __name__ == "__main__":
    print("-> Getting the dataset...")
    start_time = time.time()
    # Get the training dataset
    dataset_positive = getDataset("dataset/positive_cleaned_tokens.txt")
    dataset_negative = getDataset("dataset/negative_cleaned_tokens.txt")
    dataset_neutral = getDataset("dataset/neutral_cleaned_tokens.txt")
    print("-> %s seconds for get the dataset" % (time.time() - start_time))
    print("")
    print("-> Making the cleaned tokens list...")
    start_time = time.time()
    positive_cleaned_tokens_list = getTokensFromDataset(dataset_positive)
    negative_cleaned_tokens_list = getTokensFromDataset(dataset_negative)
    neutral_cleaned_tokens_list = getTokensFromDataset(dataset_neutral)
    print("-> %s seconds for making the cleaned tokens list" % (time.time() - start_time))
    print("")
    print("-> Making the dataset...")
    start_time = time.time()
    # Convert to a dict dataset
    positive_tokens_for_model = get_texts_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_texts_for_model(negative_cleaned_tokens_list)
    neutral_tokens_for_model = get_texts_for_model(neutral_cleaned_tokens_list)
    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]
    neutral_dataset = [(tweet_dict, "Neutral") for tweet_dict in neutral_tokens_for_model]
    dataset = positive_dataset + negative_dataset + neutral_dataset
    datasetSize = len(dataset)
    indexTraining = int(datasetSize*0.7)
    random.shuffle(dataset)
    train_data = dataset[:indexTraining]
    test_data = dataset[indexTraining:]
    print(f"-> Dataset size : {datasetSize}")
    print("-> %s seconds to make the dataset" % (time.time() - start_time))
    print("")
    print("-> Training the model...")
    start_time = time.time()
    classifier = NaiveBayesClassifier.train(train_data)
    print("-> %s seconds to train the model" % (time.time() - start_time))
    '''
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))
    '''

    custom_tweet = "I think I don't really like this man but he is nice sometime and I really like this jobs"

    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    custom_text = dict([token, True] for token in custom_tokens)

    print("-> The custom text : ", custom_tweet)
    print("-> The prediction  : ", classifier.classify(custom_text))
    print(classifier.show_most_informative_features(10))
