from functions import *

if __name__ == '__main__':
    print('-> Getting the dataset...')
    start_time = time.time()
    # Get the training dataset
    dataset = getRawDataset("dataset4.csv")
    print('-> %s seconds for get the dataset' % (time.time() - start_time))
    print('')
    print('-> Getting the tokens...')
    start_time = time.time()
    print('-> Getting the positive tokens...')
    start_time_pos = time.time()
    positiveText = getTextFromSentiment(dataset, 'POSITIVE')
    positive_text_tokens = getTokens(positiveText)
    print('-> %s seconds for get the positive tokens' % (time.time() - start_time_pos))
    print('-> Getting the negative tokens...')
    start_time_neg = time.time()
    negativeText = getTextFromSentiment(dataset, 'NEGATIVE')
    negative_text_tokens = getTokens(negativeText)
    print('-> %s seconds for get the negative tokens' % (time.time() - start_time_neg))
    print('-> %s seconds for get the positive/negative tokens' % (time.time() - start_time))

    # Define the stop words
    stop_words = stopwords.words('english')
    
    print('')
    print('-> Cleaning the tokens...')
    start_time = time.time()
    print('-> Cleaning the positive tokens...')
    start_time_pos = time.time()
    positive_cleaned_tokens_list = getCleanTokens(positive_text_tokens, stop_words)
    print('-> %s seconds for clean the positive tokens' % (time.time() - start_time_pos))
    print('-> Cleaning the negative tokens...')
    start_time_neg = time.time()
    negative_cleaned_tokens_list = getCleanTokens(negative_text_tokens, stop_words)
    print('-> %s seconds for clean the negative tokens' % (time.time() - start_time_neg))
    print('-> %s seconds for clean the tokens' % (time.time() - start_time))

    saveCleanData(positive_cleaned_tokens_list, 'positive', 'positive_cleaned_tokens')
    saveCleanData(negative_cleaned_tokens_list, 'negative', 'negative_cleaned_tokens')
