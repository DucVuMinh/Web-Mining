# Load a pre-trained model
# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import os
from word2vec import KaggleWord2VecUtility

import numpy as np  # Make sure that numpy is imported



def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    vocab = []

    for w in index2word_set:
        vocab.append(w.encode('utf-8'))
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in vocab:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word.decode('utf-8')])
        else:
            pass
            ##print("---")
            #print ( word )
    #
    # Divide the result by the number of words to get the average

    if (nwords != 0.0):
        featureVec = np.divide(featureVec, 1)
    else:
        for w in words:
            print w

    # print (featureVec)
    return featureVec
def makeFeatureVecVs2(words, model, num_features, num_fea_word_emb):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    num_w = num_features/num_fea_word_emb
    arr_w = words.split(" ")
    feature = []
    featureVec = np.zeros((num_fea_word_emb,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    vocab = []

    for w in index2word_set:
        vocab.append(w.encode('utf-8'))
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    count = 0
    for word in words:

        if word in vocab:
            if count < num_w -1:
                feature = np.append(feature, model[word.decode('utf-8')])
                count = count + 1
            else:
                featureVec = np.add(featureVec, model[word.decode('utf-8')])
                nwords = nwords + 1.
    #
    # Divide the result by the number of words to get the average

    if (nwords != 0.0):
        featureVec = np.divide(featureVec, 1)
        feature = np.append(feature, featureVec)
    else:
        feature = np.append(feature, np.zeros((num_features-count*400), dtype="float32"))
    return feature


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = int(0)
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review

        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        #
        # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs
def getAvgFeatureVecsVs2(reviews, model, num_features ,num_fea_word_emb):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = int(0)
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #/
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review

        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVecVs2(review, model, num_features,num_fea_word_emb)
        #
        # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs

def svm_learn(train_vec, train_label, test_vec, test_label):
    from sklearn import svm
    from sklearn.svm import NuSVC

    num_reviews = len(test_label)
    print ("---------------")
    for nu in [0.53, 0.56, 0.6, 0.63, 0.66, 0.7]:
        clf = svm.NuSVC(nu=nu, kernel='linear', decision_function_shape='ovr')
        clf.fit(train_vec, train_label)
        result = clf.predict(test_vec)
        error = 0
        for i in range(0, num_reviews):
            if result[i] != test_label.iloc[i]:
                error += 1
        print('error: ')
        print(error)

    print ("---------------")


if __name__ == '__main__':
    model = Word2Vec.load("300features_40minwords_10context")
    num_features = 400
    data_pos = pd.read_csv('../data/smooth_pos2.csv', header=0, \
    						   delimiter="\t", quoting=3)
    data_pos['sentiment'] = 1
    data_neg = pd.read_csv('../data/smooth_neg2.csv', header=0, \
    						   delimiter="\t", quoting=3)
    data_neg['sentiment'] = -1
    data_neu = pd.read_csv('../data/smooth_neu2.csv', header=0, \
                           delimiter="\t", quoting=3)
    data_neu['sentiment'] = 0
    train_data = data_pos.iloc[1:1201]
    train_data = train_data.append(data_neg.iloc[1:1201])
    train_data = train_data.append(data_neu.iloc[1:1201])
    test_data = data_pos.iloc[1200:]
    test_data = test_data.append(data_neg.iloc[1200:])
    test_data = test_data.append(data_neu.iloc[1200:])
    all_data = data_pos.append(data_neg)
    all_data = all_data.append(data_neu)
    # step 1 : load trained model

    # step 2 :
    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop word
    # removal.

    clean_train_reviews = []
    for review in train_data["content"]:
        clean_train_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, \
                                                                            remove_stopwords=True))

    trainDataVecs = getAvgFeatureVecs(train_data["content"], model, num_features)

    print "Creating average feature vecs for test reviews"
    clean_test_reviews = []
    for review in test_data["content"]:
        clean_test_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, \
                                                                           remove_stopwords=True))

    testDataVecs = getAvgFeatureVecs(test_data["content"], model, num_features)

    # Fit a random forest to the training data, using 100 trees
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=100)

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(trainDataVecs, train_data["sentiment"])

    # Test & extract results
    result = forest.predict(testDataVecs)

    label_test = test_data['sentiment']
    num_reviews = label_test.shape[0]
    error = 0
    for i in range(0, num_reviews):
        if str(result[i]) != str(label_test.iloc[i]):
            error += 1
            print ("--")
            print (result[i])
            print (label_test.iloc[i])

    print('error: ')
    print(error)
    svm_learn(trainDataVecs, train_data["sentiment"], testDataVecs, test_data['sentiment'])
