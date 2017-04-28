#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords



class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        #review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        #review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        #if remove_stopwords:
        #    stops = set(stopwords.words("english"))
        #    words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
                #print ( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                # remove_stopwords ) )
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=False ))
    return clean_reviews



if __name__ == '__main__':

    # Read data from files
    data_pos = pd.read_csv('../data/smooth_pos2.csv',header=0, \
                    delimiter="\t", quoting=3)
    data_neg = pd.read_csv('../data/smooth_neg2.csv',header=0, \
                    delimiter="\t", quoting=3)
    data_neu = pd.read_csv('../data/smooth_neu2.csv',header=0, \
                    delimiter="\t", quoting=3)
    data_pos['sentiment'] = 1
    data_neg['sentiment'] = -1
    data_neu['sentiment'] = 0
    train_data = data_pos.iloc[1:1201]
    train_data = train_data.append(data_neg.iloc[1:1201])
    train_data = train_data.append(data_neu.iloc[1:1201])
    test_data = data_pos.iloc[1200:]
    test_data = test_data.append(data_neg.iloc[1200:])
    test_data= test_data.append(data_neu.iloc[1200:])
    all_data = data_pos.append(data_neg)
    all_data = all_data.append(data_neu)
    bonus_data =  pd.read_csv('./data/smooth_cong_nghe2.csv',header=0, \
                    delimiter="\t", quoting=3)
    bonus_data2 = pd.read_csv('./data/smooth_out1_2.csv',header=0, \
                    delimiter="\t", quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    print ("Read %d labeled train content, %d labeled test content, " 
     % (train_data["content"].size,
     test_data["content"].size)
    )


    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from training set")
    for review in train_data["content"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    for review in test_data["content"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    for review in bonus_data["content"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    for review in bonus_data2["content"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 400    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 5          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print ("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    #print (model.doesnt_match("em ngon bền".split()) )
    s = "tốt"
    words = model.most_similar(s.decode('utf-8'))
    for w in words:
        print ( w[0].encode('utf-8') )

    model.save(model_name)


    