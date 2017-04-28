"""
Author: DucVuMinh
Studen From Ha Noi University Of Sience and Technology
20-04-2017
Email: duc0103195@gmail.com
Python version: 2.7

Numpy version: 1.12.0

Pandas version: 0.19.2
"""
# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer

def read_data():
	"""
	read preprocessed data
	Include three file, each file contains a type sentiment sentence
	then return a 3 - dimentions array as following format
	[
	[data positive]
	[data neutral]
	[data negative]
	]
	While "data positive", "data neutral", "data negative" are 2 - dimentions array
	:return:
	"""
	print ("read data ....\n")
	data_pos = pd.read_csv('../../data/smooth/demo/smooth_pos.csv', header=0, \
						   delimiter="\t", quoting=3,encoding='utf-8')
	data_pos['sentiment'] = 1
	data_neg = pd.read_csv('../../data/smooth/demo/smooth_neg.csv', header=0, \
						   delimiter="\t", quoting=3,encoding='utf-8')
	data_neg['sentiment'] = -1
	data_neu = pd.read_csv('../../data/smooth/demo/smooth_neu.csv', header=0, \
						   delimiter="\t", quoting=3,encoding='utf-8')
	data_neu['sentiment'] = 0
	batch_pos1 = np.array( data_pos.iloc[0:340] )
	batch_pos2 = np.array( data_pos.iloc[340:680] )
	batch_pos3 = np.array( data_pos.iloc[680:1020] )
	batch_pos4 = np.array( data_pos.iloc[1020:1360] )
	batch_pos5 = np.array( data_pos.iloc[1360:] )
	batch_pos = np.array([batch_pos1, batch_pos2, batch_pos3, batch_pos4, batch_pos5])

	batch_neu1 = np.array( data_neu.iloc[0:340] )
	batch_neu2 = np.array( data_neu.iloc[340:680] )
	batch_neu3 = np.array( data_neu.iloc[680:1020] )
	batch_neu4 = np.array( data_neu.iloc[1020:1360] )
	batch_neu5 = np.array( data_neu.iloc[1360:] )
	batch_neu = np.array([batch_neu1, batch_neu2, batch_neu3, batch_neu4, batch_neu5])

	batch_neg1 = np.array( data_neg.iloc[0:340] )
	batch_neg2 = np.array( data_neg.iloc[340:680] )
	batch_neg3 = np.array( data_neg.iloc[680:1020] )
	batch_neg4 = np.array(data_neg.iloc[1020:1360] )
	batch_neg5 = np.array( data_neg.iloc[1360:] )
	batch_neg = np.array([batch_neg1, batch_neg2, batch_neg3, batch_neg4, batch_neg5])

	batch = np.array([batch_pos, batch_neu, batch_neg])
	return batch

#Step 1 - cleaning data function

def create_bag_of_words(data):
	"""
	Create bag of words, and vocabulary
	:param data: input data
	:return:
	"""
	print("Creating the bag of words...\n")
	vectorizer = CountVectorizer(analyzer="word", \
								 tokenizer=None, \
								 preprocessor=None, \
								 stop_words=None, \
								 max_features=5000)


	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of
	# strings.
	train_data_features = vectorizer.fit_transform(data)

	# Numpy arrays are easy to work with, so convert the result to an
	# array
	train_data_features = train_data_features.toarray()
	print (train_data_features)
	vocab = vectorizer.get_feature_names()
	# write vocabulary to outer file
	#dist = np.sum(train_data_features, axis=0)
	#vocab_out = pd.DataFrame(data={"vocal": vocab, "count": dist})
	#vocab_out.to_csv("../../data/vocab.csv", index=False, quoting=3 ,encoding='utf-8')
	return vectorizer, train_data_features, vocab


def random_forest(train_data, train_data_features, test_data, vectorizer, vocab):
	"""

	:param batch:
	:return:
	"""
	print("Training the random forest...")
	# Numpy arrays are easy to work with, so convert the result to an
	# array
	from sklearn.ensemble import RandomForestClassifier

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators=110)
	# Fit the forest to the training set, using the bag of words as
	# features and the sentiment labels as the response variable
	#
	# This may take a few minutes to run
	label = np.asarray(train_data["sentiment"], dtype="|S6")
	forest = forest.fit(train_data_features ,  label)

	# Get a bag of words for the test set, and convert to a numpy array
	test = test_data['content']
	test_data_features = vectorizer.transform(test)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make sentiment label predictions
	result = forest.predict(test_data_features)

	num_reviews = len(test)
	error = 0
	error_pos = 0
	error_neg = 0
	error_neu = 0
	label_test = np.asarray(test_data['sentiment'], dtype="|S6")
	for i in range(0, num_reviews):

		if result[i] != label_test[i]:
			if label_test[i] == str(1):
				error_pos = error_pos + 1
			elif label_test[i] == str(-1):
				error_neg = error_neg + 1
			elif label_test[i] == str(0):
				error_neu = error_neu + 1
			error += 1
	print('error: ', error)
	print "error pos: ", (error_pos)
	print  "error neg: ", error_neg
	print "error neu: ", error_neu


def svm_learn(train_data, train_data_features, test_data, vectorizer, vocab):

	from sklearn import svm
	from sklearn.svm import NuSVC

	test = test_data['content']
	label = np.asarray(train_data["sentiment"], dtype="|S6")
	num_reviews = len(test)
	test_data_features = vectorizer.transform(test)
	test_data_features = test_data_features.toarray()

	print ("---------------")
	for nu in [0.7]:
		clf = svm.NuSVC(nu=nu, kernel='linear', decision_function_shape='ovr')
		clf.fit(train_data_features, label)
		result = clf.predict(test_data_features)
		error = 0
		error_pos = 0
		error_neg = 0
		error_neu = 0
		label_test = np.asarray(test_data['sentiment'], dtype="|S6")
		for i in range(0, num_reviews):
			if result[i] != label_test[i]:
				error += 1
				if label_test[i] == str(1):
					error_pos = error_pos + 1
				elif label_test[i] == str(-1):
					error_neg = error_neg + 1
				elif label_test[i] == str(0):
					error_neu = error_neu + 1

		print('error: ',error)
		print "error pos: ",(error_pos)
		print  "error neg: ", error_neg
		print "error neu: ", error_neu

	print ("---------------")

if __name__ == '__main__':
	#step 1: read data then get batch of data
	model = "random_forest"
	batch = read_data()

	for i in np.arange(5):
		train_data = []
		test_data = []
		if i ==0:
			train_data = batch[0][1]
			train_data = np.append(train_data, batch[0][2], axis=0)
			train_data = np.append(train_data, batch[0][3], axis=0)
			train_data = np.append(train_data, batch[0][4], axis=0)
			test_data = batch[0][0]

			train_data = np.append(train_data, batch[1][1], axis=0)
			train_data = np.append(train_data, batch[1][2], axis=0)
			train_data = np.append(train_data, batch[1][3], axis=0)
			train_data = np.append(train_data, batch[1][4], axis=0)
			test_data = np.append(test_data, batch[1][0], axis=0)

			train_data = np.append(train_data, batch[2][1], axis=0)
			train_data = np.append(train_data, batch[2][2], axis=0)
			train_data = np.append(train_data, batch[2][3], axis=0)
			train_data = np.append(train_data, batch[2][4], axis=0)
			test_data = np.append(test_data, batch[2][0], axis=0)

		elif i==1:
			train_data = batch[0][0]
			train_data = np.append(train_data, batch[0][2], axis=0)
			train_data = np.append(train_data, batch[0][3], axis=0)
			train_data = np.append(train_data, batch[0][4], axis=0)
			test_data = batch[0][1]

			train_data = np.append(train_data, batch[1][0], axis=0)
			train_data = np.append(train_data, batch[1][2], axis=0)
			train_data = np.append(train_data, batch[1][3], axis=0)
			train_data = np.append(train_data, batch[1][4], axis=0)
			test_data = np.append(test_data, batch[1][1], axis=0)

			train_data = np.append(train_data, batch[2][0], axis=0)
			train_data = np.append(train_data, batch[2][2], axis=0)
			train_data = np.append(train_data, batch[2][3], axis=0)
			train_data = np.append(train_data, batch[2][4], axis=0)
			test_data = np.append(test_data, batch[2][1], axis=0)
		elif i==2:
			train_data = batch[0][0]
			train_data = np.append(train_data, batch[0][1], axis=0)
			train_data = np.append(train_data, batch[0][3], axis=0)
			train_data = np.append(train_data, batch[0][4], axis=0)
			test_data = batch[0][2]

			train_data = np.append(train_data, batch[1][0], axis=0)
			train_data = np.append(train_data, batch[1][1], axis=0)
			train_data = np.append(train_data, batch[1][3], axis=0)
			train_data = np.append(train_data, batch[1][4], axis=0)
			test_data = np.append(test_data, batch[1][2], axis=0)

			train_data = np.append(train_data, batch[2][0], axis=0)
			train_data = np.append(train_data, batch[2][1], axis=0)
			train_data = np.append(train_data, batch[2][3], axis=0)
			train_data = np.append(train_data, batch[2][4], axis=0)
			test_data = np.append(test_data, batch[2][2], axis=0)
		elif i==3:
			train_data = batch[0][0]
			train_data = np.append(train_data, batch[0][1], axis=0)
			train_data = np.append(train_data, batch[0][2], axis=0)
			train_data = np.append(train_data, batch[0][4], axis=0)
			test_data = batch[0][3]

			train_data = np.append(train_data, batch[1][1], axis=0)
			train_data = np.append(train_data, batch[1][2], axis=0)
			train_data = np.append(train_data, batch[1][0], axis=0)
			train_data = np.append(train_data, batch[1][4], axis=0)
			test_data = np.append(test_data, batch[1][3], axis=0)

			train_data = np.append(train_data, batch[2][1], axis=0)
			train_data = np.append(train_data, batch[2][2], axis=0)
			train_data = np.append(train_data, batch[2][0], axis=0)
			train_data = np.append(train_data, batch[2][4], axis=0)
			test_data = np.append(test_data, batch[2][3], axis=0)
		else:
			train_data = batch[0][1]
			train_data = np.append(train_data, batch[0][2], axis=0)
			train_data = np.append(train_data, batch[0][3], axis=0)
			train_data = np.append(train_data, batch[0][1], axis=0)
			test_data = batch[0][4]

			train_data = np.append(train_data, batch[1][1], axis=0)
			train_data = np.append(train_data, batch[1][2], axis=0)
			train_data = np.append(train_data, batch[1][3], axis=0)
			train_data = np.append(train_data, batch[1][0], axis=0)
			test_data = np.append(test_data, batch[1][4], axis=0)

			train_data = np.append(train_data, batch[2][1], axis=0)
			train_data = np.append(train_data, batch[2][2], axis=0)
			train_data = np.append(train_data, batch[2][3], axis=0)
			train_data = np.append(train_data, batch[2][0], axis=0)
			test_data = np.append(test_data, batch[2][4], axis=0)
		train_data = pd.DataFrame(data = train_data, columns= ['content', 'sentiment'])
		test_data = pd.DataFrame(data=test_data, columns=['content', 'sentiment'])
		vectorizer, train_data_features, vocab = create_bag_of_words(train_data['content'])
		print( train_data_features.shape )
		if model == "random_forest":
			random_forest(train_data, train_data_features, test_data, vectorizer, vocab)
		elif model =="svm":
			svm_learn(train_data, train_data_features, test_data, vectorizer, vocab)
		else:pass

