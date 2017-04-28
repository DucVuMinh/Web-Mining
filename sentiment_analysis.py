# -*- coding: utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer

def read_raw_data():
	data_pos = pd.read_csv('./token/SA-training_positive.txt', sep="\n", header = None, names = ['content'] )
	data_neg = pd.read_csv('./token/SA-training_negative.txt', sep="\n", header = None, names = ['content'])
	data_neu = pd.read_csv('./token/SA-training_neutral.txt', sep="\n", header = None, names = ['content'])
	print (data_pos.shape)
	print (data_neg.shape)
	print (data_neu.shape)
#read data from file and generate train_data, test_data
def read_data():
	"""

	:return:
	"""
	print ("read data ....\n")
	data_pos = pd.read_csv('./data/smooth/smooth/smooth_pos.csv', header=0, \
						   delimiter="\t", quoting=3,encoding='utf-8')
	data_pos['sentiment'] = 1
	data_neg = pd.read_csv('./data/smooth/smooth/smooth_neg.csv', header=0, \
						   delimiter="\t", quoting=3,encoding='utf-8')
	data_neg['sentiment'] = -1
	data_neu = pd.read_csv('./data/smooth/smooth/smooth_neu.csv', header=0, \
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

def preprocess(raw_comment):
	"""
	Clearning and processing data
	:param raw_comment: a raw data, it contains some special charactor, upper case
	:return:
	"""
	letter = re.sub("😁|👍|😤|👅|👏|❤|😍|💲|🤔|😉|💪|😈|👿|👽|😌|😋|😅|😅|😂", " ", raw_comment,flags=re.UNICODE)
	letter = \
		re.sub("\\.|,|/|-|\\?|<|>|/|:|;|'|\\[|\\]|\\{|\\}|\\\\|\\||=|\\+|-|\\(|\\)|\\*|&|\\^|%|$|#|@|!|~|`",
			   " ", letter, flags=re.UNICODE)
	letter = re.sub('“|”|"', " ", letter)
	letter = re.sub("(^\d+\s)|(\s\d+\s)", " num ", letter)
	letter = re.sub("(^\d+\s)|(\s\d+\s)", " num ", letter)
	letter = re.sub("\s\w\s", " ", letter)
	letter = re.sub("\s+", " ", letter)
	letter = letter.lower()
	return letter
def generate_smooth_data(data_pos, data_neu, data_neg):
	smooth_data_pro = []
	for d in data_neu['content']:
		foo = preprocess(d)
		smooth_data_pro.append(foo)
	smooth_neu  = pd.DataFrame(data = {"content":smooth_data_pro})
	smooth_neu.to_csv("smooth_neu.csv", index=False, quoting=3 )

def create_bag_of_words(data):
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
	with open("test_matrix.txt", "w") as text_file:
		for i in range(0, train_data_features.shape[0]):
			text_file.write(train_data_features[i])
	vocab = vectorizer.get_feature_names()

	# write vocabulary to outer file
	dist = np.sum(train_data_features, axis=0)
	vocab_out = pd.DataFrame(data={"vocal": vocab, "count": dist})
	vocab_out.to_csv("./data/vocab.csv", index=False, quoting=3 ,encoding='utf-8')
	return vectorizer, train_data_features, vocab


def random_forest(train_data, train_data_features, test_data, vectorizer, vocab):
	"""

	:param batch:
	:return:
	"""
	print("Training the random forest...")
	train_data_features = vectorizer.fit_transform(train_data['content'])

	# Numpy arrays are easy to work with, so convert the result to an
	# array
	train_data_features = train_data_features.toarray()
	from sklearn.ensemble import RandomForestClassifier

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators=100)
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
	label_test = np.asarray(test_data['sentiment'], dtype="|S6")
	for i in range(0, num_reviews):

		if result[i] != label_test[i]:
			error += 1
	print('error: ')
	print(error)

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
		label_test = np.asarray(test_data['sentiment'], dtype="|S6")
		for i in range(0, num_reviews):
			if result[i] != label_test[i]:
				error += 1
		print('error: ')
		print(error)

	print ("---------------")
def mix_learn(train_data, train_data_features, test_data, vectorizer, vocab):
	pass

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

