import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np
from matplotlib import axis
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from Word2Vec_Avg import getAvgFeatureVecs, getAvgFeatureVecsVs2
from gensim.models import Word2Vec
#read data from file and generate train_data, test_data
def read_data():
	"""

	:return:
	"""
	print ("read data ....\n")
	data_pos = pd.read_csv('../data/smooth_pos.csv', header=0, \
						   delimiter="\t", quoting=3)
	data_pos['sentiment'] = 1
	data_neg = pd.read_csv('../data/smooth_neg.csv', header=0, \
						   delimiter="\t", quoting=3)
	data_neg['sentiment'] = -1
	data_neu = pd.read_csv('../data/smooth_neu.csv', header=0, \
						   delimiter="\t", quoting=3)
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

def create_bag_of_words(data, model):
	print("Creating the bag of words...\n")
	vectorizer = CountVectorizer(analyzer="word", \
								 tokenizer=None, \
								 preprocessor=None, \
								 stop_words=None, \
								 max_features=5000)
	vectorizer2 = CountVectorizer(analyzer="word",
								  tokenizer=None,
								  preprocessor=None,
								  stop_words=None,
								  max_features=5000,
								  ngram_range=(2,3)
								  )
	print ("get add feature ...")
	#add_feature = getAvgFeatureVecsVs2(data, model,2000, 400)
	#print (add_feature.shape)

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of
	# strings.
	train_data_features = vectorizer.fit_transform(data)

	# Numpy arrays are easy to work with, so convert the result to an
	# array
	train_data_features = train_data_features.toarray()
	#train_data_features = np.append(train_data_features, add_feature, axis= 1)
	print ("total feature")
	print (train_data_features.shape)
	vocab = vectorizer.get_feature_names()

	# write vocabulary to outer file
	dist = np.sum(train_data_features, axis=0)
	#vocab_out = pd.DataFrame(data={"vocal": vocab, "count": dist})
	return vectorizer, train_data_features, vocab


def random_forest(train_data,train_data_features, test_data, vectorizer, model):
	"""

	:param batch:
	:return:
	"""
	print("Training the random forest...")

	# Numpy arrays are easy to work with, so convert the result to an
	# array
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
	#add_test_feature = getAvgFeatureVecsVs2(test, model,2000, 400)
	#test_data_features = np.append(test_data_features, add_test_feature, axis= 1)

	# Use the random forest to make sentiment label predictions
	result = forest.predict(test_data_features)

	num_reviews = len(test)
	error = 0
	label_test = np.asarray(test_data['sentiment'], dtype="|S6")
	for i in range(0, num_reviews):

		if result[i] != label_test[i]:
			error += 1
	print('error: RDF')
	print(error)
	return result

def svm_learn(train_data, train_data_features, test_data, vectorizer, model):
	from sklearn import svm
	from sklearn.svm import NuSVC

	test = test_data['content']
	label = np.asarray(train_data["sentiment"], dtype="|S6")
	num_reviews = len(test)
	test_data_features = vectorizer.transform(test)
	test_data_features = test_data_features.toarray()
	#add_test_feature = getAvgFeatureVecsVs2(test, model,2000, 400)
	#test_data_features = np.append(test_data_features, add_test_feature, axis = 1)

	print ("---------------")
	clf = svm.NuSVC(nu=0.6, kernel='linear', decision_function_shape='ovr')
	clf.fit(train_data_features, label)
	result = clf.predict(test_data_features)
	error = 0
	label_test = np.asarray(test_data['sentiment'], dtype="|S6")
	for i in range(0, num_reviews):
		if result[i] != label_test[i]:
			error += 1
	print('error: SVM')
	print(error)
	return result
def mix_result(result1, result2, result3):
	num_reviews = len(result1)
	print ("lenght result 1")

	result = np.zeros(shape=(num_reviews))
	print (len(result))
	for i in range(0, num_reviews):
		if result1[i] == result2[i]:
			result[i] = result1[i]
		elif result1[i] == result3[i]:
			result[i] = result1[i]
		elif result2[i] == result3[i]:
			result[i] = result2[i]
		else:
			result[i] = 0
	return result
def mix_model(train_data, train_data_features, test_data, vectorizer, model):
	result1 = random_forest(train_data, train_data_features, test_data, vectorizer, model)

	result2 = random_forest(train_data, train_data_features, test_data, vectorizer, model)
	result3 = svm_learn(train_data, train_data_features, test_data, vectorizer, model)
	result = mix_result(result1, result2, result3)
	num_reviews = len(result)
	label_test = np.asarray(test_data['sentiment'], dtype="|S6")
	error = 0
	print (num_reviews)
	for i in range(0, num_reviews):
		if int(result[i]) != int(label_test[i]):
			error += 1


	print ("mix model error: ")
	print (error)


if __name__ == '__main__':
	model = Word2Vec.load("../300features_40minwords_10context")
	#step 1: read data then get batch of data
	model_choice = "mix"
	batch = read_data()
	all_data = []
	all_data = batch[0][0]
	all_data = np.append(all_data, batch[0][1], axis=0)
	all_data = np.append(all_data, batch[0][2], axis=0)
	all_data = np.append(all_data, batch[0][3], axis=0)
	all_data = np.append(all_data, batch[0][4], axis=0)

	all_data = np.append(all_data, batch[1][1], axis=0)
	all_data = np.append(all_data, batch[1][2], axis=0)
	all_data = np.append(all_data, batch[1][3], axis=0)
	all_data = np.append(all_data, batch[1][4], axis=0)
	all_data = np.append(all_data, batch[1][0], axis=0)

	all_data = np.append(all_data, batch[2][1], axis=0)
	all_data = np.append(all_data, batch[2][2], axis=0)
	all_data = np.append(all_data, batch[2][3], axis=0)
	all_data = np.append(all_data, batch[2][4], axis=0)
	all_data = np.append(all_data, batch[2][0], axis=0)
	all_data = pd.DataFrame(data = all_data, columns= ['content', 'sentiment'])
	vectorizer, all_data_feature, vocab = create_bag_of_words(all_data['content'], model)
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

		train_data_features = vectorizer.transform(train_data["content"])
		train_data_features = train_data_features.toarray()
		#add_test_feature = getAvgFeatureVecsVs2(train_data["content"], model, 2000, 400)
		#train_data_features = np.append(train_data_features, add_test_feature, axis=1)
		print( train_data_features.shape )
		if model_choice == "random_forest":
			random_forest(train_data,train_data_features, test_data, vectorizer, model)
		elif model_choice =="svm":
			svm_learn(train_data, train_data_features, test_data, vectorizer, model)
		else:
			mix_model(train_data, train_data_features, test_data, vectorizer, model)

