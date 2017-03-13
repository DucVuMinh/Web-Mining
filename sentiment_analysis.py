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
data_pos = pd.read_csv('smooth_pos.csv',header=0, \
                    delimiter="\t", quoting=3)
data_pos['sentiment'] = 1
data_neg = pd.read_csv('smooth_neg.csv',header=0, \
                    delimiter="\t", quoting=3)
data_neg['sentiment'] = -1
data_neu = pd.read_csv('smooth_neu.csv',header=0, \
                    delimiter="\t", quoting=3)
data_neu['sentiment'] = 0

train_data = data_pos.iloc[1:1201]
train_data = train_data.append(data_neg.iloc[1:1201])
train_data = train_data.append(data_neu.iloc[1:1201])
test_data = data_pos.iloc[1200:]
test_data = test_data.append(data_neg.iloc[1200:])
test_data= test_data.append(data_neu.iloc[1200:])
all_data = data_pos.append(data_neg)
all_data = all_data.append(data_neu)

print ( type(train_data) )
print (test_data.shape)
#Step 1 - cleaning data function
"""
- remove non-letters include: delimiter, number, . . .
- convert to lower case
"""
def preprocess(raw_comment): 
	letter = re.sub("[.,'\\{\\}\\@\\!\\#\\$\\%\\&\\*\\(\\)\\|\\[\\]\\?\\<\\>\\:\\;\\`\=\\+\\-]", " ", raw_comment)
	letter = re.sub('["]', " ", letter)
	letter = letter.lower()
	return letter
def smooth_data():
	smooth_data_pro = []
	for d in data_neu['content']:
		foo = preprocess(d)
		smooth_data_pro.append(foo)
	smooth_neu  = pd.DataFrame(data = {"content":smooth_data_pro})
	smooth_neu.to_csv("smooth_neu.csv", index=False, quoting=3 )
#%

print ("Creating the bag of words...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(train_data['content'])

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()


#write vocabulary to outer file
dist = np.sum(train_data_features, axis=0)
vocab_out = pd.DataFrame(data = {"vocal":vocab , "count" : dist})
#vocab_out = vocab_out.to_csv("vocab.csv", index=False, quoting=3 )

# Read the test data
test = test_data['content']

# Verify that there are 25,000 rows and 2 columns
print (test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test)
clean_test_reviews = [] 


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(test)
test_data_features = test_data_features.toarray()
#%
def random_forest():
	print ("Training the random forest...")
	from sklearn.ensemble import RandomForestClassifier

	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 100)
	# Fit the forest to the training set, using the bag of words as 
	# features and the sentiment labels as the response variable
	#
	# This may take a few minutes to run
	forest = forest.fit( train_data_features, train_data["sentiment"] )


	# Use the random forest to make sentiment label predictions
	result = forest.predict(test_data_features)
	

from sklearn import svm
from sklearn.svm import NuSVC
for nu in [0.53 , 0.56 , 0.6 , 0.63 , 0.66 , 0.7]:
	clf = svm.NuSVC(nu=nu,kernel='linear',decision_function_shape='ovr')
	clf.fit(train_data_features, train_data["sentiment"])  
	result=clf.predict(test_data_features)
	error = 0
	for i in range(0,num_reviews):
		if result[i] != test_data['sentiment'].iloc[i]:
			error +=1
	print ('error: ')
	print (error)
	# Copy the results to a pandas dataframe with an "id" column and
	# a "sentiment" column
	"""
	output = pd.DataFrame( data={"sentiment":result} )
	file = "Bag_of_Words_model_svm."+ str(nu) +"csv"
	# Use pandas to write the comma-separated output file
	output.to_csv( file, index=False, quoting=3 )
	"""
#%