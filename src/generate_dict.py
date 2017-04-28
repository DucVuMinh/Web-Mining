from sklearn.feature_extraction.text import CountVectorizer
import  numpy as np
import  pandas as pd

def create_dictionary(data, dictfile):
    """
    function
    :param data:
    :return:
    """
    print("Creating the bag of words...\n")
    vectorizer = CountVectorizer(analyzer="word", \
								 tokenizer=None, \
								 preprocessor=None, \
								 stop_words=None, \
								 max_features=5000)
    # fit_transform() does two functions: First, it fits the model
    #  and learns the vocabulary; second, it transforms our training data
    #  into feature vectors. The input to fit_transform should be a list of
    #  strings.
    train_data_features = vectorizer.fit_transform(data)
    # Numpy arrays are easy to work with, so convert the result to an
    #  array
    train_data_features = train_data_features.toarray()
    vocab = vectorizer.get_feature_names()

	# write vocabulary to outer file
    dist = np.sum(train_data_features, axis=0)
    vocab_out = pd.DataFrame(data={"vocal": vocab, "count": dist})
    vocab_out.to_csv(dictfile, index=False, quoting=3 )
    return vectorizer, train_data_features, vocab
