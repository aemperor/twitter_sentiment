import pandas
import numpy
import csv
import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.probability import FreqDist
from collections import defaultdict
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs
# from numpy import loadtxt, where
# from pylab import scatter, show, legend, xlabel, ylabel
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression


pos_tweets = []
neg_tweets = []
tweet_features = []
trainingData = pandas.read_csv('data/trainingdata100000.csv')
# trainingData = pandas.read_csv('data/trainingdataSentiment_short.csv')
testData = pandas.read_csv('data/testDataSentiment.csv')
labels = []
tweets = []
test_tweets = []

def populateLists():
	global pos_tweets
	global neg_tweets
	for i in trainingData.iterrows():
		sentiment = i[1][0]
		tweet = i[1][5]
		tweet = tweet.lower() #lower case for uniform dictionary

		# create lists based on sentiment
		if (sentiment == 0):
			pos_tweets.append((tweet[1:], 'positive'))
		else:
			neg_tweets.append((tweet[1:], 'negative'))


def createWordList():
	global labels
	global tweets
	for (words, sentiment) in pos_tweets + neg_tweets:
		prune = [i.lower() for i in words.split() if len(i) >= 3]
		newWords = pruneWords(prune)
		tweets.append((newWords, sentiment))

def pruneWords(words):
	newWords = []
	for word in words:
		if ('@' in word):
			newWords.append(word.replace("@", ""))
		elif ('#' in word):
			newWords.append(word.replace("#", ""))
		elif ('.' in word):
			newWords.append(word.replace(".", ""))
		else:
			newWords.append(word)

	return newWords

def getTweetWords(tweets):
	words = []
	for (words, sentiment) in tweets:
		words.extend(words)
	return words


def getTweetFeatures(wordlist):
	global tweet_features
	tweetlist = nltk.probability.FreqDist(wordlist)
	tweet_features = tweetlist.keys()
	return tweet_features

def extractFeatures(document):
	docWords = set(document);
	features = {}
	for tweetWord in tweet_features:
		features['contains(%s)' % tweetWord] = (tweetWord in docWords)
	return features

def LRSetup():
	X = []
	Y = []
	for i in trainingData.iterrows():
		sentiment = i[1][0]
		tweet = i[1][5]
		tweet = tweet.lower() #lower case for uniform dictionary

		# create lists based on sentiment
		if (sentiment == 0):
			X.append(tweet[1:])
			Y.append(0);
		else:
			X.append(tweet[1:])
			Y.append(1);
	pos = where(Y == 0)
	neg = where(Y == 1)
	scatter(X[pos, 0], X[pos, 1], marker = "o", c="b")
	scatter(X[neg, 0], X[neg, 1], marker = "x", c="r")
	xlabel("blah")
	ylabel("blah blah")
	legend(["Positive", "Negative"])
	show()

# Naive Bayes training
def train(labeled_featuresets, estimator=ELEProbDist):
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()

        for featureset, label in labeled_featuresets:
            label_freqdist.inc(label)
            for fname, fval in featureset.items():
                feature_freqdist[label, fname].inc(fval)
                feature_values[fname].add(fval)
                fnames.add(fname)

        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                feature_freqdist[label, fname].inc(None, num_samples-count)
                feature_values[fname].add(None)

        label_probdist = estimator(label_freqdist)

        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label,fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)

def prepareBayesTestData():
	test_neg_tweets = []
	test_pos_tweets = []
	for i in testData.iterrows():
		sentiment = i[1][0]
		tweet = i[1][5]
		tweet = tweet.lower() #lower case for uniform dictionary

		# create lists based on sentiment
		if (sentiment == 0):
			test_pos_tweets.append((tweet[1:], 'positive'))
		else:
			test_neg_tweets.append((tweet[1:], 'negative'))

	for (words, sentiment) in test_pos_tweets + test_neg_tweets:
		prune = [i.lower() for i in words.split() if len(i) >= 3]
		newWords = pruneWords(prune)
		test_tweets.append((newWords, sentiment))



def main():
	populateLists()
	createWordList()
	wordFeatures = getTweetFeatures(getTweetWords(tweets))
	
	# Naive Bayes Classifier
	# features = extractFeatures(wordFeatures)
	trainingSet = nltk.classify.apply_features(extractFeatures, tweets)
	# print trainingSet
	classifier = train(trainingSet)
	# print classifier.show_most_informative_features(32)

	# run test data for Naive Bayes
	prepareBayesTestData()
	f = open('output.csv', 'w')
	c = csv.writer(f)
	for tweet, sentiment in test_tweets:
		features = extractFeatures(tweet)
		res = classifier.classify(extractFeatures(tweet))
		data = [[features, res]]
		c.writerows(data)
	f.close()

	# Logistic Regression
	# LRSetup()



main()
