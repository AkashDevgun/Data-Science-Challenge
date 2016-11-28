from sklearn.linear_model import RandomizedLasso
import argparse
import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
import pandas as pd 
import re
from sklearn import preprocessing
from csv import DictReader, DictWriter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RandomizedLasso
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import sklearn
from sklearn.cross_validation import train_test_split
import csv
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

	def __init__(self):
		"""Impute missing values.

		Columns of dtype object are imputed with the most frequent value 
		in column.

		Columns of other types are imputed with mean of column.

		"""
	def fit(self, X, y=None):

		self.fill = pd.Series([X[c].mean()
			if X[c].dtype == np.dtype('float64') else X[c].value_counts().index[0] for c in X],
			index=X.columns)

		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)

	def fit_transform(self,X,y=None):
		return self.fit(X,y).transform(X)




def End ():
	print "Question_1 Finish"



if __name__ == "__main__":
	print('The scikit-learn version is {}.'.format(sklearn.__version__))
	parser = argparse.ArgumentParser(description='Question_1')

	args = parser.parse_args()

	train = list(csv.reader(open("codetest_train.txt", 'r'), delimiter='\t'))

	print len(train)

	test = list(csv.reader(open("codetest_test.txt", 'r'), delimiter='\t'))

	laEn = preprocessing.LabelEncoder()

	lengthTrain = len(train) - 1
	lengthTest = len(test) - 1



	intialtrain_df  = pd.DataFrame(train)

	intialtrain_df.columns = intialtrain_df.values[0]

	features = intialtrain_df.columns


	train_df = intialtrain_df[1:].reset_index(drop=True)
	
	Encoder_vocab = ['f_61','f_121','f_215','f_237']

	

	train_df = train_df.convert_objects(convert_numeric=True)


	finaltrain_df = DataFrameImputer().fit_transform(train_df)


	for each in Encoder_vocab:
		count = 0
		fillitem = finaltrain_df[each].value_counts().index[0]
		for eachvalue in finaltrain_df[each]:
			if eachvalue is "":
				finaltrain_df[each][count] = fillitem
			count = count + 1 

	

	count = 1
	EncodeTrainData = []

	for each in Encoder_vocab:
		if count == 1:
			EncodeTrainData = finaltrain_df[each]
			count = 2
		else:
			EncodeTrainData = np.column_stack((EncodeTrainData,finaltrain_df[each]))



	
	intialtest_df  = pd.DataFrame(test)

	intialtest_df.columns = intialtest_df.values[0]

	test_df = intialtest_df[1:].reset_index(drop=True)

	test_df = test_df.convert_objects(convert_numeric=True)

	
	finaltest_df = DataFrameImputer().fit_transform(test_df)


	for each in Encoder_vocab:
		count = 0
		fillitem = finaltest_df[each].value_counts().index[0]
		for eachvalue in finaltest_df[each]:
			if eachvalue is "":
				finaltest_df[each][count] = fillitem
			count = count + 1 


	count = 1
	EncodeTestData = []

	for each in Encoder_vocab:
		if count == 1:
			EncodeTestData = finaltest_df[each]
			count = 2
		else:
			EncodeTestData = np.column_stack((EncodeTestData, finaltest_df[each]))


	print EncodeTestData.shape
	print EncodeTrainData.shape

	train_X = np.reshape(EncodeTrainData,(lengthTrain*4)) 
	test_X = np.reshape(EncodeTestData, (lengthTest*4))

	trainLabelEncode = laEn.fit_transform(train_X)
	testLabelEncode = laEn.transform(test_X)


	Encodedtrain_X = np.reshape(trainLabelEncode, (lengthTrain,4)) 
	Encodedtest_X = np.reshape(testLabelEncode, (lengthTest,4))

	print len(EncodeTestData)
	print list(laEn.classes_)


	Values_vocab = ['target','f_61','f_121','f_215','f_237']


	count = 1
	TrainValueData = []

	for eachfeature in features:
		if eachfeature not in Values_vocab:
			if count == 1:
				TrainValueData = finaltrain_df[eachfeature]
				count = 2
			else:
				TrainValueData = np.column_stack((TrainValueData, finaltrain_df[eachfeature]))


	count = 1
	TestValueData = []

	for eachfeature in features:
		if eachfeature not in Values_vocab:
			if count == 1:
				TestValueData = finaltest_df[eachfeature]
				count = 2
			else:
				TestValueData = np.column_stack((TestValueData, finaltest_df[eachfeature]))
				


	Final_Train_X = np.column_stack((TrainValueData, Encodedtrain_X))
	Final_Test_X = np.column_stack((TestValueData, Encodedtest_X))

	print Final_Train_X.shape
	print Final_Test_X.shape



	TargetValues = finaltrain_df["target"]



	rng = np.random.RandomState(1)


	X_dummytrain, X_dummytest, y_dummytrain, y_dummytest = train_test_split(Final_Train_X, TargetValues, test_size=0.2, random_state=42)

	
	lsvc = ensemble.ExtraTreesRegressor(n_estimators = 50, random_state = 0).fit(Final_Train_X,TargetValues)	
	model = SelectFromModel(lsvc, prefit = True)
	Train_new = model.transform(Final_Train_X)
	print Train_new.shape
	
	newindices = model.get_support(True)
	

	FinalTrainLessFeature = Final_Train_X[np.ix_(np.arange(len(train) - 1), newindices)]
	FinalTestLessFeature = Final_Test_X[np.ix_(np.arange(len(test)- 1), newindices)]

	print FinalTrainLessFeature.shape
	print FinalTestLessFeature.shape




	rng = np.random.RandomState(1)

	clf = ensemble.GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 6 , random_state = 0, loss = 'ls')
	estimate = clf.fit(FinalTrainLessFeature,TargetValues)
	

	predictions = estimate.predict(FinalTestLessFeature)
	print "In writePredictions"
	o = DictWriter(open("Predictions.csv", 'w'),["target"])
	for y_val in predictions:
		o.writerow({'target': y_val})

	End()
	



