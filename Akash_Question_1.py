from sklearn.linear_model import RandomizedLasso
import argparse
import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import (RandomizedLasso, lasso_stability_path, LassoLarsCV, LassoCV, ElasticNetCV, LassoLars)
import pandas as pd 
import re
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
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
from sklearn.decomposition import PCA


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
	print X_dummytrain.shape
	print X_dummytest.shape
	print y_dummytrain.shape
	print y_dummytest.shape

	#lsvc = ensemble.RandomForestRegressor(max_depth=10, n_estimators=230, random_state = 0).fit(X_dummytrain,y_dummytrain)	

	# pca = PCA(n_components= 100)
	# seeF = pca.fit(X_dummytrain,y_dummytrain).transform(X_dummytrain)
	# seeFT = pca.fit(X_dummytrain,y_dummytrain).transform(X_dummytest)
	# print pca.explained_variance_ratio_
	# print seeF.shape

	# clf = ensemble.GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 6 , random_state = 0, loss = 'ls')
	# estimate = clf.fit(seeF,y_dummytrain)	

	# score = clf.score(seeF,y_dummytrain)
	# print score
	

	# predictions = estimate.predict(seeFT)

	# print mean_squared_error(y_dummytest, predictions)

	# print "NEXT"
	
	# clf = ensemble.GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 6 , random_state = 0, loss = 'ls')
	# estimate = clf.fit(X_dummytrain,y_dummytrain)	

	# score = clf.score(X_dummytrain,y_dummytrain)
	# print score
	

	# predictions = estimate.predict(X_dummytest)

	# print mean_squared_error(y_dummytest, predictions)

	# print "NEXT"
	# print A.shape

	
	lsvc = ensemble.ExtraTreesRegressor(n_estimators = 200, random_state = 0, max_depth = 12).fit(X_dummytrain,y_dummytrain)	
	print lsvc
	see = lsvc.feature_importances_
	hey = sorted(see, reverse=True)

	print hey[0:30]

	#lsvc = LassoCV(max_iter=100000, cv = 6)
	#lsvc = Lasso(alpha=0.1)

	# lars_cv = LassoLars(alpha=0.0005, max_iter=100000).fit(X_dummytrain,y_dummytrain)	
	# alphas = np.linspace(1*lars_cv.alphas_[0], 0.1* lars_cv.alphas_[0], 1000)
	# model = RandomizedLasso(alpha=alphas, random_state=42).fit(X_dummytrain,y_dummytrain)	
	model = SelectFromModel(lsvc, prefit = True, threshold = 5e-3)
	print model


	#model =RFECV(lsvc, step=1, cv=5).fit(X_dummytrain,y_dummytrain)
	

	Train_new = model.transform(X_dummytrain)
	print Train_new.shape

	print len(X_dummytrain)

	print len(X_dummytest)
	
	newindices = model.get_support(True)
	

	FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(len(X_dummytrain)), newindices)]
	FinalTestLessFeature = X_dummytest[np.ix_(np.arange(len(X_dummytest)), newindices)]

	print FinalTrainLessFeature.shape
	print FinalTestLessFeature.shape

	print "YOO"


	rng = np.random.RandomState(1)

	clf = ensemble.GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 6 , random_state = 0, loss = 'ls')
	estimate = clf.fit(FinalTrainLessFeature,y_dummytrain)

	score = clf.score(FinalTrainLessFeature,y_dummytrain)
	print score
	

	predictions = estimate.predict(FinalTestLessFeature)

	print mean_squared_error(y_dummytest, predictions)

	print "NEXT"

	clf = ensemble.RandomForestRegressor(max_depth=12, n_estimators=300, random_state = 0)
	estimate = clf.fit(FinalTrainLessFeature,y_dummytrain)
	

	predictions = estimate.predict(FinalTestLessFeature)

	print mean_squared_error(y_dummytest, predictions)

	print A.shape

	print "In writePredictions"
	o = DictWriter(open("Predictions.csv", 'w'),["target"])
	for y_val in predictions:
		o.writerow({'target': y_val})

	End()
	



