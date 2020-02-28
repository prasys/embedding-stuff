import autosklearn.classification
import sklearn.metrics
import pandas as pd
import scipy
import numpy as np
import sys
import pickle
from sklearn.model_selection import train_test_split


def loadTheFiles(trainX,trainY,testX,testY):
	trainX = np.load(trainX,allow_pickle=True) # load train file
	trainY = np.load(trainY,allow_pickle=True)
	testX = np.load(testX,allow_pickle=True)
	testY = np.load(testY,allow_pickle=True)
	return (trainX,trainY,testX,testY)


def doThePrediction(testX,trainX,trainY,fileName,seed=42):
	automl = autosklearn.classification.AutoSklearnClassifier(seed=seed)
	automl.fit(trainX,trainY,metric=autosklearn.metrics.balanced_accuracy)
	predictedValue = automl.predict(testX)
	with open('out_models.txt', 'a+') as f:
		print(fileName,file=f)
		print(automl.show_models(),file=f)
		print('END',file=f)

	return predictedValue

def matrixScoreCalculations(predictedX,actualX,fileName,seed=42):
	with open('out.txt', 'a+') as f:
		tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=actualX, y_pred=predictedX).ravel()
		print(fileName,file=f)
		print(tp,fp,fn,tn,file=f)
		print("Seed Value is",seed,file=f)
		print("Accuracy score", sklearn.metrics.balanced_accuracy_score(predictedX, actualX),file=f)
		print("F1 score", sklearn.metrics.f1_score(predictedX, actualX),file=f)
		# print("Recall score", sklearn.metrics.recall_score(predictedX, actualX),file=f)
		# print("Percision score", sklearn.metrics.precision_score(predictedX, actualX),file=f)
		# pos = (tp/(tp+fn))
		# neg = (tn/(tn+fp))
		# balAcc = (pos + neg)/2
		# print("Balanced Accuracy score", sklearn.metrics.accuracy_score(predictedX, actualX),file=f)
		# print("Positive Score",(tp/(tp+fn)),file=f)
		# print("Negative Score",(tn/(tn+fp)),file=f)
		print("END",file=f)


def joinTestTrain(testx,trainx):
	return np.vstack((trainx,testx))

def joinTestTrainTruth(testx,trainx):
	return np.hstack((trainx,testx))

def test_train_split(embeddings,truth,seed=42):
	trainX, testX, trainY, testX = train_test_split(embeddings,truth,random_state=seed)
	return(trainX,trainY,testX,testY)
	
if __name__ == '__main__':
	file = open('randSeed.pkl', 'rb') # load our random seed generator
	data = pickle.load(file)
	print ('Number of arguments:', len(sys.argv), 'arguments.')
	print ('Argument List:', str(sys.argv))
	if len(sys.argv[1]) > 1:
		trainX = sys.argv[1]
		fileName = sys.argv[1]
		trainY = sys.argv[2]
		testX = sys.argv[3]
		testY = sys.argv[4]
		for seed in data:
			print(seed)
			trainX , trainY , testX , testY = loadTheFiles(trainX,trainY,testX,testY)
			concatEmbeddings = joinTestTrain(trainX,testX) # concat the embeddings
			concatTruth = joinTestTrainTruth(trainY,testY)
			trainX , trainY , testX , testY = test_train_split(concatEmbeddings,concatTruth,seed=seed)
			predictedX = doThePrediction(testX,trainX,trainY,fileName,seed=seed)
			matrixScoreCalculations(predictedX,testY,fileName,seed=seed)
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit
