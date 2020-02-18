import io
import pandas as pd
import flair
import torch
import numpy as np
import sys
from flair.embeddings import WordEmbeddings
from flair.embeddings import CharacterEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import TransformerXLEmbeddings
from flair.embeddings import OpenAIGPTEmbeddings
from flair.embeddings import OpenAIGPT2Embeddings
from flair.embeddings import DocumentPoolEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm ## tracks progress of loop ##



# def convertPandasTextoList(dataFrame):
#     text = dataframe.tolist()
#     return texts


def generateEmbeddings(document_embeddings,texts):
	listOfArray = []

	for text in texts:
		sentence = Sentence(text)
		document_embeddings.embed(sentence)
		documentEmbed = sentence.get_embedding()
		documentEmbed = documentEmbed.cpu().detach().numpy()
		listOfArray.append(documentEmbed)
		
	
	output = np.vstack(listOfArray)
	print(output.shape)

	return -1

def read_csv(filepath):
	df_chunk = pd.read_csv(filepath, sep=',', header=0,encoding = "utf-8")
	#df_chuck = df_chuck.fillna(0)
	return df_chunk

def pickModel(text):
	model = {
		'elmo': ELMoEmbeddings(),
		'gpt' : OpenAIGPTEmbeddings(),
		'gpt2' : OpenAIGPT2Embeddings(),
		'transformer' : TransformerXLEmbeddings()
	}
	return model.get(text,-1)

def getDocumentModel(text):
	# embeddingModel = ELMoEmbeddings()
	embeddingModel = pickModel(text)
	DocumentEmbedding = DocumentPoolEmbeddings([embeddingModel])
	return DocumentEmbedding


if __name__ == '__main__':
	print ('Number of arguments:', len(sys.argv), 'arguments.')
	print ('Argument List:', str(sys.argv))
	if len(sys.argv[1]) > 1:  # File to be open/to process
		df = read_csv(sys.argv[1])
		CommentList = df['Comment'].tolist() # pick the item/column that we want to do BERT embeddings
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit
	if len(sys.argv[2]) > 1: # pick the embedding model
		embedding = sys.argv[2]
		model = getDocumentModel(embedding)
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit
	
	# if len(sys.argv[3]) > 1:
	# 	model = sys.argv[3]
	# else:
	# 	print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
	# 	exit() #force exit

	if len(sys.argv[3]) > 1: #output name
		outPutFileName = sys.argv[3]
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit		
	
	# CommentList = df['Comment'].tolist() # pick the item/column that we want to do BERT embeddings
	print("Start Embedding Service Client")
	# randomSentence = CommentList[1] #takes a random sentence , well not really :P
	doclength = generateEmbeddings(model,CommentList)
	print(doclength.shape)