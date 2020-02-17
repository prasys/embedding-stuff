import io
import pandas as pd
import flair
import torch
from flair.embeddings import WordEmbeddings
from flair.embeddings import CharacterEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import FlairEmbeddings
from tqdm import tqdm ## tracks progress of loop ##



def convertPandasTextoList(dataFrame):
    text = dataframe.tolist()
    return texts


def generateEmbeddings(document_embeddings,texts,testSentence):
    embedSize = testSentence.embedding.size()[1] 
    tensorSize = torch.zeros(0,embedSize)
    for text in texts:
        sentence = Sentence(text)
        document_embeddings.embed(sentence)
        
        # Add it as a tensor
        



