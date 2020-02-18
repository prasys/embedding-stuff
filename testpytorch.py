from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
import numpy as np

# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding])


sentence = Sentence('lorem ipsum.')

document_embeddings.embed(sentence)

x = sentence.get_embedding()

y = x.cpu().detach().numpy()

print(y.shape)

sentence2 = Sentence('lorem ipsum.')

document_embeddings.embed(sentence2)

a = sentence2.get_embedding()

b = a.cpu().detach().numpy()

c = np.concatenate((y, b), axis=0)
d = np.vstack((y,b))
print(d.shape)
