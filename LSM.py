import pyndri
import pyndri.compat

import gensim
from gensim import corpora
from gensim import models
from gensim.models import lsimodel
from gensim.models import ldamodel
from gensim.models import tfidfmodel
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity, Similarity

import os

from raven import Client
client = Client('https://8c8609001d6745419fc953a175fded42:04c8cf7bfb9f46ccac62607f38c6d77a@sentry.io/278884')

# Hyperparameter to optimize
NUM_TOPICS = 50

index = pyndri.Index('index/')
dictionary = pyndri.extract_dictionary(index)
docs = pyndri.compat.IndriSentences(index, dictionary)

# 'Connector' class.
class CorpusConnector:
    def __init__(self):
        self.index = index
        self.dictionary = corpora.Dictionary(docs)
        self.dictionary.filter_extremes(no_below=5)

    def __iter__(self):
        for doc in docs:
            yield self.dictionary.doc2bow(doc)

    def save_dict(self):
        self.dictionary.save('dictionary.dict')

if not os.path.isfile('corpus.mm'):
    mm = CorpusConnector()
    corpora.MmCorpus.serialize('corpus.mm', mm)
    mm.save_dict()
else:
    dictionary = corpora.Dictionary.load('dictionary.dict')
    mm = corpora.MmCorpus('corpus.mm')

if not os.path.isfile('lsi.model'):
    # LSI model
    lsi = lsimodel.LsiModel(corpus = mm, id2word = mm.dictionary, num_topics = NUM_TOPICS)
    lsi.save('lsi.model')

    lsi_corpora = lsi[mm]
    corpora.MmCorpus.serialize('lsi_corpora.mm', lsi_corpora)
else:
    lsi_corpora = gensim.corpora.MmCorpus('lsi_corpora.mm')
    lsi = gensim.models.LsiModel.load('lsi.model')


if not os.path.isfile('lda.model'):
    # LDA model
    lda = ldamodel.LdaModel(corpus = mm, num_topics = NUM_TOPICS, id2word = mm.dictionary)
    lda.save('lda.model')

    lda_corpora = lda[mm]
    corpora.MmCorpus.serialize('lda_corpora.mm', lda_corpora)
else:
    lda_corpora = gensim.corpora.MmCorpus('lda_corpora.mm')
    lda = gensim.models.LdaModel.load('lda.model')


similarity_index_file = 'sim_index.index'

if not os.path.isfile('sim_index.index'):
    similarity_index = Similarity('./sim_index', lsi_corpora, NUM_TOPICS)
    similarity_index.save('sim_index.index')
else:
    similarity_index = Similarity.load('sim_index.index')
