import pyndri
import pyndri.compat

import gensim
from gensim import corpora
from gensim import models
from gensim.models import lsimodel
from gensim.models import ldamodel

import os

# Hyperparameter to optimize
NUM_TOPICS = 50

index = pyndri.Index('index/')
dictionary = pyndri.extract_dictionary(index)

# 'Connector' class.
class CorpusConnector(object):
    def __init__(self, index):
        self.index = index
        self.docs = pyndri.compat.IndriSentences(index, dictionary)
        self.dictionary = corpora.Dictionary(self.docs)
        self.dictionary.filter_extremes(no_below=5)

    def __iter__(self):
        for doc in self.docs:
            yield self.dictionary.doc2bow(doc)

if not os.path.isfile('corpus.mm'):
    mm = CorpusConnector(index)
    corpora.MmCorpus.serialize('corpus.mm', mm)
else:
    mm = corpora.MmCorpus('corpus.mm')

# LSI model
lsi = lsimodel.LsiModel(corpus = mm, id2word = dictionary, num_topics = NUM_TOPICS)
lsi.save('lsi.model')

# LDA model
lda = ldamodel.LdaModel(corpus = mm, num_topics = NUM_TOPICS, id2word=dictionary)
lda.save('lda.model')
