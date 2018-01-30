# Imports
import pyndri
import pyndri.compat

import gensim
from gensim import corpora
from gensim import models
from gensim.models import lsimodel
from gensim.models import ldamodel
from gensim.similarities import Similarity

import os

if not os.path.exists('./LSM/'):
    os.makedirs('./LSM/')

# File Locations
dict_file = './LSM/dictionary.dict'
mm_corpus_file = './LSM/corpus.mm'
lsi_model_file = './LSM/lsi.model'
lsi_corpora_file = './LSM/lsi_corpora.mm'
lsi_sim_file = './LSM/lsi_sim.index'
lda_model_file = './LSM/lda.model'
lda_corpora_file = './LSM/lda_corpora.mm'
lda_sim_file = './LSM/lda_sim.index'

# Hyperparameters
LSI_TOPICS = 200
LDA_TOPICS = 100

class CorpusConnector:

    def __init__(self, index):
        self.index = index
        dictionary = pyndri.extract_dictionary(self.index)
        self.docs = pyndri.compat.IndriSentences(self.index, dictionary)
        self.dictionary = corpora.Dictionary(self.docs)
        self.dictionary.filter_extremes(no_below=5)

    def __iter__(self):
        for doc in self.docs:
            yield self.dictionary.doc2bow(doc)

    def save_dict(self):
        self.dictionary.save(dict_file)

class LSM:

    def __init__(self, model_name, index):
        self.model_name = model_name

        if self.model_name == 'LSI':
            self.model_file = lsi_model_file
            self.corpora_file = lsi_corpora_file
            self.similarity_file = lsi_sim_file
            self.num_topics = LSI_TOPICS
        elif self.model_name == 'LDA':
            self.model_file = lda_model_file
            self.corpora_file = lda_corpora_file
            self.similarity_file = lda_sim_file
            self.num_topics = LDA_TOPICS

        if not os.path.isfile(mm_corpus_file) or not os.path.isfile(dict_file):
            self.corpus = CorpusConnector(index)
            corpora.MmCorpus.serialize(mm_corpus_file, self.corpus)
            self.corpus.save_dict()
            self.dictionary = self.corpus.dictionary
        else:
            self.dictionary = corpora.Dictionary.load(dict_file)
            self.corpus = corpora.MmCorpus(mm_corpus_file)

        self.model = None
        self.corpora = None
        self.similarity_index = None

    def create_model(self):
        if not os.path.isfile(self.model_file):
            if self.model_name == 'LSI':
                self.model = lsimodel.LsiModel(corpus = self.corpus, \
                        id2word = self.dictionary, num_topics = self.num_topics)
            else:
                self.model = ldamodel.LdaModel(corpus = self.corpus, \
                        num_topics = self.num_topics, id2word = self.dictionary)
            self.model.save(self.model_file)

            self.corpora = self.model[self.corpus]
            corpora.MmCorpus.serialize(self.corpora_file, self.corpora)
        else:
            self.corpora = gensim.corpora.MmCorpus(self.corpora_file)
            if self.model_name == 'LSI':
                self.model = gensim.models.LsiModel.load(self.model_file)
            else:
                self.model = gensim.models.LdaModel.load(self.model_file)

    def create_similarity_index(self):
        if not os.path.isfile(self.similarity_file):
            self.similarity_index = Similarity('./LSM/', self.corpora, self.num_topics)
            self.similarity_index.save(self.similarity_file)
        else:
            self.similarity_index = Similarity.load(self.similarity_file)
