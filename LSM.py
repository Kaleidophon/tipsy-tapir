import gensim
import pyndri
from gensim import corpora
from gensim import models
from gensim.models import lsimodel
from gensim.models import tfidfmodel
import os
import pyndri.compat

index = pyndri.Index('index/')
dictionary = pyndri.extract_dictionary(index)

# 'Connect' class.
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

# getting tfidf of the terms
tfidf = tfidfmodel.TfidfModel(corpus = mm, id2word = dictionary, normalize = True)

# transform tfidf to latent space
lsi = lsimodel.LsiModel(corpus = tfidf[mm], id2word = dictionary, num_topics = 64)

# Transformed_corpora
tfidf_corpora = tfidf[mm]
lsi_corpora = lsi[tfidf[mm]]


for i, query in enumerate(queries.items()):
    print("\nScoring query {} out of {} queries".format(i, len(queries)))
    query_id, _ = query
    query_term_ids = tokenized_queries[query_id]

    query_scores = []

    query_bow = dictionary.doc2bow(query_term_ids)
    query_lsi = lsi[tfidf[query_bow]]

    for n, document_id in enumerate(document_ids):
        ext_doc_id, document_word_positions = index.document(document_id)
        score = 0
        for query_term_id in query_term_ids:
            document_term_frequency = inverted_index[query_term_id][document_id]
            score +=

        query_scores.append((score, ext_doc_id))

    data[query_id] = list(sorted(query_scores, reverse=True))[:max_objects_per_query]
