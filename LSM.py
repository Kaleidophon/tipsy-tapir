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
import sys
import io
import logging
import collections
import time
import heapq
from operator import itemgetter

from math import log

import numpy as np

# Hyperparameter to optimize
NUM_TOPICS = 50

index = pyndri.Index('index/')
dictionary = pyndri.extract_dictionary(index)
docs = pyndri.compat.IndriSentences(index, dictionary)
num_documents = index.maximum_document() - index.document_base()
token2id, id2token, id2df = index.get_dictionary()
document_ids = list(range(index.document_base(), index.maximum_document()))



# Funtion to generate run and write it to out_f
def write_run(model_name, data, out_f,
              max_objects_per_query=sys.maxsize,
              skip_sorting=False):
    """
    Write a run to an output file.
    Parameters:
        - model_name: identifier of run.
        - data: dictionary mapping topic_id to object_assesments;
            object_assesments is an iterable (list or tuple) of
            (relevance, object_id) pairs.
            The object_assesments iterable is sorted by decreasing order.
        - out_f: output file stream.
        - max_objects_per_query: cut-off for number of objects per query.
    """
    for subject_id, object_assesments in data.items():
        if not object_assesments:
            logging.warning('Received empty ranking for %s; ignoring.',
                            subject_id)

            continue

        # Probe types, to make sure everything goes alright.
        # assert isinstance(object_assesments[0][0], float) or \
        #     isinstance(object_assesments[0][0], np.float32)
        assert isinstance(object_assesments[0][1], str) or \
            isinstance(object_assesments[0][1], bytes)

        if not skip_sorting:
            object_assesments = sorted(object_assesments, reverse=True)

        if max_objects_per_query < sys.maxsize:
            object_assesments = object_assesments[:max_objects_per_query]

        if isinstance(subject_id, bytes):
            subject_id = subject_id.decode('utf8')

        for rank, (relevance, object_id) in enumerate(object_assesments):
            if isinstance(object_id, bytes):
                object_id = object_id.decode('utf8')

            out_f.write(
                '{subject} Q0 {object} {rank} {relevance} '
                '{model_name}\n'.format(
                    subject=subject_id,
                    object=object_id,
                    rank=rank + 1,
                    relevance=relevance,
                    model_name=model_name))

# Function to parse the query file
def parse_topics(file_or_files,
                 max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    if not isinstance(file_or_files, list) and \
            not isinstance(file_or_files, tuple):
        if hasattr(file_or_files, '__iter__'):
            file_or_files = list(file_or_files)
        else:
            file_or_files = [file_or_files]

    for f in file_or_files:
        assert isinstance(f, io.IOBase)

        for line in f:
            assert(isinstance(line, str))

            line = line.strip()

            if not line:
                continue

            topic_id, terms = line.split(delimiter, 1)

            if topic_id in topics and (topics[topic_id] != terms):
                    logging.error('Duplicate topic "%s" (%s vs. %s).',
                                  topic_id,
                                  topics[topic_id],
                                  terms)

            topics[topic_id] = terms

            if max_topics > 0 and len(topics) >= max_topics:
                break

    return topics

with open('./ap_88_89/topics_title', 'r') as f_topics:
    queries = parse_topics([f_topics])

tokenized_queries = {
    query_id: [dictionary.translate_token(token)
               for token in index.tokenize(query_string)
               if dictionary.has_token(token)]
    for query_id, query_string in queries.items()}

# Record in which query specific query terms are occurring (inverted index)
query_terms_inverted = collections.defaultdict(set)
for query_id, query_term_ids in tokenized_queries.items():
    for query_term_id in query_term_ids:
        query_terms_inverted[query_term_id].add(query_id)


query_term_ids = set(
    query_term_id
    for query_term_ids in tokenized_queries.values()
    for query_term_id in query_term_ids)

# inverted index creation.
start_time = time.time()

document_lengths = {}
unique_terms_per_document = {}

inverted_index = collections.defaultdict(lambda: collections.defaultdict(int))
collection_frequencies = collections.defaultdict(int)

total_terms = 0

for int_doc_id in range(index.document_base(), index.maximum_document()):
    ext_doc_id, doc_token_ids = index.document(int_doc_id)

    document_bow = collections.Counter(
        token_id for token_id in doc_token_ids
        if token_id > 0)
    document_length = sum(document_bow.values())

    document_lengths[int_doc_id] = document_length
    total_terms += document_length

    unique_terms_per_document[int_doc_id] = len(document_bow)

    for query_term_id in query_term_ids:
        assert query_term_id is not None

        document_term_frequency = document_bow.get(query_term_id, 0)

        if document_term_frequency == 0:
            continue

        collection_frequencies[query_term_id] += document_term_frequency
        inverted_index[query_term_id][int_doc_id] = document_term_frequency

avg_doc_length = total_terms / num_documents

print('Inverted index creation took', time.time() - start_time, 'seconds.')

def idf(term_id, num_documents):
    df_t = id2df[term_id]
    return log(num_documents) - log(df_t)

def tfidf(int_document_id, query_term_id, document_term_freq):
    tf = document_term_freq
    return log(1 + tf) * idf(query_term_id, num_documents)

tfidf_ranking = collections.defaultdict(list)

for document_id in range(index.document_base(), index.maximum_document()):
    ext_doc_id, _ = index.document(document_id)

    scores = []

    for i, query in enumerate(queries.items()):
        query_id, _ = query
        query_term_ids = tokenized_queries[query_id]

        score = 0

        for term_id in query_term_ids:
            document_term_freq = inverted_index[term_id][document_id]
            score += tfidf(document_id, term_id, document_term_freq)
        tfidf_ranking[query_id].append((score, ext_doc_id))

for query_id in tfidf_ranking:
    tfidf_ranking[query_id] = list(sorted(tfidf_ranking[query_id], reverse=True))[:1000]

# Connector class.
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
    similarity_index = Similarity('./sim_index/', lsi_corpora, NUM_TOPICS)
    similarity_index.save('sim_index.index')
else:
    similarity_index = Similarity.load('sim_index.index')

def lsi_score(ranked_queries, similarity_index):
    lsi_reranking = collections.defaultdict(list)
    similarity_index.num_best = None

    for query_id, docs in ranked_queries.items():
        query_terms = [id2token[term_id] for term_id in tokenized_queries[query_id]]
        query_bow = dictionary.doc2bow(query_terms)
        query_lsi = lsi[query_bow]

        # Get similarity of query with all documents
        sims = similarity_index[query_lsi]

        for document_id in document_ids:
            ext_doc_id, _ = index.document(document_id)
            lsi_reranking[query_id].append((sims[document_id-1], ext_doc_id))

    for query_id in lsi_reranking:
        lsi_reranking[query_id] = list(sorted(lsi_reranking[query_id], reverse=True))[:1000]

    return lsi_reranking

run_out_path = '{}.run'.format('LSI')

lsi_reranking = lsi_score(ranked_queries = tfidf_ranking, similarity_index=similarity_index)

with open('./lexical_results/{}'.format(run_out_path), 'w') as f_out:
    write_run(
        model_name='LSI',
        data=lsi_reranking,
        out_f=f_out,
        max_objects_per_query=1000)
