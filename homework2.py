# -----------------------
#  Homework 2 Python file
# -----------------------

# Imports

import sys
import os
import io
import logging
import collections

### Pyndri primer
import pyndri

import time

from math import log

# -----------------------
#  Pre-handed code
# -----------------------

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

# ------------------------------------------------
# Initialize variables to use through the exercise
# ------------------------------------------------

index = pyndri.Index('index/')
token2id, id2token, id2df = index.get_dictionary()
num_documents = index.maximum_document() - index.document_base()
dictionary = pyndri.extract_dictionary(index)

# ------------------------------------------------
# Task 1: Implement and compare lexical IR methods
# ------------------------------------------------

### Functions provided:
with open('./ap_88_89/topics_title', 'r') as f_topics:
    queries = parse_topics([f_topics])

tokenized_queries = {
    query_id: [dictionary.translate_token(token)
               for token in index.tokenize(query_string)
               if dictionary.has_token(token)]
    for query_id, query_string in queries.items()}

# print(dictionary.has_token('airbus'))
# print(dictionary.translate_token('airbus'))
# print(index.tokenize('Asbestos Related Lawsuits'))
# print(tokenized_queries)

query_term_ids = set(
    query_term_id
    for query_term_ids in tokenized_queries.values()
    for query_term_id in query_term_ids)

# print(query_term_ids)

print('Gathering statistics about', len(query_term_ids), 'terms.')

# Inverted Index creation.
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


def run_retrieval(model_name, score_fn):
    """
    Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.

    :param model_name: the name of the model (a string)
    :param score_fn: the scoring function (a function - see below for an example)
    """
    run_out_path = '{}.run'.format(model_name)

    if os.path.exists(run_out_path):
        return

    retrieval_start_time = time.time()

    print('Retrieving using', model_name)

    data = {}

    # XXX: fill the data dictionary.
    # The dictionary data should have the form: query_id --> (document_score, external_doc_id)

    for i, query in enumerate(queries.items()):
        print("Scoring query {} out of {} queries".format(i, len(queries)))
        query_id, _ = query
        query_term_ids = tokenized_queries[query_id]

        scores = []

        for document_id in range(index.document_base(), index.maximum_document()):
            ext_doc_id, _ = index.document(document_id)
            score = 0
            for term_id in query_term_ids:
                document_term_freq = inverted_index[term_id][document_id]
                score += score_fn(document_id, term_id, document_term_freq)
            scores.append((score, ext_doc_id))

        data[query_id] = scores

    with open(run_out_path, 'w') as f_out:
        write_run(
            model_name=model_name,
            data=data,
            out_f=f_out,
            max_objects_per_query=1000)

def idf(term_id, num_documents):
    df_t = id2df[term_id]
    return log(num_documents) - log(df_t)

def tfidf(int_document_id, query_term_id, document_term_freq):
    tf = document_term_freq
    return log(1 + tf) * idf(query_term_id, num_documents)

def bm25(int_document_id, query_term_id, document_term_freq):
    l_d = index.document_length(int_document_id)
    l_avg = avg_doc_length
    k_1 = 1.2
    b = 0.75
    tf = document_term_freq
    bm25 = idf(query_term_id, num_documents)*((k_1 + 1) * tf) / (k_1 * ((1-b) + b * (l_d/l_avg)) + tf)
    return bm25

tf_C = collections.defaultdict(int)

for term_id in query_term_ids:
    for document_id in range(index.document_base(), index.maximum_document()):
        tf_C[term_id] += inverted_index[term_id][document_id]


def LM_jelinek_mercer_smoothing(int_document_id, query_term_id, document_term_freq):
    tf = document_term_freq
    lamb = 0.1
    doc_length = index.document_length(int_document_id)
    C = num_documents

    try:
        prob_q_d = lamb * (tf / doc_length) + (1 - lamb) * (tf_C[query_term_id] / C)
    except ZeroDivisionError as err:
        prob_q_d = 0

    return prob_q_d

def LM_dirichelt_smoothing(int_document_id, query_term_id, document_term_freq):
    tf = document_term_freq
    mu = 500
    doc_length = index.document_length(int_document_id)
    C = num_documents

    prob_q_d = (tf + mu * (tf_C[query_term_id] / C)) / (doc_length + mu)

    return prob_q_d

def LM_absolute_discounting(int_document_id, query_term_id, document_term_freq):
    delta = 0.1
    doc_length = index.document_length(int_document_id)
    document = index.document(index.document_base())
    num_unique_terms = len(set(document[1]))
    C = num_documents

    try:
        prob_q_d = max(document_term_freq - delta, 0)/doc_length + ((delta * num_unique_terms)/doc_length) * (tf_C[query_term_id]/C)
    except ZeroDivisionError as err:
        prob_q_d = 0 

    return prob_q_d

# combining the two functions above:
# run_retrieval('tfidf', tfidf)
# run_retrieval('bm25', bm25)
# run_retrieval('LM_jelinek_mercer_smoothing', LM_jelinek_mercer_smoothing)
# run_retrieval('LM_dirichelt_smoothing', LM_dirichelt_smoothing)
run_retrieval('LM_absolute_discounting', LM_absolute_discounting)


# # TODO implement the rest of the retrieval functions
#
# # TODO implement tools to help you with the analysis of the results.
#
#
# # # ------------------------------
# # # Task 2: Latent Semantic Models
# # # ------------------------------
# #
# # # -----------------------------------
# # # Task 3: Word embeddings for ranking
# # # -----------------------------------
# #
# #
# # # ------------------------------
# # # Task 4: Learning to rank (LTR)
# # # ------------------------------
# #
# #
# # # --------------------
# # # Task 5: Write report
# # # --------------------
# #
# # # Overleaf link: https://www.overleaf.com/13270283sxmcppswgnyd#/51107064/
