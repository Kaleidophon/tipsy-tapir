# -----------------------
#  Homework 2 Python file
# -----------------------

# -----------------------
#  Pre-handed code
# -----------------------
import sys
import os
import io
import logging
import collections

### Pyndri primer
import pyndri

import time

from math import log

import numpy as np

from PLM import PLM_score

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

# The following writes the run to standard output.
# In your code, you should write the runs to local
# storage in order to pass them to trec_eval.


write_run(
    model_name='example',
    data={
        'Q1': ((1.0, 'DOC1'), (0.5, 'DOC2'), (0.75, 'DOC3')),
        'Q2': ((-0.1, 'DOC1'), (1.25, 'DOC2'), (0.0, 'DOC3')),
    },
    out_f=sys.stdout,
    max_objects_per_query=1000)


index = pyndri.Index('index/')

total_number_of_documents = index.maximum_document() - index.document_base()

example_document = index.document(index.document_base())
# print(example_document)

token2id, id2token, id2df = index.get_dictionary()
id2tf = index.get_term_frequencies()

# print(list(id2token.items())[:15])

# print([id2token[word_id] for word_id in example_document[1] if word_id > 0])

query_tokens = index.tokenize("University of Massachusetts")
# print("Query by tokens:", query_tokens)
query_id_tokens = [token2id.get(query_token, 0) for query_token in query_tokens]
# print("Query by ids with stopwords:", query_id_tokens)
query_id_tokens = [word_id for word_id in query_id_tokens if word_id > 0]
# print("Query by ids without stopwords:", query_id_tokens)

matching_words = sum([True for word_id in example_document[1] if word_id in query_id_tokens])
# print("Document %s has %d word matches with query: \"%s\"." % (example_document[0], matching_words, ' '.join(query_tokens)))
# print("Document %s and query \"%s\" have a %.01f%% overlap." % (example_document[0], ' '.join(query_tokens),matching_words/float(len(example_document[1]))*100))

# Parsing the query file


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

# with open('./ap_88_89/topics_title', 'r') as f_topics:
    # print(parse_topics([f_topics]))

# ------------------------------------------------
# Task 1: Implement and compare lexical IR methods
# ------------------------------------------------

# IMPORTANT You should structure your code around the helper functions we provide below.

### Functions provided:
with open('./ap_88_89/topics_title', 'r') as f_topics:
    queries = parse_topics([f_topics])

index = pyndri.Index('index/')

num_documents = index.maximum_document() - index.document_base()

dictionary = pyndri.extract_dictionary(index)

tokenized_queries = {
    query_id: [dictionary.translate_token(token)
               for token in index.tokenize(query_string)
               if dictionary.has_token(token)]
    for query_id, query_string in queries.items()}

query_term_ids = set(
    query_term_id
    for query_term_ids in tokenized_queries.values()
    for query_term_id in query_term_ids)

print('Gathering statistics about', len(query_term_ids), 'terms.')

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

# tf_C = collections.defaultdict(lambda: 1)
print("Creating tf_c")

tf_C = collections.Counter()
# tf_C = collections.defaultdict(lambda: 1)

document_ids = list(range(index.document_base(), index.maximum_document()))
print("Number of query term ids: ", len(query_term_ids))
for term_id in query_term_ids:
    for document_id in document_ids:
        #term = inverted_index[term_id][document_id]
        tf_C[term_id] += inverted_index[term_id][document_id]

print("Done creating tf_C")


def cosine_similarity(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    res = dot_product / (norm_a * norm_b)
    if np.isnan(res):
        # If one of the vectors have zero length,
        # we can not score the similarity between the two vectors, so we assume the worst
        return 0.0  # TODO Isn't worst case -1?? Dennis

    return res


def query_document_similarity(query_term_ids, document_id):
    """
    Calculates the scores for both the query and the document given by the query_term_ids
    and the document_id in light of the given score function. These scores are then
    represented in a vector space where the axis is the query term, and
    each vector is represented by the score for each term.
    Finally we calculate the cosine similarity between the two vectors and return the value.

    :param query_term_ids:
    :param document_id:
    :param score_fn:
    :return:
    """
    # Assuming that words are not repeated in queries for now. TOOD: Remove assumption and do actual counts

    query_vector = np.array([tf_idf(query_term_id, 1/len(query_term_ids)) for query_term_id in query_term_ids])
    document_vector = np.array([tf_idf(query_term_id, inverted_index[query_term_id][document_id]) for query_term_id in query_term_ids])
    return cosine_similarity(query_vector, document_vector)


def run_retrieval(model_name, score_fn, document_ids, max_objects_per_query=1000):
    """
    Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.
    :param model_name: the name of the model (a string)
    :param score_fn: the scoring function (a function - see below for an example)
    :param accumulate_score: boolean indicating weather the score for each term should be accumulated or not.
    If accumulate_score is false, we will use the cosine similarity between document and query instead.
    """
    run_out_path = '{}.run'.format(model_name)

    run_id = 0
    while os.path.exists(run_out_path):
        run_id += 1
        run_out_path = '{}_{}.run'.format(model_name, run_id)

    print('Retrieving using', model_name)

    # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
    data = {}

    if model_name == "PLM":
        # Should probably just make this one of the global variables
        query_model = generate_query_likelihood_model()

    for i, query in enumerate(queries.items()):
        print("Scoring query {} out of {} queries".format(i, len(queries)))
        query_id, _ = query
        query_term_ids = tokenized_queries[query_id]

        query_scores = []

        for document_id in document_ids:

            if document_id == 10: return
            ext_doc_id, document_word_positions = index.document(document_id)
            score = 0
            if model_name == "PLM": # PLMs need the query in it's entirety
                print("Scoring document {} out of {} documents".format(document_id, index.maximum_document()))
                document_length = index.document_length(document_id)
                plm = PLM_score(query_term_ids, document_length, total_number_of_documents, document_word_positions, query_model, None)
                score = plm.best_position_strategy_score()

            else:
                for query_term_id in query_term_ids:
                    document_term_frequency = inverted_index[query_term_id][document_id]
                    score += score_fn(document_id, query_term_id, document_term_frequency)

            query_scores.append((score, ext_doc_id))

        data[query_id] = list(sorted(query_scores, reverse=True))[:max_objects_per_query]

    with open(run_out_path, 'w') as f_out:
        write_run(
            model_name=model_name,
            data=data,
            out_f=f_out,
            max_objects_per_query=max_objects_per_query)

    print("Done writing results to run file")
    return


def generate_query_likelihood_model():
    counter = collections.Counter()

    for query in queries.items():
        query_id, _ = query
        query_term_ids = tokenized_queries[query_id]

        for query_term_id in query_term_ids:
            counter[query_term_id] += 1

    total_elements = len(counter.items())
    model = {query_term_id : count / total_elements for query_term_id, count in counter.items()}
    # If a term has never been a part of a query we assign it probability 0 in the query model
    model = collections.defaultdict(int, model)
    return model


def idf(term_id):
    df_t = id2df[term_id]
    return log(total_number_of_documents) - log(df_t)


def tf_idf(_, term_id, document_term_freq):
    return log(1 + document_term_freq) * idf(term_id)


def bm25(document_id, term_id, document_term_freq):
    """
    :param document_id:
    :param term_id:
    :param document_term_freq: How many times this term appears in the given document
    :returns: A score for the given document in the light of the given query term

    Since all the scoring functions are supposed to only score one query term,
    the BM25 summation is being done outside this function.
    Do note that we leave the k_3 factor out, since all the queries
    in this assignment can be assumed to be reasonably short.
    """

    def bm25_formula(query_term_id, document_term_freq, l_d, l_average):
        enumerator = (k_1 + 1) * document_term_freq
        denominator = k_1 * ((1-b) + b * (l_d/l_average)) + document_term_freq
        bm25_score = idf(query_term_id)* enumerator/denominator

        if bm25_score == 0:
            return 0

        return log(bm25_score)

    k_1 = 1.2
    b = 0.75
    l_average = avg_doc_length
    l_d = index.document_length(document_id)

    return bm25_formula(term_id, document_term_freq, l_d, l_average)


def absolute_discounting(document_id, term_id, document_term_freq):
    discount = 0.1
    d = index.document_length(document_id)
    if d == 0: return 0
    number_of_unique_terms = len(set(index.document(document_id)[1]))
    return max(document_term_freq - discount, 0) / d + ((discount * number_of_unique_terms) / d) * (tf_C[term_id] / total_number_of_documents)

# run_retrieval('tfidf', tf_idf)
# run_retrieval('BM25', bm25)
import cProfile as profile

profile.run("run_retrieval('PLM', None, document_ids=document_ids)", sort=True)


# TODO implement the rest of the retrieval functions

# TODO implement tools to help you with the analysis of the results.

# End of provided functions

# ------------------------------
# Task 2: Latent Semantic Models
# ------------------------------

# -----------------------------------
# Task 3: Word embeddings for ranking
# -----------------------------------


# ------------------------------
# Task 4: Learning to rank (LTR)
# ------------------------------


# --------------------
# Task 5: Write report
# --------------------

# Overleaf link: https://www.overleaf.com/13270283sxmcppswgnyd#/51107064/
