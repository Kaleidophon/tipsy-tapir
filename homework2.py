# -----------------------
#  Homework 2 Python file
# -----------------------


# Imports

import collections
import io
import logging
import os

import sys
import time

from math import log

import numpy as np
### Pyndri primer
import pyndri

from scipy.stats import ttest_rel

from PLM import PLM
from kernels import k_passage, k_gaussian, k_cosine, k_triangle, k_circle
from LSM import LSM


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
document_ids = list(range(index.document_base(), index.maximum_document()))
# Ranking dictionary
ranking = {}

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

# Record in which query specific query terms are occurring (inverted index)
query_terms_inverted = collections.defaultdict(set)
for query_id, query_term_ids in tokenized_queries.items():
    for query_term_id in query_term_ids:
        # A lookup-table for what queries this term appears in
        query_terms_inverted[query_term_id].add(int(query_id))


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
# Experimental variable
collection_length = 0

for int_doc_id in range(index.document_base(), index.maximum_document()):
    ext_doc_id, doc_token_ids = index.document(int_doc_id)

    document_bow = collections.Counter(
        token_id for token_id in doc_token_ids
        if token_id > 0)
    document_length = sum(document_bow.values())

    collection_length += len(doc_token_ids)

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
print("Creating tf_c, query_word_positions and num_unique_words")
tf_C = collections.Counter()
num_unique_words = collections.defaultdict(int)
# Document id + Query id -> Positions of terms occurring in that query inside the document
query_word_positions = collections.defaultdict(lambda: collections.defaultdict(list))
# tf_C = collections.defaultdict(lambda: 1)

print("Number of query term ids: ", len(query_term_ids))
for document_id in document_ids:
    _, positions = index.document(document_id)
    document_words = collections.Counter(positions)
    num_unique_words[document_id] = len(document_words)

    for pos, id_at_pos in enumerate(positions):
        if id_at_pos in query_term_ids:
            for query_id in query_terms_inverted[id_at_pos]:
                query_word_positions[document_id][int(query_id)].append(pos)

    for term_id in query_term_ids:
        #term = inverted_index[term_id][document_id]
        tf_C[term_id] += inverted_index[term_id][document_id]

print("Done creating tf_c, query_word_positions and num_unique_words")

def run_retrieval(model_name, score_fn, document_ids, max_objects_per_query=1000, **retrieval_func_params):
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

    ranking[model_name] = collections.defaultdict(list)

    # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
    data = {}

    # XXX: fill the data dictionary.
    # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
    if model_name == "PLM":
        # Should probably just make this one of the global variables
        query_word_positions = retrieval_func_params["query_word_positions"]
        kernel = retrieval_func_params.get("kernel", k_gaussian)
    elif model_name == "Embeddings":
        # TODO: Fetch Embedding parameters here
        pass

    start = time.time()

    query_times = 0

    for i, query in enumerate(queries.items()):

        print("Scoring query {} out of {} queries".format(i, len(queries)))

        query_start_time = time.time()

        query_id, _ = query
        query_term_ids = tokenized_queries[query_id]

        scores = []

        for n, document_id in enumerate(document_ids):
            ext_doc_id, document_word_positions = index.document(document_id)

            score = 0
            # TODO: Add embedding scoring
            if model_name == "PLM":  # PLMs need the query in it's entirety
                # if n % 100 == 0:
                    # print("\rScoring document {} out of {} documents ({:.2f} %)".format(
                    #         document_id, index.maximum_document(), document_id / index.maximum_document() * 100
                    #     ), flush=True, end=""
                    # )
                document_length = index.document_length(document_id)
                plm = PLM(

                    query_term_ids, document_length, query_word_positions[document_id][int(query_id)],
                    background_model=None, kernel=kernel
                )
                score = plm.best_position_strategy_score()
            else:
                for query_term_id in query_term_ids:
                    document_term_frequency = inverted_index[query_term_id][document_id]
                    score += score_fn(document_id, query_term_id, document_term_frequency, \
                            tuning_parameter=retrieval_func_params["tuning_parameter"])
            ranking[model_name][query_id].append((score, ext_doc_id))

        ranking[model_name][query_id] = list(sorted(ranking[model_name][query_id], reverse=True))[:max_objects_per_query]

        query_end_time = time.time()
        query_time = query_end_time - query_start_time
        query_times += query_time
        average_query_time = query_times / max(i, 1)
        querys_left = len(queries) - i
        print("\rThe average query time is {} seconds, so for the remaining {} querys we estimate that it will take {} seconds"\
                .format(average_query_time, querys_left, average_query_time * querys_left))

    with open('./lexical_results/{}'.format(run_out_path), 'w') as f_out:
        write_run(
            model_name=model_name,
            data=ranking[model_name],
            out_f=f_out,
            max_objects_per_query=1000)

def idf(term_id):
    df_t = id2df[term_id]
    return log(num_documents) - log(df_t)

def tf_idf(_, term_id, document_term_freq, tuning_parameter=None):
    return log(1 + document_term_freq) * idf(term_id)

def bm25(document_id, term_id, document_term_freq, tuning_parameter=None):
    """
    :param document_id:
    :param term_id:
    :param document_term_freq: How many times this term appears in the given document
    :_ unused tuning parameter
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
        return bm25_score

    k_1 = 1.2
    b = 0.75
    l_average = avg_doc_length
    l_d = index.document_length(document_id)

    return bm25_formula(term_id, document_term_freq, l_d, l_average)

def LM_jelinek_mercer_smoothing(int_document_id, query_term_id, document_term_freq, tuning_parameter=0.1):
    tf = document_term_freq
    lamb = tuning_parameter
    doc_length = index.document_length(int_document_id)
    C = collection_length

    try:
        prob_q_d = lamb * (tf / doc_length) + (1 - lamb) * (tf_C[query_term_id] / C)
    except ZeroDivisionError as err:
        prob_q_d = 0

    return np.log(prob_q_d)

def LM_dirichelt_smoothing(int_document_id, query_term_id, document_term_freq, tuning_parameter=500):
    tf = document_term_freq
    mu = tuning_parameter
    doc_length = index.document_length(int_document_id)
    C = collection_length

    prob_q_d = (tf + mu * (tf_C[query_term_id] / C)) / (doc_length + mu)

    return np.log(prob_q_d)

def absolute_discounting(document_id, term_id, document_term_freq, tuning_parameter=0.1):
    discount = tuning_parameter
    d = index.document_length(document_id)
    C = collection_length
    if d == 0: return 0
    number_of_unique_terms = num_unique_words[document_id]

    return np.log(max(document_term_freq - discount, 0) / d + ((discount * number_of_unique_terms) / d) * (tf_C[term_id] / C))

def create_all_lexical_run_files():
    print("##### Creating all run files! #####")

    start = time.time()
    print("Running TFIDF")
    run_retrieval('tfidf', tf_idf, document_ids, tuning_parameter=None)
    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))

    start = time.time()
    print("Running BM25")
    run_retrieval('BM25', bm25, document_ids, tuning_parameter=None)
    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))

    j_m__lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for val in j_m__lambda_values:
        start = time.time()
        print("Running LM_jelinek", val)
        run_retrieval('jelinek_mercer_{}'.format(str(val).replace(".", "_")), LM_jelinek_mercer_smoothing, document_ids, tuning_parameter=val)
        end = time.time()
        print("Retrieval took {:.2f} seconds.".format(end-start))

    dirichlet_values = [500, 1000, 1500]
    for val in dirichlet_values:
        start = time.time()
        print("Running Dirichlet", val)
        run_retrieval('dirichlet_mu_{}'.format(str(val).replace(".", "_")), LM_dirichelt_smoothing, document_ids, tuning_parameter=val)
        end = time.time()
        print("Retrieval took {:.2f} seconds.".format(end-start))

    absolute_discounting_values = j_m__lambda_values
    for val in absolute_discounting_values:
        start = time.time()
        print("Running ABS_discount", val)
        run_retrieval('abs_disc_delta_{}'.format(str(val).replace(".", "_")), absolute_discounting, document_ids, tuning_parameter=val)
        end = time.time()
        print("Retrieval took {:.2f} seconds.".format(end-start))

    start = time.time()
    run_retrieval('PLM_passage', None, document_ids=document_ids, query_word_positions=query_word_positions, kernel=k_passage)
    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))

    start = time.time()
    run_retrieval('PLM_gaussian', None, document_ids=document_ids, query_word_positions=query_word_positions, kernel=k_gaussian)
    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))
    start = time.time()

    run_retrieval('PLM_triangle', None, document_ids=document_ids, query_word_positions=query_word_positions, kernel=k_triangle)
    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))
    start = time.time()

    run_retrieval('PLM_cosine', None, document_ids=document_ids, query_word_positions=query_word_positions, kernel=k_cosine)
    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))
    start = time.time()

    run_retrieval('PLM_circle', None, document_ids=document_ids, query_word_positions=query_word_positions, kernel=k_circle)
    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))

# create_all_run_files()

# TODO implement tools to help you with the analysis of the results.

# End of provided functions

# ------------------------------
# Task 2: Latent Semantic Models
# ------------------------------

def lsm_reranking(ranked_queries, LSM_model):
    reranking = collections.defaultdict(list)
    LSM_model.similarity_index.num_best = None

    for query_id, docs in ranked_queries.items():
        query_terms = [id2token[term_id] for term_id in tokenized_queries[query_id]]
        query_bow = LSM_model.dictionary.doc2bow(query_terms)
        query_lsm = LSM_model.model[query_bow]

        # Get similarity of query with all documents
        sims = LSM_model.similarity_index[query_lsm]

        for document_id in document_ids:
            ext_doc_id, _ = index.document(document_id)
            reranking[query_id].append((sims[document_id-1], ext_doc_id))

    for query_id in reranking:
        reranking[query_id] = list(sorted(reranking[query_id], reverse=True))[:1000]

    return reranking

# LSI
start = time.time()
lsi = LSM('LSI', index)
lsi.create_model()
end = time.time()
print("LSI model creation took {:.2f} seconds.".format(end-start))

start = time.time()
lsi.create_similarity_index()
end = time.time()
print("LSI similarity index creation took {:.2f} seconds.".format(end-start))

start = time.time()
lsi_reranking = lsm_reranking(ranked_queries = ranking['tfidf'], LSM_model=lsi)
end = time.time()
print("LSI reranking took {:.2f} seconds.".format(end-start))

start = time.time()
run_out_path = '{}.run'.format('LSI')
with open('./lexical_results/{}'.format(run_out_path), 'w') as f_out:
    write_run(
        model_name='LSI',
        data=lsi_reranking,
        out_f=f_out,
        max_objects_per_query=1000)
end = time.time()
print("LSI run file creation {:.2f} seconds.".format(end-start))

# LDA
start = time.time()
lda = LSM('LDA', index)
lda.create_model()
end = time.time()
print("LDA model creation took {:.2f} seconds.".format(end-start))

start = time.time()
lda.create_similarity_index()
end = time.time()
print("LDA similarity index creation took {:.2f} seconds.".format(end-start))

start = time.time()
lda_reranking = lsm_reranking(ranked_queries = ranking['tfidf'], LSM_model=lda)
end = time.time()
print("LDA reranking took {:.2f} seconds.".format(end-start))

start = time.time()
run_out_path = '{}.run'.format('LDA')
with open('./lexical_results/{}'.format(run_out_path), 'w') as f_out:
    write_run(
        model_name='LDA',
        data=lda_reranking,
        out_f=f_out,
        max_objects_per_query=1000)
end = time.time()
print("LDA run file creation {:.2f} seconds.".format(end-start))

-----------------------------------
Task 3: Word embeddings for ranking
-----------------------------------

TODO: Load model and stuff

class VectorCollection:

    def __init__(self, word_vectors, context_vectors):
        self.word_vectors = word_vectors
        self.context_vectors = context_vectors

def calculate_document_centroids(pyndri_index, vector_collection, stop_words=tuple(),
                                 vector_func=lambda word, collection: collection.word_vectors[word]):
    centroids = dict()

    token2id, id2token, _ = pyndri_index.get_dictionary()

    stop_word_ids = []
    if len(stop_words) > 0:
        stop_word_ids.extend([token2id[stop_word] for stop_word in stop_words])
        stop_word_ids = set(stop_word_ids)

    for document_id in range(pyndri_index.document_base(), pyndri_index.maximum_document()):
        ext_doc_id, token_ids = pyndri_index.document(document_id)
        _, id2token, _ = pyndri_index.get_dictionary()

        token_ids = [token_id for token_id in token_ids if token_id not in stop_word_ids]  # Filter out stop words
        centroid = sum([vector_func(id2token[token_id], vector_collection) for token_id in token_ids]) / len(token_ids)

        centroids[document_id] = centroid

    return centroids

def score_by_summing(query_token_ids, pyndri_index, vector_collection,
                     vector_func_query=lambda word, collection: collection.word_vectors[word],
                     vector_func_doc=lambda word, collection: collection.word_vectors[word], **kwargs):
    """
    Score a query and documents by taking the word embeddings of the words they contain and simply sum them,
    afterwards comparing the summed vectors with cosine similarity.
    """
    # Get query vector
    _, id2token, _ = pyndri_index.get_dictionary()
    # Just sum
    query_vector = sum([
        vector_func_query(id2token[query_token_id], vector_collection) for query_token_id in query_token_ids
    ])

    # Score documents
    document_scores = []
    for document_id in range(pyndri_index.document_base(), pyndri_index.maximum_document()):
        ext_doc_id, token_ids = pyndri_index.document(document_id)

        document_vector = sum([vector_func_doc(id2token[token_id], vector_collection) for token_id in token_ids])

        # TODO: Add cosine similarity here
        score = 0

        document_scores.append((score, ext_doc_id))

    return document_scores

def score_by_centroids(query_token_ids, pyndri_index, vector_collection, document_centroids, stop_words=tuple(),
                       vector_func_query=lambda word, collection: collection.word_vectors[word], **kwargs):
    """
    Score a query and documents by taking the word embeddings of the words they contain and calculate the centroids.
    Finally, compare the centroids using cosine similarities.
    """
    # Get query vector
    token2id, id2token, _ = pyndri_index.get_dictionary()

    # Remove stopwords from query / documents
    stop_word_ids = []
    if len(stop_words) > 0:
        stop_word_ids.extend([token2id[stop_word] for stop_word in stop_words])

    # Just sum
    # Filter out stop words
    query_token_ids = [query_token_id for query_token_id in query_token_ids if query_token_id not in stop_word_ids]
    query_vector = sum([
        vector_func_query(id2token[query_token_id], vector_collection) for query_token_id in query_token_ids
    ]) / len(query_token_ids)

    # Score documents
    document_scores = []
    for document_id in range(pyndri_index.document_base(), pyndri_index.maximum_document()):
        ext_doc_id, token_ids = pyndri_index.document(document_id)

        document_vector = document_centroids[document_id]

        # TODO: Add cosine similarity here
        score = 0

        document_scores.append((score, ext_doc_id))

    return document_scores

# ------------------------------
# Task 4: Learning to rank (LTR)
# ------------------------------

# TODO implement the rest of the retrieval functions

# TODO implement tools to help you with the analysis of the results.

# --------------------
# Task 5: Write report
# --------------------

# Overleaf link: https://www.overleaf.com/13270283sxmcppswgnyd#/51107064/
