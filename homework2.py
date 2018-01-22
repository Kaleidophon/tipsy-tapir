# -----------------------
#  Homework 2 Python file
# -----------------------

# -----------------------
#  Pre-handed code
# -----------------------
import logging
import sys
import os

import time


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
# write_run(
#     model_name='example',
#     data={
#         'Q1': ((1.0, 'DOC1'), (0.5, 'DOC2'), (0.75, 'DOC3')),
#         'Q2': ((-0.1, 'DOC1'), (1.25, 'DOC2'), (0.0, 'DOC3')),
#     },
#     out_f=sys.stdout,
#     max_objects_per_query=1000)

### Pyndri primer
import pyndri

index = pyndri.Index('index/')

# print("There are %d documents in this collection." % (index.maximum_document() - index.document_base()))
#
# example_document = index.document(index.document_base())
# print(example_document)
#
# token2id, id2token, _ = index.get_dictionary()
# print(list(id2token.items())[:15])
#
# print([id2token[word_id] for word_id in example_document[1] if word_id > 0])
#
# query_tokens = index.tokenize("University of Massachusetts")
# print("Query by tokens:", query_tokens)
# query_id_tokens = [token2id.get(query_token,0) for query_token in query_tokens]
# print("Query by ids with stopwords:", query_id_tokens)
# query_id_tokens = [word_id for word_id in query_id_tokens if word_id > 0]
# print("Query by ids without stopwords:", query_id_tokens)
#
# matching_words = sum([True for word_id in example_document[1] if word_id in query_id_tokens])
# print("Document %s has %d word matches with query: \"%s\"." % (example_document[0], matching_words, ' '.join(query_tokens)))
# print("Document %s and query \"%s\" have a %.01f%% overlap." % (example_document[0], ' '.join(query_tokens),matching_words/float(len(example_document[1]))*100))

### Parsing the query file
import collections
import io
import logging
import sys
#
# def parse_topics(file_or_files,
#                  max_topics=sys.maxsize, delimiter=';'):
#     assert max_topics >= 0 or max_topics is None
#
#     topics = collections.OrderedDict()
#
#     if not isinstance(file_or_files, list) and \
#             not isinstance(file_or_files, tuple):
#         if hasattr(file_or_files, '__iter__'):
#             file_or_files = list(file_or_files)
#         else:
#             file_or_files = [file_or_files]
#
#     for f in file_or_files:
#         assert isinstance(f, io.IOBase)
#
#         for line in f:
#             assert(isinstance(line, str))
#
#             line = line.strip()
#
#             if not line:
#                 continue
#
#             topic_id, terms = line.split(delimiter, 1)
#
#             if topic_id in topics and (topics[topic_id] != terms):
#                     logging.error('Duplicate topic "%s" (%s vs. %s).',
#                                   topic_id,
#                                   topics[topic_id],
#                                   terms)
#
#             topics[topic_id] = terms
#
#             if max_topics > 0 and len(topics) >= max_topics:
#                 break
#
#     return topics
#
# with open('./ap_88_89/topics_title', 'r') as f_topics:
#     print(parse_topics([f_topics]))

# ------------------------------------------------
# Task 1: Implement and compare lexical IR methods
# ------------------------------------------------

# IMPORTANT You should structure your code around the helper functions we provide below.

### Functions provided:
# with open('./ap_88_89/topics_title', 'r') as f_topics:
#     queries = parse_topics([f_topics])

index = pyndri.Index('index/')

num_documents = index.maximum_document() - index.document_base()

dictionary = pyndri.extract_dictionary(index)

# tokenized_queries = {
#     query_id: [dictionary.translate_token(token)
#                for token in index.tokenize(query_string)
#                if dictionary.has_token(token)]
#     for query_id, query_string in queries.items()}
#
# query_term_ids = set(
#     query_term_id
#     for query_term_ids in tokenized_queries.values()
#     for query_term_id in query_term_ids)

# print('Gathering statistics about', len(query_term_ids), 'terms.')
#
# inverted index creation.
# start_time = time.time()
#
# document_lengths = {}
# unique_terms_per_document = {}
#
# inverted_index = collections.defaultdict(dict)
# collection_frequencies = collections.defaultdict(int)
#
# total_terms = 0
#
# for int_doc_id in range(index.document_base(), index.maximum_document()):
#     ext_doc_id, doc_token_ids = index.document(int_doc_id)
#
#     document_bow = collections.Counter(
#         token_id for token_id in doc_token_ids
#         if token_id > 0)
#     document_length = sum(document_bow.values())
#
#     document_lengths[int_doc_id] = document_length
#     total_terms += document_length
#
#     unique_terms_per_document[int_doc_id] = len(document_bow)
#
#     for query_term_id in query_term_ids:
#         assert query_term_id is not None
#
#         document_term_frequency = document_bow.get(query_term_id, 0)
#
#         if document_term_frequency == 0:
#             continue
#
#         collection_frequencies[query_term_id] += document_term_frequency
#         inverted_index[query_term_id][int_doc_id] = document_term_frequency
#
# avg_doc_length = total_terms / num_documents
#
# print('Inverted index creation took', time.time() - start_time, 'seconds.')
#
# def run_retrieval(model_name, score_fn):
#     """
#     Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.
#     :param model_name: the name of the model (a string)
#     :param score_fn: the scoring function (a function - see below for an example)
#     """
#     run_out_path = '{}.run'.format(model_name)
#
#     if os.path.exists(run_out_path):
#         return
#
#     retrieval_start_time = time.time()
#
#     print('Retrieving using', model_name)
#
#     data = {}
#
#     # TODO: fill the data dictionary.
#     # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
#     with open(run_out_path, 'w') as f_out:
#         write_run(
#             model_name=model_name,
#             data=data,
#             out_f=f_out,
#             max_objects_per_query=1000)
#
# def tfidf(int_document_id, query_term_id, document_term_freq):
#     """
#     Scoring function for a document and a query term
#     :param int_document_id: the document id
#     :param query_token_id: the query term id (assuming you have split the query to tokens)
#     :param document_term_freq: the document term frequency of the query term
#     """
#     # TODO implement the function
#     score = 0
#     return score
#
# # combining the two functions above:
# run_retrieval('tfidf', tfidf)
#
# # TODO implement the rest of the retrieval functions
#
# # TODO implement tools to help you with the analysis of the results.
#
# ### End of provided functions

# ------------------------------
# Task 2: Latent Semantic Models
# ------------------------------

# -----------------------------------
# Task 3: Word embeddings for ranking
# -----------------------------------

from gensim.models import Word2Vec
import pyndri.compat
import time

print("Training word embeddings...")


def train_word_embeddings(pyndri_index, epochs, **model_parameters):
    vocabulary = pyndri.extract_dictionary(pyndri_index)
    sentences = pyndri.compat.IndriSentences(pyndri_index, vocabulary)

    word2vec = Word2Vec(**model_parameters)
    word2vec.build_vocab(sentences, trim_rule=None)

    print("Start training...")
    for epoch in range(epochs):
        start = time.time()

        word2vec.train(sentences, total_examples=len(sentences), epochs=1, compute_loss=True)

        end = time.time()
        print("Epoch #{} took {:.2f} seconds.".format(epoch+1, end-start))
        print("Loss epoch #{}: {}".format(epoch+1, word2vec.running_training_loss))

    return word2vec


def save_word2vec_model(model, path):
    model.save(path)


def load_word2vec_model(path, to_train=False):
    model = Word2Vec.load(path)

    if to_train:
        return model

    # In case it doesn't need to be trained, delete train code to free up ram
    word_vectors = model.wv
    del model
    return word_vectors


WORD_EMBEDDING_PARAMS = {
    "size": 300,  # Embedding size
    "window": 5,  # One-sided window size
    "sg": True,  # Skip-gram.
    "min_count": 5,  # Minimum word frequency.
    "sample": 1e-3,  # Sub-sample threshold.
    "hs": False,  # Hierarchical softmax.
    "negative": 10,  # Number of negative examples.
    "iter": 1,  # Number of iterations.
    "workers": 4  # Number of workers
}

w2v_model = train_word_embeddings(index, epochs=2, **WORD_EMBEDDING_PARAMS)
save_word2vec_model(w2v_model, "./w2v_test")

# ------------------------------
# Task 4: Learning to rank (LTR)
# ------------------------------


# --------------------
# Task 5: Write report
# --------------------

# Overleaf link: https://www.overleaf.com/13270283sxmcppswgnyd#/51107064/
