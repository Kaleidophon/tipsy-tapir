# -----------------------
#  Homework 2 Python file
# -----------------------

# STD
import collections
import io
import logging
from math import log
import sys
import time
import os

# EXT
import numpy as np
import pyndri

# PROJECT
from embeddings import doc_centroid, VectorCollection, load_document_representations
from kernels import k_passage, k_gaussian, k_cosine, k_triangle, k_circle
from lsm import LSM
from plm import PLM


# Function to generate run and write it to out_f
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


# ------------------------
#  Commonly used functions
# ------------------------

def create_index_resources(index_path="index/"):
    index = pyndri.Index(index_path)
    token2id, id2token, id2df = index.get_dictionary()
    dictionary = pyndri.extract_dictionary(index)
    document_ids = list(range(index.document_base(), index.maximum_document()))
    return index, token2id, id2token, id2df, dictionary, document_ids


def create_query_resources(query_path='./ap_88_89/topics_title'):
    with open(query_path, 'r') as f_topics:
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

    return queries, tokenized_queries, query_terms_inverted, query_term_ids


def build_misc_resources(document_ids):
    # Inverted Index creation.
    start_time = time.time()

    inverted_index = collections.defaultdict(lambda: collections.defaultdict(int))
    tf_C = collections.Counter()
    query_word_positions = collections.defaultdict(lambda: collections.defaultdict(list))
    document_lengths = {}
    unique_terms_per_document = {}
    collection_length = 0
    total_terms = 0

    for int_doc_id in document_ids:
        ext_doc_id, doc_token_ids = index.document(int_doc_id)

        for pos, id_at_pos in enumerate(doc_token_ids):
            if id_at_pos in query_term_ids:
                for query_id in query_terms_inverted[id_at_pos]:
                    query_word_positions[doc_token_ids][int(query_id)].append(pos)

        document_bow = collections.Counter(
            token_id for token_id in doc_token_ids
            if token_id > 0
        )
        document_length = sum(document_bow.values())

        collection_length += len(doc_token_ids)

        document_lengths[int_doc_id] = document_length
        total_terms += document_length

        unique_terms_per_document[int_doc_id] = len(document_bow)

        for query_term_id in query_term_ids:
            assert query_term_id is not None

            tf_C[query_term_id] += inverted_index[query_term_id][int_doc_id]

            document_term_frequency = document_bow.get(query_term_id, 0)

            if document_term_frequency == 0:
                continue

            inverted_index[query_term_id][int_doc_id] = document_term_frequency

    print('Inverted index creation took', time.time() - start_time, 'seconds.')
    print("Done creating tf_c and query_word_positions.")
    avg_doc_length = total_terms / len(document_ids)
    return inverted_index, tf_C, query_word_positions, unique_terms_per_document, avg_doc_length, document_length, \
           collection_length


# ------------------------------------------------
# Task 1: Implement and compare lexical IR methods
# ------------------------------------------------


def run_retrieval(index, model_name, queries, document_ids, scoring_func, max_objects_per_query=1000,
                  target_dir="./lexical_results/", **resource_params):
    """
    Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.

    :param index: Pyndri index object.
    :type index: pyndri.Index
    :param model_name: Name of the model (used for screen printing).
    :type model_name: str
    :param queries: Queries to be evaluated.
    :type queries: dict
    :param document_ids: Collection of document ids in data set.
    :type document_ids: list or set or tuple
    :param scoring_func: Function that uses the prepared resources, query and document id to score a query and a doc.
    :type scoring_func: func
    :param max_objects_per_query: Only keep a certain number of best ranked documents per query.
    :type max_objects_per_query: int
    :param resource_params: Named arguments that are used to build resources used by the model for scoring.
    :type resource_params: dict
    """
    run_out_path = '{}.run'.format(model_name)

    run_id = 0
    while os.path.exists(run_out_path):
        run_id += 1
        run_out_path = '{}_{}.run'.format(model_name, run_id)

    print('Retrieving using', model_name)

    # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
    data = {}

    start = time.time()
    query_times = 0
    n_queries = len(queries)

    for i, query in enumerate(queries.items()):
        query_start_time = time.time()
        query_id, _ = query
        query_scores = []

        # Do actual scoring here
        for n, document_id in enumerate(document_ids):
            ext_doc_id, document_word_positions = index.document(document_id)
            score = scoring_func(index, query_id, document_id, **resource_params)
            query_scores.append((score, ext_doc_id))

        data[query_id] = list(sorted(query_scores, reverse=True))[:max_objects_per_query]

        query_end_time = time.time()
        query_time = query_end_time - query_start_time
        query_times += query_time
        average_query_time = query_times / max(i, 1)
        querys_left = len(queries) - i
        time_left = average_query_time * querys_left
        m, s = divmod(time_left, 60)
        h, m = divmod(m, 60)
        print(
            "\rAverage query time for query {} out of {}: {:.2f} seconds. {} hour(s), {} minute(s) and {:.2f} seconds "
            "remaining for {} queries.".format(
                i+1, n_queries, average_query_time, int(h), int(m), s, querys_left
            ), flush=True, end=""
        )

    # Write results
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open('{}{}'.format(target_dir, run_out_path), 'w') as f_out:
        write_run(
            model_name=model_name,
            data=data,
            out_f=f_out,
            max_objects_per_query=max_objects_per_query)

    print("Done writing results to run file")

    end = time.time()
    print("Retrieval took {:.2f} seconds.".format(end-start))
    return data


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


def idf(term_id, id2df, num_documents):
    df_t = id2df[term_id]
    return log(num_documents) - log(df_t)


def tf_idf(term_id, document_term_freq, tuning_parameter=None):
    return log(1 + document_term_freq) * idf(term_id)


def bm25(document_id, term_id, document_term_freq, avg_doc_length, tuning_parameter=None):
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


def LM_jelinek_mercer_smoothing(int_document_id, query_term_id, document_term_freq, collection_length, tf_C,
                                tuning_parameter=0.1):
    tf = document_term_freq
    lamb = tuning_parameter
    doc_length = index.document_length(int_document_id)
    C = collection_length

    try:
        prob_q_d = lamb * (tf / doc_length) + (1 - lamb) * (tf_C[query_term_id] / C)
    except ZeroDivisionError as err:
        prob_q_d = 0

    return np.log(prob_q_d)


def LM_dirichelt_smoothing(int_document_id, query_term_id, document_term_freq, collection_length, tuning_parameter=500):
    tf = document_term_freq
    mu = tuning_parameter
    doc_length = index.document_length(int_document_id)
    C = collection_length

    prob_q_d = (tf + mu * (tf_C[query_term_id] / C)) / (doc_length + mu)

    return np.log(prob_q_d)


def absolute_discounting(document_id, term_id, document_term_freq, num_unique_words, tuning_parameter=0.1):
    discount = tuning_parameter
    d = index.document_length(document_id)
    C = collection_length
    if d == 0: return 0
    number_of_unique_terms = num_unique_words[document_id]

    return np.log(max(document_term_freq - discount, 0) / d + ((discount * number_of_unique_terms) / d) * (tf_C[term_id] / C))


def create_all_lexical_run_files():
    print("##### Creating all lexical run files! #####")

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


def create_latent_semantic_model_runfiles():
    # LSI
    start = time.time()
    lsi = LSM('LSI', index)
    lsi.create_model()
    end = time.time()
    print("LSI model creation took {:.2f} seconds.".format(end - start))

    start = time.time()
    lsi.create_similarity_index()
    end = time.time()
    print("LSI similarity index creation took {:.2f} seconds.".format(end - start))

    start = time.time()
    lsi_reranking = lsm_reranking(ranked_queries=ranking['tfidf'], LSM_model=lsi)
    end = time.time()
    print("LSI reranking took {:.2f} seconds.".format(end - start))

    start = time.time()
    run_out_path = '{}.run'.format('LSI')
    with open('./lexical_results/{}'.format(run_out_path), 'w') as f_out:
        write_run(
            model_name='LSI',
            data=lsi_reranking,
            out_f=f_out,
            max_objects_per_query=1000)
    end = time.time()
    print("LSI run file creation {:.2f} seconds.".format(end - start))

    # LDA
    start = time.time()
    lda = LSM('LDA', index)
    lda.create_model()
    end = time.time()
    print("LDA model creation took {:.2f} seconds.".format(end - start))

    start = time.time()
    lda.create_similarity_index()
    end = time.time()
    print("LDA similarity index creation took {:.2f} seconds.".format(end - start))

    start = time.time()
    lda_reranking = lsm_reranking(ranked_queries=ranking['tfidf'], LSM_model=lda)
    end = time.time()
    print("LDA reranking took {:.2f} seconds.".format(end - start))

    start = time.time()
    run_out_path = '{}.run'.format('LDA')
    with open('./lexical_results/{}'.format(run_out_path), 'w') as f_out:
        write_run(
            model_name='LDA',
            data=lda_reranking,
            out_f=f_out,
            max_objects_per_query=1000)
    end = time.time()
    print("LDA run file creation {:.2f} seconds.".format(end - start))

# -----------------------------------
# Task 3 Word Embeddings for Ranking
# -----------------------------------


def run_retrieval_embeddings_So(index, model_name, queries, document_ids, id2token, vector_collection,
                             document_representations, combination_func, doc2repr=None):
    def score_func(index, query_id, document_id, **resource_params):
        id2token = resource_params["id2token"]
        vector_collection = resource_params["vector_collection"]
        tokenized_queries = resource_params["tokenized_queries"]
        query_token_ids = tokenized_queries[query_id]
        doc2repr = resource_params.get("doc2repr", None)
        vector_func_query = resource_params.get(
            "vector_func_query", lambda word, collection: collection.word_vectors[word]
        )
        query_vectors = []
        for query_token_id in query_token_ids:
            if query_token_id == 0: continue
            try:
                query_vectors.append(vector_func_query(id2token[query_token_id], vector_collection))
            except KeyError:
                # OOV word
                continue
        if len(query_vectors) == 0: return -1

        query_vector = combination_func(query_vectors)
        try:
            lookup_id = document_id if doc2repr is None else doc2repr[document_id]
            document_vector = document_representations[lookup_id]
        except KeyError as ie:
            # Empty documents, give worst score
            return -1

        return cosine_similarity(query_vector, document_vector)

    return run_retrieval(
        index, model_name, queries, document_ids, score_func,
        # Named key word arguments to build as resources before scoring
        id2_token=id2token, vector_collection=vector_collection, document_representations=document_representations,
        combination_func=combination_func, doc2repr=doc2repr
    )


def run_retrieval_embeddings_Savg(index, model_name, queries, document_ids, id2token, vector_collection,
                             document_representations, tokenized_queries, doc2repr=None):
    def score_func(index, query_id, document_id, **resource_params):
        id2token = resource_params["id2token"]
        vector_collection = resource_params["vector_collection"]
        tokenized_queries = resource_params["tokenized_queries"]
        doc2repr = resource_params.get("doc2repr", None)
        query_token_ids = tokenized_queries[query_id]
        vector_func_query = resource_params.get(
            "vector_func_query", lambda word, collection: collection.word_vectors[word]
        )
        query_vectors = []
        for query_token_id in query_token_ids:
            if query_token_id == 0: continue
            try:
                query_vectors.append(vector_func_query(id2token[query_token_id], vector_collection))
            except KeyError:
                # OOV word
                continue
        if len(query_vectors) == 0: return -1

        try:
            lookup_id = document_id if doc2repr is None else doc2repr[document_id]
            document_vector = document_representations[lookup_id]
        except KeyError as ie:
            # Empty documents, give worst score
            return -1

        return sum(
            [cosine_similarity(query_vector, document_vector) for query_vector in query_vectors]
        ) / len(query_vectors)

    return run_retrieval(
        index, model_name, queries, document_ids, score_func,
        # Named key word arguments to build as resources before scoring
        id2token=id2token, vector_collection=vector_collection, document_representations=document_representations,
        tokenized_queries=tokenized_queries, doc2repr=doc2repr
    )


def run_retrieval_plm(index, model_name, queries, document_ids, query_word_positions, background_model,
                      tokenized_queries, kernel=k_gaussian):
    def score_func(index, query_id, document_id, **resource_params):
        query_word_positions = resource_params["query_word_positions"]
        background_model = resource_params.get("background_model", None)
        document_length = index.document_length(document_id)
        plm = PLM(
            query_term_ids, document_length, query_word_positions[document_id][int(query_id)],
            background_model=background_model, kernel=kernel
        )
        score = plm.best_position_strategy_score()
        return score

    return run_retrieval(
        index, model_name, queries, document_ids, score_func=score_func,
        # Named key word arguments to build as resources before scoring
        query_word_positions=query_word_positions, kernel=kernel, background_model=background_model,
        tokenized_queries=tokenized_queries
    )


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
        return -1

    return res


def create_document_id_to_repr_map(document_ids):
    """
    Compensate for empty documents in document collections.
    Because of using hd5 storage, we can only store vectors of the same length, so no zero-length vectors for empty
    documents. Because they were disregarded during the strorage procedure, the alignment between document ids and
    indices of the list with their vector representations is now off and has to be fixed.

    It's a little dirty and hack-y, but was the best solution given the time constraint.
    """
    empty_ids = {93688, 102435, 104040, 121863, 121866, 122113, 147904, 149905, 153512, 154467, 155654}
    doc2repr = dict()

    zone = 1
    for i in range(1, len(document_ids)+1):
        if i in empty_ids:
            zone += 1
            continue

        doc2repr[i] = i - zone

    return doc2repr


# ------------------------------
# Task 4: Learning to rank (LTR)
# ------------------------------

if __name__ == "__main__":
    index, token2id, id2token, id2df, dictionary, document_ids = create_index_resources()
    num_documents = len(document_ids)

    queries, tokenized_queries, query_terms_inverted, query_term_ids = create_query_resources()
    inverted_index, tf_C, query_word_positions, unique_terms_per_document, avg_doc_length, document_length, \
           collection_length = build_misc_resources(document_ids)

    # create_all_lexical_run_files()

    # run_retrieval_plm("PLM", index, queries, document_ids, query_word_positions, background_model=None)
    print("Reading word embeddings...")
    vectors = VectorCollection.load_vectors("./w2v_60")
    doc2repr = create_document_id_to_repr_map(document_ids)
    print("Reading document representations...")
    doc_representations = load_document_representations("./win_representations_1_4")
    run_retrieval_embeddings_Savg(
        index, "embeddings_Savg", queries, document_ids, id2token=id2token, vector_collection=vectors,
        document_representations=doc_representations.get("doc_centroid"), tokenized_queries=tokenized_queries,
        doc2repr=doc2repr
    )
