# STD
import sys
import logging
import io
from collections import Counter, defaultdict, OrderedDict

# EXT
import h5py
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


def parse_topics(file_or_files,
                 max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = OrderedDict()

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


def idf(term_id, id2df, number_of_documents):
    df_t = id2df[term_id]
    return np.log(number_of_documents) - np.log(df_t)


def tf_idf(term_id, document_term_freq, id2df, number_of_documents):
    return np.log(1 + document_term_freq) * idf(term_id, id2df, number_of_documents)


def doc_centroid(vectors, cache=None, **kwargs):
    """ Calculate the centroid of some vectors. """
    return np.add.reduce(vectors) / len(vectors) if len(vectors) > 0 else np.array([]), cache


def _doc_kmeans(vectors, k=None, **kwargs):
    """
    Helper function for doc_kmeans.

    Calculate the centroid of vectors after clustering them using k-means and only keeping those which belong to the
    biggest cluster. If k is None, it will be chosen using a heuristic.
    """
    # Simple cases
    if len(vectors) == 0:
        return np.array([])
    if len(vectors) == 1:
        return vectors

    def most_common(lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    # Dirty but fast heuristic
    if k is None:
        k = int(np.sqrt(len(vectors) / 2))

    kmeans = KMeans(n_clusters=k).fit(vectors)
    labels = kmeans.labels_
    biggest_cluster = most_common(labels)  # Get label of biggest cluster

    # Return the vectors belonging to it
    return [vector for vector, label in zip(vectors, labels) if label == biggest_cluster]


def doc_kmeans(vectors, k=None, cache=None, **kwargs):
    """
    Calculate the centroid of vectors after clustering them using k-means and only keeping those which belong to the
    biggest cluster. If k is None, it will be chosen using a heuristic.
    """
    filtered_vectors = _doc_kmeans(vectors, k)

    # Cache most recent vectors for doc_kmeans_tfidf so you don't have to do it twice
    if cache is not None:
        cache["kmeans_filtered"] = filtered_vectors
    return doc_centroid(filtered_vectors), cache  # Calculate the centroid with vector from biggest cluster.


def doc_tfidf_scaling(vectors, token_ids, document_term_freqs, id2df, number_of_documents, document_id, cache=dict(),
                      **kwargs):
    """
    Calculate the centroid of a cluster, but scale the vectors by their tf-idf value.
    """
    # Easy case, empty document
    if len(vectors) == 0:
        return np.array([])

    for i, (vector, token_id) in enumerate(zip(vectors, token_ids)):
        # Use caching
        if hasattr(cache, "tfidf"):
            tf_idf_values = cache["tfidf"]
        else:
            tf_idf_values = cache["tfidf"] = dict()
        term_tf_idf = tf_idf_values.get(
            (token_id, document_id),
            tf_idf(token_id, document_term_freqs[token_id][document_id], id2df, number_of_documents)
        )
        cache["tfidf"][(token_id, document_id)] = term_tf_idf

        vectors[i] = vector * term_tf_idf  # Scale

    return doc_centroid(vectors), cache  # Now compute centroid


class VectorCollection:
    """ Class to store word embeddings created by Word2Vec. """
    def __init__(self, word_vectors, context_vectors):
        self.word_vectors = {
            word: word_vectors.syn0[vocab_item.index] for word, vocab_item in word_vectors.vocab.items()
        }  # W_in vectors
        self.context_vectors = {
            word_vectors.index2word[i]: vec for i, vec in enumerate(context_vectors)
        }  # W_out vectors

    @staticmethod
    def load_vectors(path, to_train=False):
        """
        Load word embeddings created by gensim Word2Vec from a directory.

        If the to_train flag is set, the whole model will be returned, otherwise just an instance of VectorCollection
        only containing the embeddings to save memory.
        """
        model = Word2Vec.load(path)

        if to_train:
            return model

        # In case it doesn't need to be trained, delete train code to free up ram
        word_vectors = model.wv

        context_vectors = dict()
        if hasattr(model, "syn1"):
            # For hierarchical softmax
            context_vectors = model.syn1
        elif hasattr(model, "syn1neg"):
            # For negative sampling
            context_vectors = model.syn1neg

        del model  # Save memory
        return VectorCollection(word_vectors, context_vectors)


def calculate_document_representations(pyndri_index, vector_collection, document_ids, *combination_funcs,
                                       vector_func=lambda word, collection: collection.word_vectors[word], **kwargs):
    """
    Pre-compute document vector representations to speed up retrieval later.

    :param: pyndri_index: Pyndri index object
    :type: pyndri_index: pyndri.index
    :param: vector_collection: VectorCollection containing both word and context vectors.
    :type: VectorCollection
    :param: document_ids: IDs of all the documents inside the index.
    :type: set or list or tuple
    :param: combination_funcs: Functions that are used to combine word embeddings.
    :type: tuple of functions
    :param: vector_func: Function used to retrieve an embedding from the vector collection (is useful to determine
    whether word or context vectors are being used.)
    :type: vector_func: func
    """
    representations = {func.__name__: dict() for func in combination_funcs}  # Create result dict

    token2id, id2token, _ = pyndri_index.get_dictionary()
    unkowns = set()  # Set of unknown words for which no embedding exists
    n_documents = len(document_ids)
    cache = dict()  # For some calculations, caching some computations is helpful

    for i, document_id in enumerate(document_ids):
        print("\r{:.2f} % of documents processed.".format((i+1)/n_documents*100), end="", flush=True)

        _, token_ids = pyndri_index.document(document_id)

        token_ids = [token_id for token_id in token_ids if token_id != 0]  # Filter out stop words
        vectors = []
        for token_id in token_ids:
            word = id2token[token_id]
            try:
                vectors.append(vector_func(word, vector_collection))
            except KeyError:
                unkowns.add(word)

        for func in combination_funcs:
            # Create the representations, store them, discard them if the document was empty.
            (representation, _), cache = func(
                vectors, token_ids=token_ids, document_id=document_id, cache=cache, **kwargs
            )
            if representation.shape == (0, ): break
            representations[func.__name__][document_id] = representation

    print("\n{} unknown words encountered.".format(len(unkowns)))

    return representations


def save_document_representations(reps, path):
    """ Store word embeddings using the HDF5 file format. """
    file = h5py.File(path, 'w')

    for name, vector_dict in reps.items():
        vectors = [vector_dict[i] for i in vector_dict.keys()]
        file.create_dataset(name, data=vectors)

    file.close()


def load_document_representations(path):
    """ Load word embeddings using the HDF5 file format. """
    return h5py.File(path, "r")


# Legacy code doing all the heavy computations

# print("Loading index and such...")
# index = pyndri.Index('index/')
# token2id, id2token, id2df = index.get_dictionary()
# num_documents = index.maximum_document() - index.document_base()
# term_frequencies = defaultdict(lambda: Counter())
#
# dictionary = pyndri.extract_dictionary(index)
# document_ids = list(range(index.document_base(), index.maximum_document()))
#
# with open('./ap_88_89/topics_title', 'r') as f_topics:
#     queries = parse_topics([f_topics])
#
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
#
# print('Gathering statistics about', len(query_term_ids), 'terms.')
#
# for int_doc_id in document_ids:
#     ext_doc_id, doc_token_ids = index.document(int_doc_id)
#     term_frequencies[int_doc_id] += Counter([token_id for token_id in doc_token_ids if token_id in query_term_ids])
#
#
# print("Reading word embeddings...")
#
# vectors = VectorCollection.load_vectors("./w2v_60")
# different representations:
# 1. Coordinate-wise min
# 2. Coordinate-wise max
# 3. Centroid
# 4. Sum
# 5. K-means filtering
# 6. tf-idf scaling
# 7. K-means + tfidf
# 8. Circular convolution

# W_in representations
# 1. - 4.
# print("Precomputing vector document representations 1. - 4. with W_in vectors...")
# precomputed_document_representations_win_1_4 = calculate_document_representations(
#     index, vectors, document_ids, doc_min, doc_max, doc_centroid, doc_sum
# )
# save_document_representations(precomputed_document_representations_win_1_4, "./win_representations_1_4")
# del precomputed_document_representations_win_1_4

# 5. - 8.
# print("Precomputing vector document representations 5. - 7. with W_in vectors...")
# precomputed_document_representations_win_5_7 = calculate_document_representations(
#     index, vectors, document_ids, doc_kmeans, doc_tfidf_scaling, #doc_kmeans_tfidf, #doc_circular_conv,
#     document_term_freqs=term_frequencies, id2df=id2df, number_of_documents=num_documents
# )
# save_document_representations(precomputed_document_representations_win_5_7, "./win_representations_5_7")
# del precomputed_document_representations_win_5_7


# W_out representations
# 1. - 4.
# print("Precomputing vector document representations 1. - 4. with W_out vectors...")
# precomputed_document_representations_wout_1_4 = calculate_document_representations(
#     index, vectors, document_ids, doc_min, doc_max, doc_centroid, doc_sum,
#     vector_func=lambda word, collection: collection.context_vectors[word]
# )
# save_document_representations(precomputed_document_representations_wout_1_4, "./wout_representations_1_4.pkl")
# del precomputed_document_representations_wout_1_4
#
# # 5. - 7.
# print("Precomputing vector document representations 5. - 7. with W_out vectors...")
# precomputed_document_representations_wout_5_7 = calculate_document_representations(
#     index, vectors, document_ids, doc_kmeans, doc_tfidf_scaling, doc_kmeans_tfidf, # doc_circular_conv,
#     vector_func=lambda word, collection: collection.context_vectors[word],
#     document_term_freqs=term_frequencies, id2df=id2df, number_of_documents=num_documents
# )
# save_document_representations(precomputed_document_representations_wout_5_7, "./wout_representations_5_7")
# del precomputed_document_representations_wout_5_7

