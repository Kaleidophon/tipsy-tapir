import h5py
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import Counter

from homework2 import query_term_id, cosine_similarity, index, document_ids


def idf(term_id, id2df, number_of_documents):
    df_t = id2df[term_id]
    return np.log(number_of_documents) - np.log(df_t)


def tf_idf(term_id, document_term_freq, id2df, number_of_documents):
    return np.log(1 + document_term_freq) * idf(term_id, id2df, number_of_documents)


def doc_centroid(vectors, **kwargs):
    return np.add.reduce(vectors) / len(vectors) if len(vectors) > 0 else np.array([])


def doc_sum(vectors, **kwargs):
    return np.add.reduce(vectors) if len(vectors) > 0 else np.array([])


def doc_min(vectors, **kwargs):
    return np.minimum.reduce(vectors) if len(vectors) > 0 else np.array([])


def doc_max(vectors, **kwargs):
    return np.maximum.reduce(vectors) if len(vectors) > 0 else np.array([])


def _doc_kmeans(vectors, k=None, **kwargs):
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
    biggest_cluster = most_common(labels)

    return [vector for vector, label in zip(vectors, labels) if label == biggest_cluster]


def doc_kmeans(vectors, k=None):
    return doc_centroid(_doc_kmeans(vectors, k))


def doc_tfidf_scaling(vectors, token_ids, document_term_freq, id2df, number_of_documents, **kwargs):
    if len(vectors) == 0:
        return np.array([])

    for i, (vector, token_id) in enumerate(zip(vectors, token_ids)):
        term_tf_idf = tf_idf(token_id, document_term_freq, id2df, number_of_documents)
        vectors[i] = vector * term_tf_idf

    return doc_centroid(vectors)


def doc_circular_conv(vectors, **kwargs):
    def circular_conf(vec1, vec2):
        return [
            sum([vec1[k] * vec2[i - k] for k in range(len(vec1))])
            for i in range(len(vec1))
        ]

    if len(vectors) == 0:
        return np.array([])

    if len(vectors) == 1:
        return vectors[0]

    start_vector, to_convolve = vectors[0], vectors[1:]
    for vector in to_convolve:
        start_vector = circular_conf(start_vector, vector)

    return start_vector / len(vectors)



def doc_kmeans_tfidf(vectors, token_ids, document_term_freq, id2df, number_of_documents, k=None, **kwargs):
    # Dirty but fast heuristic
    filtered_vectors = _doc_kmeans(vectors, k)
    return doc_tfidf_scaling(filtered_vectors, token_ids, document_term_freq, id2df, number_of_documents)


class VectorCollection:

    def __init__(self, word_vectors, context_vectors):
        self.word_vectors = {
            word: word_vectors.syn0[vocab_item.index] for word, vocab_item in word_vectors.vocab.items()
        }
        self.context_vectors = {word_vectors.index2word[i]: vec for i, vec in enumerate(context_vectors)}

    @staticmethod
    def load_vectors(path, to_train=False):
        model = Word2Vec.load(path)

        if to_train:
            return model

        # In case it doesn't need to be trained, delete train code to free up ram
        word_vectors = model.wv

        # TODO: Test this code
        context_vectors = dict()
        if hasattr(model, "syn1"):
            # For hierarchical softmax
            context_vectors = model.syn1
        elif hasattr(model, "syn1neg"):
            # For negative sampling
            context_vectors = model.syn1neg

        del model
        return VectorCollection(word_vectors, context_vectors)


def calculate_document_representations(pyndri_index, vector_collection, document_ids, *combination_funcs,
                                       vector_func=lambda word, collection: collection.word_vectors[word]):
    representations = {func.__name__: dict() for func in combination_funcs}

    token2id, id2token, _ = pyndri_index.get_dictionary()
    unkowns = set()
    n_documents = len(document_ids)

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
        #vectors = [vector_func(id2token[token_id], vector_collection) for token_id in token_ids]

        for func in combination_funcs:
            representation = func(vectors)
            if representation.shape == (0, ): break
            representations[func.__name__][document_id] = representation

    print("\n{} unknown words encountered.".format(len(unkowns)))

    return representations


def score_embeddings(query_token_ids, pyndri_index, vector_collection,
                     vector_func_query=lambda word, collection: collection.word_vectors[word],
                     vector_func_doc=lambda word, collection: collection.word_vectors[word],
                     vector_combination_func=lambda vectors: np.sum(vectors), **kwargs):
    """
    Score a query and documents by taking the word embeddings of the words they contain and simply sum them,
    afterwards comparing the summed vectors with cosine similarity.
    """
    # Get query vector
    _, id2token, _ = pyndri_index.get_dictionary()
    # Just sum
    query_vectors = [
        vector_func_query(id2token[query_token_id], vector_collection)
        for query_token_id in query_token_ids if query_term_id != 0
    ]
    query_vector = vector_combination_func(query_vectors)

    # Score documents
    document_scores = []
    for document_id in range(pyndri_index.document_base(), pyndri_index.maximum_document()):
        ext_doc_id, token_ids = pyndri_index.document(document_id)

        document_vectors = [
            vector_func_doc(id2token[token_id], vector_collection) for token_id in token_ids if token_id != 0
        ]
        document_vector = vector_combination_func(document_vectors)
        score = cosine_similarity(query_vector, document_vector)

        document_scores.append((score, ext_doc_id))

    return document_scores


def save_document_representations(reps, path):
    file = h5py.File(path, 'w')

    for name, vector_dict in reps.items():
        vectors = [vector_dict[i] for i in vector_dict.keys()]
        #print(name)
        #for vector in vectors:
        #    print(vector)
        #print(name)
        #print(len(vectors))
        #size0 = len(vectors[0])
        #print(size0)
        #print([len(vector) for vector in vectors if len(vector) != size0])
        file.create_dataset(name, data=vectors)

    file.close()


print("Reading word embeddings...")

vectors = VectorCollection.load_vectors("./w2v_60")
precomputed_document_representations_win = calculate_document_representations(
    index, vectors, document_ids, doc_centroid, doc_min, doc_max
)
precomputed_document_representations_wout = calculate_document_representations(
    index, vectors, document_ids, doc_centroid, doc_min, doc_max,
    vector_func=lambda word, collection: collection.context_vectors[word]
)

print("Precomputing vector document representations with W_in vectors...")

save_document_representations(precomputed_document_representations_win, "./win_representations.pkl")
del precomputed_document_representations_win

print("Precomputing vector document representations with W_out vectors...")

save_document_representations(precomputed_document_representations_wout, "./wout_representations.pkl")
del precomputed_document_representations_wout
