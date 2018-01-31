from collections import defaultdict

def fill_features_from_run_file(filepath, features):
    with open(filepath, "r") as f:
        for line in f.readlines():
            query_id, _, document_id, ranking, score, class_name = line.split()
            # Append feature values to the features lookup table directly
            features[int(query_id)][document_id].append(float(score))


def extract_feature_vectors():
    """
    ### Features already included:

    Tf-idf score
    BM25 score
    Language model scores

    ### Desired features:

    Latent Semantic model score
    Cosine similarity between vector representations
    Document length
    Query length
    Percentage of query words in document
    Percentage of stop words in document
    Average frequency of query terms in document
    """

    features = defaultdict(lambda: defaultdict(list))

    # tf-idf
    fill_features_from_run_file("./lexical_results/tfidf.run", features)

    # BM25
    fill_features_from_run_file("./lexical_results/bm25.run", features)

    # Jelinek-Mercer
    # TODO: Find which parameter is the best
    fill_features_from_run_file("./lexical_results/LM_jelinek_mercer_smoothing_0_1.run", features)

    # Drichlet prior
    # TODO: Find which parameter is the best
    fill_features_from_run_file("./lexical_results/LM_dirichelt_smoothing_500.run", features)

    # Absolute discounting
    # TODO: Find which parameter is the best
    fill_features_from_run_file("./lexical_results/LM_absolute_discounting_0_1.run", features)

    return features
