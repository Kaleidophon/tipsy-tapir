import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset

class LTRDataSet(Dataset):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

def create_input_and_output_matrix(features, training_set, test_set):
    input_matrix = []
    output_vector = []

    test_features = []

    # We only iterate over the query-document pairs we actually
    # have features for
    for query_id, documents in features.items():
        for document_id, feature_vector in documents.items():
            # Since we are using cross-validation,
            # we risk running into query-document pairs that are
            # not in the training set, but held out
            # in the test_set
            if document_id in training_set[query_id]:
                input_matrix.append(feature_vector)
                output_vector.append(training_set[query_id][document_id])

            if document_id in test_set[query_id]:
                test_features.append((query_id, document_id, feature_vector))


    return input_matrix, output_vector, test_features

def cross_validation_set(filepath, k, i):
    """
    A function that reads in all the data in the given
    dataset, partitions it into k parts, and returns
    the i-th partition as the test set, and the
    remaining k-1 parts as training data.

    :param filepath: string containing the filepath to the dataset
    :param k: the desired number of partitions
    :param i: the location of which partition to use as a test set

    :returns training_set, test_set: Two dictionarys of the form dict[query_id][document_id] = relevance_of_document_to_query
    """
    assert k > i, "The index of the desired test set partition cannot be greater than the number of partitions"

    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            query_id, _, external_document_id, relevance = line.split()
            data.append((query_id, external_document_id, relevance))

    # Partition the data into k-parts
    partitions = []
    partition_lengths = len(data) // k

    for j in range(0, len(data), partition_lengths):
        splice = j + partition_lengths
        if splice > len(data):
            # The data doesn't split evenly, so we simply
            # discard the last few data points. In our case
            # this is 6 out of 49536 points.
            continue

        partition = data[j:splice]
        partitions.append(partition)

    held_out_partition = partitions.pop(i)

    training_set = defaultdict(dict)
    test_set = defaultdict(dict)

    for partition in partitions:
        for point in partition:
            query_id, document_id, relevance = point
            training_set[int(query_id)][document_id] = int(relevance)

    for point in held_out_partition:
        query_id, document_id, relevance = point
        test_set[int(query_id)][document_id] = int(relevance)

    return training_set, test_set

def get_dataset_for_features(features, k=10, i=0):
    data_filepath = "./ap_88_89/qrel_test"

    training_data, test_data = cross_validation_set(data_filepath, k, i)
    # We can now use different values of i to use different portions of the data as test data

    inputs, outputs, test_features = create_input_and_output_matrix(features, training_data, test_data)
    return LTRDataSet(inputs, outputs), test_features

print()
