import numpy as np
from collections import defaultdict
from extract_features import extract_feature_vectors

# The data we recieve will be of the form
# data[query_id][document_id] = features

def create_input_and_output_matrix(features, training_data):
    inputs = []
    outputs = []

    for query_id, documents in training_data.items():
        for document_id, relevance in documents.items():
            # Add the feature to the inputs matrix
            inputs.append(features[int(query_id)][document_id])
            # Add the relevance label to the coresponding ouput
            outputs.append(relevance)

    return inputs, outputs

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

data_filepath = "./ap_88_89/qrel_test"

training_data, test_data = cross_validation_set(data_filepath, 10, 5)
# We can now use different values of i to use different portions of the data as test data

features = extract_feature_vectors()
inputs, ouputs = create_input_and_output_matrix(features, training_data)

print("Length", len(inputs))
for i in range(len(inputs)):
    print(inputs[i])
