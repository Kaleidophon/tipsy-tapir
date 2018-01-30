# data[query_id][document_id] = tf-idf_score_for_document

data_filepath = "./ap_88_89/qrel_test"

def cross_validation_set(filepath, k, i):
    """
    A function that reads in all the data in the given
    dataset, partitions it into k parts, and returns
    the i-th partition as the test set, and the
    remaining k-1 parts as training data.

    :param filepath: string containing the filepath to the dataset
    :param k: the desired number of partitions
    :param i: the location of which partition to use as a test set
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

    test_set = partitions.pop(i)

    return partitions, test_set

# Example usage
training_data, test_data = cross_validation_set(data_filepath, 10, 11)
# We can now use different values of i to use different portions of the data as test data

