import matplotlib.pyplot as plt

def plot(x_values, y_values, x_label, y_label):
    plt.plot(x_values, y_values, 'o')
    plt.axis([0, max(x_values), 0, max(y_values) + 0.01])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_ndcg_for_model(filename_prefix, parameters, paramterer_name):
    ndcg_values = []
    for param in parameters:
        with open("{}_{}.txt".format(filename_prefix, param), "r") as f:
            for line in f.readlines():
                value_name, _, value = line.split()
                if value_name == "ndcg_cut_10":
                    ndcg_values.append(float(value))
    plot(parameters, ndcg_values, paramterer_name, "NDCG")

def get_relevant_values_for_model(filename, requested_values):
    values = {}

    with open(filename, "r") as f:
        for line in f.readlines():
            value_name, _, value = line.split()
            if value_name in requested_values:
                values[value_name] = value

    return values

### Models with no varying parameters ###
relevant_values = set(["ndcg_cut_1000", "map_cut_1000", "P_5", "recall_1000"])

# TF-IDF
tf_idf_values = get_relevant_values_for_model("tfidf_results.txt", relevant_values)
# View the results
for key, value in tf_idf_values.items():
    print(key, value)

# BM25



### Models with varying paramters ###

# Jelinek-Mercer

# Test that it works and plot the values
jel_params = [0.1, 0.3, 0.5, 0.7, 0.9]
plot_ndcg_for_model("JM", jel_params, "lambda")

# Dirichlet prior

# Absolute discounting

# PLM
