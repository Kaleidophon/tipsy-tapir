import matplotlib.pyplot as plt

def plot(x_values, y_values, x_label, y_label, title):
    plt.plot(x_values, y_values, 'o')
    plt.axis([0, max(x_values), 0, max(y_values) + 0.01])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def plot_ndcg_for_model(filename_prefix, filename_suffix, parameters, paramterer_name, plot_title):
    ndcg_values = []
    for param in parameters:
        with open("./lexical_results/{}_results_{}_{}.txt".format(filename_prefix, str(param).replace(".", "_"), filename_suffix), "r") as f:
            for line in f.readlines():
                value_name, _, value = line.split()
                if value_name == "ndcg_cut_10":
                    ndcg_values.append(float(value))
    plot(parameters, ndcg_values, paramterer_name, "NDCG@10", plot_title)
    return ndcg_values

def get_relevant_values_for_model(filename, requested_values):
    values = {}

    with open("./lexical_results/{}".format(filename), "r") as f:
        for line in f.readlines():
            value_name, _, value = line.split()
            if value_name in requested_values:
                values[value_name] = value

    return values

def print_model_values(model_values):
    value_to_name = {
            "ndcg_cut_1000" : "NDCG@1000",
            "map_cut_1000" : "MAP@1000",
            "P_5" : "Presicion@5",
            "recall_1000" : "Recall@1000"
            }

    for key, value in model_values.items():
        print(value_to_name[key], value)

### Models with no varying parameters ###
relevant_values = set(["ndcg_cut_1000", "map_cut_1000", "P_5", "recall_1000"])

# TF-IDF
print("\nTF-IDF values")
tf_idf_values = get_relevant_values_for_model("tfidf_results_validation.txt", relevant_values)
print_model_values(tf_idf_values)

# BM25
print("\nBM25 values")
bm25_values = get_relevant_values_for_model("BM25_results_validation.txt", relevant_values)
print_model_values(bm25_values)

### Models with varying paramters ###

# Jelinek-Mercer
jel_params = [0.1, 0.3, 0.5, 0.7, 0.9]
title = "Jelinek-Mercer values of NDCG@10 for different values of lambda"
jel_ndcg_values = plot_ndcg_for_model("jelinek_mercer", "validationset", jel_params, "lambda", title)
print("JM values:")
print(jel_ndcg_values)

# dirichlet prior
dir_params = [500, 1000, 1500]
title = "Dirichlet prior values of NDCG@10 for different values of mu"
dir_ndcg_values = plot_ndcg_for_model("dirichlet_mu", "validationset", dir_params, "mu", title)
print("Dirichlet values:")
print(dir_ndcg_values)

# Absolute discounting
title = "Absolute dicounting values of NDCG@10 for different values of delta"
abs_ndcg_values = plot_ndcg_for_model("abs_disc_delta", "validationset", jel_params, "delta", title)
print("Absolute disc values:")
print(abs_ndcg_values)

# PLM
