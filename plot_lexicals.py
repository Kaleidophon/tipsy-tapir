import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(12345678) # fix random seed to get same numbers

relevant_values = set(["ndcg_cut_10", "map_cut_1000", "P_5", "recall_1000"])
lang_models = set(['LM_jelinek_mercer_smoothing','LM_dirichelt_smoothing','LM_absolute_discounting', 'PLM_passage','PLM_gaussian','PLM_triangle', 'PLM_cosine','PLM_circle'])
value_to_name = {
        "ndcg_cut_10" : "NDCG@1000",
        "map_cut_1000" : "MAP@1000",
        "P_5" : "Presicion@5",
        "recall_1000" : "Recall@1000"
        }

class Evaluation():

    def __init__(self, model_name):
        self.relevant_values = {}
        self.model_name = model_name

    def get_results(self, filename):

        values = {}

        with open("./lexical_results/{}".format(filename), "r") as f:
            for line in f.readlines():
                value_name, _, value = line.split()
                if value_name in relevant_values:
                    self.relevant_values[value_name] = value

        for key, value in self.relevant_values.items():
            print(value_to_name[key], value)

    def plot(x_values, y_values, x_label, y_label):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        x_indexes = [i for i in range(len(x_values))]
        plt.bar(x_indexes, y_values)
        plt.xticks(x_indexes, tuple(["{}".format(val) for val in x_values]))
        plt.show()

    def plot_ndcg(dataset_type, parameters, paramterer_name):

        if self.model_name in lang_models:
            ndcg_values = []
            for param in parameters:
                with open("./lexical_results/{}_results_{}_{}.txt".format(model_name, str(param).replace(".", "_"), eval_type), "r") as f:
                    for line in f.readlines():
                        value_name, _, value = line.split()
                        if value_name == "ndcg_cut_10":
                            ndcg_values.append(float(value))
            plot(parameters, ndcg_values, paramterer_name, "NDCG@10")

            return ndcg_values


    def significance_testing(base_scores, experimental_scores):
        stats.ttest_rel(eval1,rvs2)
        pass

# TF-IDF
print("\nTF-IDF values")
tfidf = Evaluation('tfidf')
tfidf.get_results("tfidf_results_validation.txt")

# BM25
print("\nBM25 values")
bm25 = Evaluation('bm25')
bm25.get_results("BM25_results_validation.txt")

### Models with varying paramters ###
dataset_type = "validationset"

# Jelinek-Mercer
jel_params = [0.1, 0.3, 0.5, 0.7, 0.9]
jm = Evaluation('LM_jelinek_mercer_smoothing')
jel_ndcg_values = jm.plot_ndcg(dataset_type, jel_params, "lambda")
print("\nJM values:")
print(jel_ndcg_values)

# Dirichlet Prior
dirich_params = [500, 1000, 1500]
dirich = Evaluation('LM_dirichelt_smoothing')
dir_ndcg_values = dirich.plot_ndcg(dataset_type, dirich_params, "mu")
print("\nDirichlet values:")
print(dir_ndcg_values)

# Absolute discounting
ab_dis = Evaluation('LM_absolute_discounting')
abs_ndcg_values = ab_dis.plot_ndcg(dataset_type, jel_params, "delta")
print("\nAbsolute disc values:")
print(abs_ndcg_values)

# PLM
# TODO: evaluate PLM
