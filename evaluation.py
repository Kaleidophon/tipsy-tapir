import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import collections

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
        self.all_values = collections.defaultdict(list)

    def get_results(self, filename):

        values = {}

        # TODO: check with Stian ?
        # with open("./lexical_results/{}".format(filename), "r") as f:
        #     for line in f.readlines():
        #         value_name, _, value = line.split()
        #         if value_name in relevant_values:
        #             self.relevant_values[value_name] = value

        with open("./lexical_results/{}".format(filename), "r") as f:
            for line in f.readlines():
                value_name, query_id, value = line.split()
                if value_name in relevant_values:
                    if query_id == 'all':
                        self.relevant_values[value_name] = value
                    else:
                        self.all_values[value_name] = value

        for key, value in self.relevant_values.items():
            print(value_to_name[key], value)

    def plot(self, x_values, y_values, x_label, y_label):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        x_indexes = [i for i in range(len(x_values))]
        plt.bar(x_indexes, y_values)
        plt.xticks(x_indexes, tuple(["{}".format(val) for val in x_values]))
        plt.show()

    def plot_ndcg(self, parameters, paramterer_name):

        if self.model_name in lang_models:
            ndcg_values = []
            for param in parameters:
                with open("./lexical_results/{}_results_{}.txt".format(self.model_name, str(param).replace(".", "_")), "r") as f:
                    for line in f.readlines():
                        value_name, _, value = line.split()
                        if value_name == "ndcg_cut_10":
                            ndcg_values.append(float(value))
            self.plot(parameters, ndcg_values, paramterer_name, "NDCG@10")

            return ndcg_values


def significance_testing(all_values_base, all_values_experiment):

    # Bonferroni correction
    # https://en.wikipedia.org/wiki/Bonferroni_correction
    alpha = 0.05
    m = None

    for measure in relevant_values:
        if len(all_values_base[measure]) == len(all_values_experiment[measure]):
            m = len(all_values_base[measure])
            t, p_value = ttest_rel(all_values_base[measure], all_values_experiment[measure])
            if p_value <= (alpha/m):
                print('Null hypothesis of equal averages rejected for {0}'.format(measure))
            else:
                print('Null hypothesis of equal averages accepted for {0}'.format(measure))
        else:
            print('Error: m not equal')

# TF-IDF
print("\nTF-IDF values")
tfidf = Evaluation('tfidf')
tfidf.get_results("tfidf_results_validation.txt")

# BM25
print("\nBM25 values")
bm25 = Evaluation('bm25')
bm25.get_results("BM25_results_validation.txt")

### Models with varying paramters ###

# Jelinek-Mercer
jel_params = [0.1, 0.3, 0.5, 0.7, 0.9]
jm = Evaluation('LM_jelinek_mercer_smoothing')
jel_ndcg_values = jm.plot_ndcg(dataset_type, jel_params, "lambda")
print("\nJM values:")
print(jel_ndcg_values)

# Dirichlet Prior
dirich_params = [500, 1000, 1500]
dirich = Evaluation('LM_dirichelt_smoothing')
dir_ndcg_values = dirich.plot_ndcg(dirich_params, "mu")
print("\nDirichlet values:")
print(dir_ndcg_values)

# Absolute discounting
ab_dis = Evaluation('LM_absolute_discounting')
abs_ndcg_values = ab_dis.plot_ndcg(dataset_type, jel_params, "delta")
print("\nAbsolute disc values:")
print(abs_ndcg_values)

# PLM
# TODO: evaluate PLM
