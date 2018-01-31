import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import collections

np.random.seed(12345678) # fix random seed to get same numbers

relevant_values = set(["ndcg_cut_10", "map_cut_1000", "P_5", "recall_1000"])
lang_models = set(['LM_jelinek_mercer_smoothing','LM_dirichelt_smoothing','LM_absolute_discounting', 'PLM'])
value_to_name = {
        "ndcg_cut_10" : "NDCG@10",
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

        with open("./lexical_results/{}".format(filename), "r") as f:
            for line in f.readlines():
                value_name, query_id, value = line.split()
                if value_name in relevant_values:
                    if query_id == 'all':
                        self.relevant_values[value_name] = value
                    else:
                        self.all_values[value_name].append(float(value))

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
                        value_name, query_id, value = line.split()
                        if value_name == "ndcg_cut_10" and query_id == 'all':
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
            t, p_value = stats.ttest_rel(all_values_base[measure], all_values_experiment[measure])
            if p_value <= (alpha/m):
                print('Null hypothesis rejected for {0}'.format(measure))
            else:
                print('Null hypothesis accepted for {0}'.format(measure))
        else:
            print('Error: m not equal')

# TF-IDF
print("\nTF-IDF values")
tfidf = Evaluation('tfidf')
tfidf.get_results("tfidf_results.txt")

# BM25
print("\nBM25 values")
bm25 = Evaluation('bm25')
bm25.get_results("bm25_results.txt")

### Models with varying paramters ###

# Jelinek-Mercer
jel_params = [0.1, 0.3, 0.5, 0.7, 0.9]
jm = Evaluation('LM_jelinek_mercer_smoothing')
jel_ndcg_values = jm.plot_ndcg(jel_params, "lambda")
print("\nJM values:")
print(jel_ndcg_values)
max_jm_value = max(jel_ndcg_values)
max_index = None
for i in range(len(jel_ndcg_values)):
    if jel_ndcg_values[i] == max_jm_value:
        max_index = i
        break
print("\nMax JM values")
print('LM_jelinek_mercer_smoothing_{}'.format(str(jel_params[max_index]).replace(".", "_")))
jm_max = Evaluation('LM_jelinek_mercer_smoothing_{}'.format(str(jel_params[max_index]).replace(".", "_")))
jm_max.get_results('LM_jelinek_mercer_smoothing_results_{}.txt'.format(str(jel_params[max_index]).replace(".", "_")))

# Dirichlet Prior
dirich_params = [500, 1000, 1500]
dirich = Evaluation('LM_dirichelt_smoothing')
dir_ndcg_values = dirich.plot_ndcg(dirich_params, "mu")
print("\nDirichlet values:")
print(dir_ndcg_values)
max_dirich_value = max(dir_ndcg_values)
max_index = None
for i in range(len(dir_ndcg_values)):
    if dir_ndcg_values[i] == max_dirich_value:
        max_index = i
        break
print("\nMax Dirichlet Prior values")
print('LM_dirichelt_smoothing_{}'.format(str(dirich_params[max_index]).replace(".", "_")))
dirich_max = Evaluation('LM_dirichelt_smoothing_{}'.format(str(dirich_params[max_index]).replace(".", "_")))
dirich_max.get_results('LM_dirichelt_smoothing_results_{}.txt'.format(str(dirich_params[max_index]).replace(".", "_")))

# Absolute discounting
ab_dis = Evaluation('LM_absolute_discounting')
abs_ndcg_values = ab_dis.plot_ndcg(jel_params, "delta")
print("\nAbsolute disc values:")
print(abs_ndcg_values)
max_abs_value = max(abs_ndcg_values)
max_index = None
for i in range(len(abs_ndcg_values)):
    if abs_ndcg_values[i] == max_abs_value:
        max_index = i
        break
print("\nMax Absolute discounting values")
print('LM_absolute_discounting_{}'.format(str(jel_params[max_index]).replace(".", "_")))
abs_max = Evaluation('LM_absolute_discounting_{}'.format(str(jel_params[max_index]).replace(".", "_")))
abs_max.get_results('LM_absolute_discounting_results_{}.txt'.format(str(jel_params[max_index]).replace(".", "_")))

# PLM
plm_params = ['circle','cosine','triangle','gaussian','passage']
plm = Evaluation('PLM')
plm_ndcg_values = plm.plot_ndcg(plm_params, "delta")
print("\nPLM values:")
print(plm_ndcg_values)
max_plm_value = max(plm_ndcg_values)
max_index = None
for i in range(len(plm_ndcg_values)):
    if plm_ndcg_values[i] == max_plm_value:
        max_index = i
        break
print("\nMax PLM values")
print('PLM_{}'.format(str(plm_params[max_index]).replace(".", "_")))
plm_max = Evaluation('PLM_{}'.format(str(plm_params[max_index]).replace(".", "_")))
plm_max.get_results('PLM_results_{}.txt'.format(str(plm_params[max_index]).replace(".", "_")))


# Significance Testing TF-IDF vs all LMs
print("\nTF-IDF vs. BM25")
significance_testing(tfidf.all_values, bm25.all_values)
print("\nTF-IDF vs. LM-Jelinek-Mercer")
significance_testing(tfidf.all_values, jm_max.all_values)
print("\nTF-IDF vs. LM-Dirichelt")
significance_testing(tfidf.all_values, dirich_max.all_values)
print("\nTF-IDF vs. LM-Absolute-Discounting")
significance_testing(tfidf.all_values, abs_max.all_values)
print("\nTF-IDF vs. PLM")
significance_testing(tfidf.all_values, plm_max.all_values)

# LSM
print("\nLSI values")
lsi = Evaluation('LSI')
lsi.get_results("LSI_results.txt")

print("\nLDA values")
lda = Evaluation('LDA')
lda.get_results("LDA_results.txt")

# Significance Testing TF-IDF vs all LSMs
print("\nTF-IDF vs. LSI")
significance_testing(tfidf.all_values, lsi.all_values)
print("\nTF-IDF vs. LDA")
significance_testing(tfidf.all_values, lda.all_values)
