relative_trec_eval_path=~/Desktop/ir/trec_eval/trec_eval

run_single_evaluation () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1".run | grep "\sall\s" > ./lexical_results/"$1"_results.txt
}

run_parameters_between_1 () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_1.run | grep "\sall\s" > ./lexical_results/"$1"_results_0_1.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_3.run | grep "\sall\s" > ./lexical_results/"$1"_results_0_3.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_5.run | grep "\sall\s" > ./lexical_results/"$1"_results_0_5.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_7.run | grep "\sall\s" > ./lexical_results/"$1"_results_0_7.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_9.run | grep "\sall\s" > ./lexical_results/"$1"_results_0_9.txt
}

run_dirichlet () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_500.run | grep "\sall\s" > ./lexical_results/"$1"_results_500.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_1000.run | grep "\sall\s" > ./lexical_results/"$1"_results_1000.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_1500.run | grep "\sall\s" > ./lexical_results/"$1"_results_1500.txt
}

# Run tfidf evaluation
#run_single_evaluation "tfidf"

# BM25
#run_single_evaluation "BM25"

# Jelinek-Mercer
#run_parameters_between_1 "jelinek_mercer"

# Dirichlet
#run_dirichlet "dirichlet_mu"

# Absolute discounting
#run_parameters_between_1 "abs_disc_delta"

# PLM
#run_single_evaluation "PLM_passage"

#run_single_evaluation "PLM_gaussian"

#run_single_evaluation "PLM_triangle"

#run_single_evaluation "PLM_cosine"

#run_single_evaluation "PLM_circle"

# Word embedding
run_single_evaluation "embeddings_Savg"
run_single_evaluation "embeddings_So"
run_single_evaluation "embeddings_So_kmeans_win"
run_single_evaluation "embeddings_So_tfidf_win"
run_single_evaluation "embeddings_So_centroid_win_wout"
run_single_evaluation "embeddings_So_centroid_wout_win"
run_single_evaluation "embeddings_So_centroid_wout_wout"
