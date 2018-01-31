# relative_trec_eval_path=~/Downloads/trec_eval/trec_eval
relative_trec_eval_path=~/Programming/IR/trec_eval/trec_eval

run_single_evaluation () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1".run > ./lexical_results/"$1"_results.txt
}

run_parameters_between_1 () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_1.run  > ./lexical_results/"$1"_results_0_1.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_3.run  > ./lexical_results/"$1"_results_0_3.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_5.run  > ./lexical_results/"$1"_results_0_5.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_7.run  > ./lexical_results/"$1"_results_0_7.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_9.run  > ./lexical_results/"$1"_results_0_9.txt
}

run_dirichlet () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_500.run  > ./lexical_results/"$1"_results_500.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_1000.run > ./lexical_results/"$1"_results_1000.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_1500.run > ./lexical_results/"$1"_results_1500.txt
}

# Run tfidf evaluation
run_single_evaluation "tfidf"

# BM25
run_single_evaluation "bm25"

# Jelinek-Mercer
run_parameters_between_1 "LM_jelinek_mercer_smoothing"

# Dirichlet
run_dirichlet "LM_dirichelt_smoothing"

# Absolute discounting
run_parameters_between_1 "LM_absolute_discounting"

# PLM
run_single_evaluation "PLM_passage"

run_single_evaluation "PLM_gaussian"

run_single_evaluation "PLM_triangle"

run_single_evaluation "PLM_cosine"

run_single_evaluation "PLM_circle"

# LSM
run_single_evaluation "LSI"

run_single_evaluation "LDA"
