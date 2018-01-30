relative_trec_eval_path=~/Downloads/trec_eval/trec_eval

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

# Run tfidf evaulation
filename="tfidf"
run_single_evaluation $filename

# BM25
filename="BM25"
run_single_evaluation $filename

# Jelinek-Mercer
filename="jelinek_mercer"
run_parameters_between_1 $filename

# Dirichlet
filename="dirichlet_mu"
run_dirichlet $filename

# Absolute discounting
filename="abs_disc_delta"
run_parameters_between_1 $filename

# PLM
filename="PLM_passage"
run_single_evaluation $filename

filename="PLM_gaussian"
run_single_evaluation $filename

filename="PLM_triangle"
run_single_evaluation $filename

filename="PLM_cosine"
run_single_evaluation $filename

filename="PLM_circle"
run_single_evaluation $filename
