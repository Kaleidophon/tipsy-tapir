relative_trec_eval_path=~/Downloads/trec_eval/trec_eval

# Run tfidf evaulation
filename="tfidf"
output_filename="$filename"_results
$relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$filename".run | grep "\sall\s" > ./lexical_results/"$output_filename"_testset.txt
$relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$filename".run | grep "\sall\s" > ./lexical_results/"$output_filename"_validation.txt

# BM25
filename="BM25"
output_filename="$filename"_results
$relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$filename".run | grep "\sall\s" > ./lexical_results/"$output_filename"_testset.txt
$relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$filename".run | grep "\sall\s" > ./lexical_results/"$output_filename"_validation.txt

run_parameters_between_1 () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_1.run | grep "\sall\s" > ./lexical_results/"$2"_0_1_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_3.run | grep "\sall\s" > ./lexical_results/"$2"_0_3_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_5.run | grep "\sall\s" > ./lexical_results/"$2"_0_5_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_7.run | grep "\sall\s" > ./lexical_results/"$2"_0_7_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0_9.run | grep "\sall\s" > ./lexical_results/"$2"_0_9_testset.txt

  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0_1.run | grep "\sall\s" > ./lexical_results/"$2"_0_1_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0_3.run | grep "\sall\s" > ./lexical_results/"$2"_0_3_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0_5.run | grep "\sall\s" > ./lexical_results/"$2"_0_5_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0_7.run | grep "\sall\s" > ./lexical_results/"$2"_0_7_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0_9.run | grep "\sall\s" > ./lexical_results/"$2"_0_9_validationset.txt
}

run_dirichlet () {
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_500.run | grep "\sall\s" > ./lexical_results/"$2"_500_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_1000.run | grep "\sall\s" > ./lexical_results/"$2"_1000_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_1500.run | grep "\sall\s" > ./lexical_results/"$2"_1500_testset.txt

  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_500.run | grep "\sall\s" > ./lexical_results/"$2"_500_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_1000.run | grep "\sall\s" > ./lexical_results/"$2"_1000_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_1500.run | grep "\sall\s" > ./lexical_results/"$2"_1500_validationset.txt
}


# # Jelinek-Mercer
filename="jelinek_mercer"
output_filename="$filename"_results

run_parameters_between_1 $filename $output_filename

# Dirichlet
filename="dirichlet_mu"
output_filename="$filename"_results

run_dirichlet $filename $output_filename

# Absolute discounting
filename="abs_disc_delta"
output_filename="$filename"_results

run_parameters_between_1 $filename $output_filename
