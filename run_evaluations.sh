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
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0.1.run | grep "\sall\s" > ./lexical_results/"$2"_0.1_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0.3.run | grep "\sall\s" > ./lexical_results/"$2"_0.3_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0.5.run | grep "\sall\s" > ./lexical_results/"$2"_0.5_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0.7.run | grep "\sall\s" > ./lexical_results/"$2"_0.7_testset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$1"_0.9.run | grep "\sall\s" > ./lexical_results/"$2"_0.9_testset.txt

  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0.1.run | grep "\sall\s" > ./lexical_results/"$2"_0.1_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0.3.run | grep "\sall\s" > ./lexical_results/"$2"_0.3_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0.5.run | grep "\sall\s" > ./lexical_results/"$2"_0.5_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0.7.run | grep "\sall\s" > ./lexical_results/"$2"_0.7_validationset.txt
  $relative_trec_eval_path -m all_trec -q ap_88_89/qrel_validation ./lexical_results/"$1"_0.9.run | grep "\sall\s" > ./lexical_results/"$2"_0.9_validationset.txt
}

run_dirichlet () {
  # Old version: ~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test ./lexical_results/"$filename"_500.run | grep "\sall\s" > ./lexical_results/"$output_filename"_500_testset.txt
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
