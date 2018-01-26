# Run tfidf evaulation
~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test tfidf_official.run | grep "\sall\s" > tfidf_results.txt

# BM25
# ~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test BM25.run | grep "\sall\s" > BM25_results.txt

# Jelinek-Mercer
~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test lm_jel_official_lambda_0.1.run | grep "\sall\s" > JM_0.1.txt
~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test lm_jel_official_lambda_0.3.run | grep "\sall\s" > JM_0.3.txt
~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test lm_jel_official_lambda_0.5.run | grep "\sall\s" > JM_0.5.txt
~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test lm_jel_official_lambda_0.7.run | grep "\sall\s" > JM_0.7.txt
~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test lm_jel_official_lambda_0.9.run | grep "\sall\s" > JM_0.9.txt

# Absolute discounting
# ~/Downloads/trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_test :!.run > tfidf_results.txt

