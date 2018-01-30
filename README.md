# Retrieval models

Homework assignment 2 for Universiteit van Amsterdam's Information Retrieval lecture during the winter term 2017/2018 for Stian Steinbakken, Dennis Ulmer and Santhosh Kumar Rajamanickam.

## Instructions

The program can be executed by running the `run_retrieval` function with the desired model. An example would be to alter `homework2.py`
to include: `run_retrieval('PLM', None, document_ids=document_ids, query_word_positions=query_word_positions)` and run the file by typing
`python3 homework2.py`. This will produce a .run file of the model run previously.

After the .run file has been created, one can evaluate the produced run file by running (for instance):
```  
$relative_trec_eval_path -m all_trec -q ap_88_89/qrel_test ./path_to/the_run_file_produced.run | grep "\sall\s" > output_results.txt
```
Where relative_trec_eval_path is your path to wherever you have installed trec_eval. This will produce `ouput_results.txt`
which now contains the results for all the the queries in the test set.

Alternatively, if all the run files are availalbe, one can run:
```
chmod +x run_evaluations.sh
./run_evaluations.sh
```

in order to produce all the result files at once. It is also possible to look into the last mentioned file to see more
commands one can run to evaluate.
