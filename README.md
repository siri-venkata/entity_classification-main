
# WETC

Dataset preparation code is present in the dataprep folder. process.sh script handles the dataset generation

clean_json.json is the taxonomy file
code2query contains what query needs to be run for each type queries are present in queries folder. You may have to place the files in the right locations.

main.py file runs training and inference both. All necessary arguments are documented with argparser.
 It logs the predictions in a csv format. You can use the eval.py file to generate the reported metrics.
