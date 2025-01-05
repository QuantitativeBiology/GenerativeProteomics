#!/bin/bash

path="/data/benchmarks/proteomics/ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/absolute-expression/quantms-data"

projects=("PXD016999.2" "PXD030304")

for project in "${projects[@]}";
do
    for file in "$path"/$project/*;
    do
        if [[ $file == *absolute.tsv ]]; then
            for run in {1..5}
            do
                echo "PROJECT $project"
                echo "Running run = $run"
                /home/leandrosobral/miniconda3/bin/python3 protogain.py -i "$file" -o "$project"_imputed_"$run" --ofolder /home/leandrosobral/ProtoGain/Yasset/$project/proteomics_branch_test/ --oeval "$project"_imputed_eval_"$run" --it 2001 --miss 0.1 --hint 0.9 --outall 1 --project "$project" --cores 20 --missforest 0
            done
        fi
    done
done
