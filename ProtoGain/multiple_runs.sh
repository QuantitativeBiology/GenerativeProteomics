#!/bin/bash

path="/data/"

projects=('PXD017052.2' 'MSV000085836')
trial_name="missforest_without_mnar"
output_path_base="/data/"


for project in "${projects[@]}";
do
    output_path="$output_path_base/$project/$trial_name/"
    for file in "$path"/$project/*;
    do
        if [[ $file == *absolute.tsv ]]; then
            for run in {1..1}
            do
                echo "PROJECT $project"
                echo "Running run = $run"
                /bin/python3 protogain.py -i "$file" -o "$project"_imputed_"$run" --ofolder "$output_path" --oeval "$project"_imputed_eval_"$run" --it 2001 --miss 0.05 --hint 0.9 --outall 1 --project "$project" --cores 20 --missforest 1 --mnar 1 --sigma 0.10
            done
        fi
    done
done