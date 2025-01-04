#!/bin/bash

path="/data/benchmarks/proteomics/ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/absolute-expression/quantms-data"

# projects=("PXD020192")
# projects=("PXD030304")
# projects=('PXD005445' 'PXD010271' 'PXD017052.2' 'MSV000085836' 'PXD002179' 'PXD016999.2' 'PXD004452' 'PXD016999.1' 'PXD019909.1' 'PXD000865')
# projects=('PXD000561' 'PXD009737.1' 'PXD012131' 'PXD020192' 'PXD010154' 'PXD024364.1' 'PXD000612' 'PXD019909' 'PXD024364.2' 'PXD008840' 'PXD006675' 'PXD030304' 'PXD005736')
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
                /home/leandrosobral/miniconda3/bin/python3 protogain.py -i "$file" -o "$project"_imputed_"$run" --ofolder /home/leandrosobral/ProtoGain/Yasset/$project/MF_test_all/ --oeval "$project"_imputed_eval_"$run" --it 2001 --miss 0.1 --hint 0.9 --outall 1 --project "$project" --cores 20
            done
        fi
    done
done
