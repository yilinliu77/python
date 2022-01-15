#!/bin/bash -e

str1="qiao"
str2="xuexiao"

for file in /d/Projects/Reconstructability/training_data/v2/*
do
    if test -f $file
    then
          echo Jump $file
    else
        if [[ ($file =~ $str1 || $file =~ $str2) && !($file =~ "qx") ]]
        then
#            echo $file
##            python src/regress_reconstructability_hyper_parameters/preprocess_view_features.py $file
             IFS='/' read -ra ADDR <<< $file
#             echo ${ADDR[-1]}
             scp -P 12222 -r "/d/Projects/Reconstructability/training_data/v2/"${ADDR[-1]}"/training_data" "root@172.31.224.143:/mnt/d/reconstructability/v2/"${ADDR[-1]}"/"

        fi
    fi
done
