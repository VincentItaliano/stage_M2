#!/bin/bash

# run notebook given in argument. ex : sh run.sh MHD.ipynb
jupyter-nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute "notebooks/$1"