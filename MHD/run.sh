#!/bin/bash

jupyter-nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute "notebooks/$1"