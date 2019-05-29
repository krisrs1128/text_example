#!/usr/bin/env bash

Rscript data_prep/gutenberg.R
mv sentences.csv /data/
python3 exper.py
