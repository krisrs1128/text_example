#!/usr/bin/env bash

Rscript gutenberg.R
mv sentences.csv /data/
python3 exper.py
