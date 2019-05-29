#!/usr/bin/env bash

cd /home/
Rscript data_prep/gutenberg.R
python3 exper.py
