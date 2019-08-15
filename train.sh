#!/bin/bash

echo "Running baseline"
for seed in 1 2 3 4 5 
  do
    python main.py --seed $seed --resume
  done