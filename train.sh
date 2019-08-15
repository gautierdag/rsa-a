#!/bin/bash

echo "Running baseline"
for seed in {1..10} 
  do
    python main.py --seed $seed --resume
  done

echo "Computing RSAs"
python compute_rsas.py