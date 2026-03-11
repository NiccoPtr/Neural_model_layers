#!/usr/bin/env bash

for seed in $(seq 0 1 5); do
	echo 'Running simulation with seed=' $seed
	python Model_Simulation.py -m save -tr 200 -t 1000 -s $seed
done
