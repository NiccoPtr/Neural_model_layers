BLA_IC#!/usr/bin/env bash

for seed in $(seq 0 1 30); do
	echo 'Running simulation with seed=' $seed
	python Test_MGV_MC.py --mode short_save --seed $seed
done
