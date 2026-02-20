#!/usr/bin/env bash

for delta in $(seq 20 7 420); do
	echo 'Running simulation with delta=' $delta
	python Test_BLA_IC.py --mode short_save --delta $delta
done
