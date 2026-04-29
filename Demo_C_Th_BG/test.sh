#!/usr/bin/env bash

export PYTHONPATH=~/Desktop/CNR_Model
export PATH=$PATH:$PYTHONPATH

for seed in $(seq 2 1 9); do
	python $PYTHONPATH/Test_Cortex_Th_BG.py -s $seed 
	python $PYTHONPATH/Test_Cortex_Th_BG.py --W_C 1.0 -s $seed 
	python $PYTHONPATH/Test_Cortex_Th_BG.py --W_C 1.0 --inp_BLA 0.7 0.0 -s $seed  
	python $PYTHONPATH/Test_Cortex_Th_BG.py --W_C 1.0 --inp 1.0 1.0 -s $seed 
	python $PYTHONPATH/Test_Cortex_Th_BG.py --W_C 1.0 --inp_BLA 0.7 0.0 --inp 1.0 1.0 -s $seed
done 