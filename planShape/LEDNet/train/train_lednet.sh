#!/bin/bash


dataset_dir=/home/kushagra/IIIT-H/PipesDataset/PipesDatasetUnshuffled

python main.py --savedir logs --datadir $dataset_dir --num-epochs 2 --batch-size 2 --num-workers 2 --epochs-save 1 --resume 
