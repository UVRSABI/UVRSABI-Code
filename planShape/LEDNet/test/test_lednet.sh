#!/bin/bash


dataset_dir=/home/kushagra/IIIT-H/PipesDataset/PipesDatasetUnshuffled
result_dir=/home/kushagra/IIIT-H/PipesDataset/PipesDatasetUnshuffled/Results


python eval_cityscapes_color.py --loadDir ../save/logs --datadir $dataset_dir --subset val --resultdir $result_dir 