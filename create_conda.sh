#!/usr/bin/env bash
source deactivate
conda create --name pycourse_task
source activate pycourse_task
conda install -c conda-forge keras jupyter matplotlib flask tensorflow scikit-learn python=3.6
