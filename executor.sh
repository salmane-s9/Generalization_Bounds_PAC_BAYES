#!/bin/bash
echo "before activating environment :$PATH$"
source /home/veshalaeva/miniconda3/bin/activate apriori
echo "after activating environment :$PATH$"
python3 -u /home/veshalaeva/Generalization_Bounds_PAC_BAYES/Main.py -model 'T-600' -prior_mean 'zeros' > /home/veshalaeva/Generalization_Bounds_PAC_BAYES/output_T-600_epoch_20_zeros.txt