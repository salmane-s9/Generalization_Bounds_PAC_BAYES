#!/bin/bash
echo "before activating environment :$PATH$"
source /home/veshalaeva/miniconda3/bin/activate apriori
echo "after activating environment :$PATH$"
python3 -u /home/veshalaeva/Generalization_Bounds_PAC_BAYES/Main.py -model 'R-600' > /home/veshalaeva/Generalization_Bounds_PAC_BAYES/output_R-600.txt