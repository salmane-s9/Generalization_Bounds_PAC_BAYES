#!/bin/bash
echo "before activating environment :$PATH$"
source /home/veshalaeva/miniconda3/bin/activate apriori
echo "after activating environment :$PATH$"
python3 -u -m torch.utils.bottleneck /home/veshalaeva/Generalization_Bounds_PAC_BAYES/Main.py -model 'T-600' > /home/veshalaeva/Generalization_Bounds_PAC_BAYES/profile_out_T-600.txt