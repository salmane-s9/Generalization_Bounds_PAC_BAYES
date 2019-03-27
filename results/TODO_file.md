#### Experiments

1. For the model T-600 (pretrained model) and parameters setting as in the paper, get the following outputs

  * intermediate values during PAC-Bayes bound optimization (BRE_Loss, NN-loss, parameters norms, including weights, $\sigma$ and $\lamda$). Make a plot, where x-axis is the number of epochs and y-axis reveals the value. (Needs change in the main function)
  * PAC-Bayes bound value.

2. Create a file that contains a function to train custom neural network. It takes as an input a custom model (defined in Architectures), and return weights.
In the case we don't pass *weight_pass* parameter, meaning there is no pretrained model. 
Then, it will be trained first and obtained weights will be passed to PAC-Bayes bound optimizer.

3. Add to README.md file, the short description (one-two sentences) of files in the project.

#### Ideas
* After running small experiments, we can run its full version (setting all parameters as in the paper) on a server.
We expect to get results as in the paper, probably not exactly the same but similar.


#### To run on the server 

##### Reserve resources

oarsub -p "GPU <> 'NO'" -l "host=1, walltime=0:20:00" "/home/veshalaeva/Generalization_Bounds_PAC_BAYES/"

