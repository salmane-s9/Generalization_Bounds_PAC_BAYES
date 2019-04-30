### Experiments

#### TODO

* For the model T-600 (pretrained model) and parameters setting as in the paper, get the following outputs

  * intermediate values during PAC-Bayes bound optimization (BRE_Loss, NN-loss, parameters norms, including weights, $\sigma$ and $\lamda$). Make a plot, where x-axis is the number of epochs and y-axis reveals the value. (Needs change in the main function)
  * PAC-Bayes bound value.

Note, that the norm of sigma parameter is growing during the training because it depends on weight values (initialization of sigma is absolute weight values).

* Add to README.md file, the short description (one-two sentences) of files in the project.ls

* Analyze dependence of bound value from initial choice of $w_0$ as a mean of prior distribution.  

#### DONE

* Create a file that contains a function to train custom neural network. It takes as an input a custom model (defined in Architectures), and return weights.
In the case we don't pass *weight_pass* parameter, meaning there is no pretrained model. 
Then, it will be trained first and obtained weights will be passed to PAC-Bayes bound optimizer.

* Customize a parameter "Model-name".

##### Running experiments
* Pac-Bayes Bound Optimization for the model T-600, JOB_ID = 1678534
* Pac-Bayes Bound Optimization for the model R-600, JOB_ID = 

#### Ideas
* After running small experiments, we can run its full version (setting all parameters as in the paper) on a server.
We expect to get results as in the paper, probably not exactly the same but similar.

#### To run on the Grid5000 cluster 
* How to pause job if it's not enough resources to finish it in one row?

##### Usepolicycheck
* to check current usage:
usagepolicycheck -t

* to check daily allowance:
usagepolicycheck -l --sites lille

#####  Reserve resources

* Passive mode with planning
oarsub -p "cluster='chifflet'" -r '2019-04-15 19:00:00' -l nodes=1,walltime=14:00:00 -O ~/Generalization_Bounds_PAC_BAYES/output_T-600.txt ~/Generalization_Bounds_PAC_BAYES/executor.sh

* Passive mode without planning
oarsub -p "cluster='chifflet'" -l nodes=1,walltime=72:00:00 -O output_T-600.txt ~/Generalization_Bounds_PAC_BAYES/executor.sh

Interactive mode (
  * use >>tmux to not get job deleted when the terminal is closed; 
  * to restore the previous session type >>tmux a;
  * to close the terminal type Ctrl+b then d)

oarsub -p "cluster='chifflet'" -r '2019-03-29 13:05:00' -I -l nodes=4,walltime=6:55:0 "./executor.sh"

oarsub -q production -p "cluster='grele'" -I
oarsub -t deploy -q production -p "cluster='grele'" -l nodes=1,walltime=2 -I

###### To see statistics of the job:
oarstat -s -j <JOB_ID>

cuda_9.0.176_384.81_linux-run

###### Documentation
[Grid5000 Getting Started](https://www.grid5000.fr/w/Getting_Started#Deploying_your_nodes_to_get_root_access_and_create_your_own_experimental_environment)

[Grid5000 Accelerators](https://www.grid5000.fr/w/Accelerators_on_Grid5000#Compiling_the_CUDA_Toolkit_examples)

[Grid5000 Resources schedule](https://intranet.grid5000.fr/oar/Lille/drawgantt-svg/)

[Grid5000 DeepLearning Tutorial](http://deeploria.gforge.inria.fr/g5k/Tuto%20Deep%20Learning%20-%20Grid5000.html#nvidia-smi_tool)



