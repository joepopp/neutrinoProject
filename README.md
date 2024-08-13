# neutrinoProject
A simple machine learning model is created in pytorch to reconstruct the source direction of neutrino events from raw data measured in the IceCube detector. 
The first file labeled "controlModel.py" is the full, standard pytorch model without any surpising features. 
The second file labeled "timeConvolutionModel.py" contains a special "custom" feature. In order to utilize the time ordered relationships within the data, appropriate time features already present in the data are formatted into columns of a matrix where each row represents the charge measurements of each DOM in chronological order. A single, two dimensional convolution takes place with a kernel of height one and width corresponding to the (variable) number of inputted time features. The matrix then collapses into a single column holding the time ordered information contained within each given time related feature. This new matrix is then reformatted and used as a regular feature to be inputted into the standard pytorch model along with any other chosen (variable) features. Both of these training files use multi gpu processing, capable of using a variable number of gpus to speed training. 
The third file labeled "optimizer.py" is the optimizer developed to tune hyperparameters. It is created with raytune and uses the optuna algorithm to tune a variable list of hyperparameters. 

Testing on these models proved to be difficult due to exploding gradients. No good data could be collected on the effectiveness of this "time convolved" feature as a result. Further work will requre a fix to this issue as well as further testing of different parameters. 
