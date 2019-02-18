# Gaussian-Mixture-Model
Expectation Maximization Algorithm

Gaussian Mixture Models (GMM) are effective for multi model density estimation. In this experiment GMM algorithm results are shown for two datasets. The GMM algorithm and plotting functions are given in python code.

Following are the requirements to run this code:
Python 3

To run this code type:
python main.py


In our experiments we considered two data sets:
1. Iris
2. Glass Classification Dataset


1. Results for Iris Dataset
First for each feature in iris dataset, Gaussian mixture models (GMM) parameters are estimated by using two or three GMM components. The Number of components in GMM are determined by visualizing respective feature’s histogram.










Gaussian Mixture Estimation for Two Features:

It can be seen here that distribution of all three classes has been estimated correctly by using three components of GMM.






 
GMM can properly learn the distribution with only two components.








2. Results of Glass Classification Dataset
The dataset contains eight features. Following results are the density estimation of each feature using GMM. Different number of components of GMM are used for each feature, determined by visualizing histogram of that feature.




