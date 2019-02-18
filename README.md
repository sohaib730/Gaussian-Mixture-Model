# Gaussian-Mixture-Model
Expectation Maximization Algorithm

Gaussian Mixture Models (GMM) are effective for multi model density representation. In this experiment GMM are estimated using Expectation Maximization(EM) algorithm results are shown for two datasets. The GMM algorithm and plotting functions are given in python code.

Following are the requirements to run this code:
Python 3.7.2

To run this code type:
python main.py
Modify main.py, if number of components of GMM or particular feature in dataset needs to be selected.


For our experiments we considered two data sets:
1. Iris
2. Glass Classification Dataset

1. Results for Iris Dataset:
First for each feature in Iris dataset, Gaussian mixture models (GMM) parameters are estimated by using two or three GMM components. The Number of components in GMM are determined by visualizing respective feature's histogram. usually for this dataset features, two components were enough.

![](Figures/Iris/Figure_1.png)
![](Figures/Iris/Figure_2.png)
![](Figures/Iris/Figure_3.png)
![](Figures/Iris/Figure_4.png)


GMM Estimation for Two Features:
 
GMM can properly learn the distribution with only two components.

![](Figures/Iris/Figure_11.png)
![](Figures/Iris/Figure_12.png)
![](Figures/Iris/Figure_13.png)
![](Figures/Iris/Figure_14.png)
![](Figures/Iris/Figure_15.png)

2. Results of Glass Classification Dataset:
This dataset contains eight features. Following results are the density estimation of each feature using GMM. Different number of components of GMM are used for each feature, determined by visualizing histogram of that feature.
![](Figures/Glass/Figure_1.png)
![](Figures/Glass/Figure_2.png)
![](Figures/Glass/Figure_3.png)
![](Figures/Glass/Figure_4.png)
![](Figures/Glass/Figure_5.png)
![](Figures/Glass/Figure_6.png)
![](Figures/Glass/Figure_7.png)
![](Figures/Glass/Figure_8.png)
![](Figures/Glass/Figure_9.png)


