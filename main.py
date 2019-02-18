
# Author -----Sohaib Kiani and Usman Sajid
import pandas as pd
from GMM import *
import util as plot

def For_Iris(features,No_Component=2):

    data = pd.read_csv("Iris.csv", header = 0)
    data = data.reset_index()



    #For one feature
    if (features==1):

        col='SepalWidthCm'
        x=data[[col]]
        x=np.array(x)
        gmm = GaussianMixModel(x,No_Component)
        gmm.fit()
        plot.plot_1D(gmm,x,col)

    else:

        replace_map = {'Species': {'Iris-virginica': 1, 'Iris-versicolor': 2,'Iris-setosa':3}}
        data.replace(replace_map, inplace=True)
        label=data[['Species']]
        col=['SepalWidthCm','PetalLengthCm']
        x=data[col]
        x=np.array(x)
        gmm = GaussianMixModel(x,No_Component)
        gmm.fit()
        plot.plot_2D(gmm,x,col,label)

def For_Glass(No_Component=2):
    data = pd.read_csv("glass.csv", header = 0)
    col='Fe'
    x=data[[col]]
    x=np.array(x)
    No_Component=1

    gmm = GaussianMixModel(x,No_Component)
    gmm.fit()
    plot.plot_1D(gmm,x,col)




def main():
  #For_Glass()
  For_Iris(1,2)
if __name__== "__main__":
  main()
