
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import matplotlib as mpl

import pandas as pd

def plot_1D(gmm,x,col):
  plt.hist(x,normed=True)
  x = np.linspace(x.min(), x.max(), 100, endpoint=False)
  ys = np.zeros_like(x)

  i=0
  for w in gmm.phi:
      y=sp.multivariate_normal.pdf(x, mean=gmm.mean_arr[i], cov=gmm.sigma_arr[i])*w

      plt.plot(x, y)
      ys += y
      i+=1

  plt.xlabel(col)
  plt.plot(x,ys)
  plt.show()


def make_ellipses(gmm, ax):
    colors = ['turquoise', 'orange']
    for n, color in enumerate(colors):
        covariances = gmm.sigma_arr[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 3. * np.sqrt(2.) * np.sqrt(v)
        mean=gmm.mean_arr[n]
        mean=mean.reshape(2,1)
        print(mean)
        ell = mpl.patches.Ellipse(mean, v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')

def plot_2D(gmm,x,col,label):

    h = plt.subplot(111, aspect='equal')
    make_ellipses(gmm, h)



    plt.scatter(x[:,0],x[:,1],c=label['Species'],marker='x')
    plt.xlim(-3, 9)
    plt.ylim(-3, 9)
    plt.xlabel(col[0])
    plt.ylabel(col[1])
    #plot_cov_ellipse(gmm.sigma_arr[:,:,0],gmm.mean_arr[:,1],ax=ax[0,0])
    plt.show()
