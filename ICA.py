import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def do_ICA():

    fp = '/home/admin123/MLExperiments/data/glass.csv'
    df = pd.read_csv(fp)
    # initial value of number of components based on experiments
    nc_min = 4
    nc_max = df.shape[1] # one more than num columns in data

    X = df.ix[:, 0:9]
    X_s = preprocessing.scale(X)

    # fit an ICA

    for num_comp in range(nc_min, nc_max):
        ica = FastICA(n_components = num_comp)
        S_ = ica.fit_transform(X_s)
        A_ = ica.mixing_

        # check the fit
        X_sA = np.dot(S_, A_.T) + ica.mean_

        good_fit = np.allclose(X_s, X_sA, rtol = 1e-02, atol=1e-02)

        if good_fit:
            print("Required number of components is :" + str(num_comp))
            break

    # plot the fit
    plt.rc('axes', color_cycle=['r', 'g', 'b', 'y'])

    num_comps = S_.shape[1]

    for comp in range(num_comps):
        plt.plot(S_[:, comp])

    plt.show()

    return
    

    
    
    
