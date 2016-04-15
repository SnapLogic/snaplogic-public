#!/usr/bin/python3
"""
2d_ml_demo.py
Copyright 2016 Snaplogic

First draft written by Shayne Hodge
Written for Python3, should work with 2.7
Requires seaborn, matplotlib, numpy, sklearn, and pandas

"""

from __future__ import print_function, division
import matplotlib.font_manager
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import svm
from lib_sklearn_demo import *

"""
Notes: The two-variable data pair is notated by "x1" and "x2". Also,
unlike the one-variable script, this script uses pandas.

Indices = The binomial function used to generate anomalies returns a
vector that is the same length as that of the signal.  It is basically
a one-hot vector - e.g., [0, 0, 1, 0, ... 1, 0], where a one corresponds
to an anomaly at that point.  Numpy/Python cannot directly index off
of that, so np.where() is used to generate a vector of index values. In
at least one instance, the raw long vector is converted to Boolean,
which allows its complement (points without anomalies) to be indexed.

A clean signal is used to train the detector (and hence functions as
the training set).  A dirty signal acts as the test set.  Since we
aren't varying model parameters no validation set is used.

The two demo files share the same lib (lib_sklearn_demo) and the same
config json, though no variable should affect both files.
"""


COL = ['x1', 'x2', 'label']


def get_mean_cov_array(set_d, clean=True, test_n=False):
    points = (set_d["2-points-train"] if (clean and not test_n) else
              set_d["2-points-test"])
    mu1 = set_d["mu1"] if clean else set_d["a_mu1"]
    mu2 = set_d["mu2"] if clean else set_d["a_mu2"]
    c12 = set_d["c12"] if clean else set_d["a_c12"]
    c21 = set_d["c21"] if clean else set_d["a_c21"]
    c11 = 1
    c22 = 1
    mean = np.array([mu1, mu2])
    cov = np.array([[c11, c12], [c21, c22]])
    return (points, mean, cov)


def two_var_clean_signal(settings_dict, test_n=False):
    (points, mean, cov) = get_mean_cov_array(settings_dict, clean=True,
                                             test_n=test_n)
    random_points = np.random.multivariate_normal(mean, cov, points)
    labels = np.zeros(points)
    df = pd.DataFrame(np.column_stack((random_points, labels)),
                      columns=COL)
    return df


def two_var_anomalies(settings_dict):
    '''Generate the dirty signal and accouterments.'''
    a_prob = settings_dict["2-var-anom-prob"]
    df = two_var_clean_signal(settings_dict, test_n=True)
    (points, mean, cov) = get_mean_cov_array(settings_dict, clean=False)
    random_points = np.random.multivariate_normal(mean, cov, points)
    labels = np.ones(points)
    df_dirty = pd.DataFrame(np.column_stack((random_points, labels)),
                            columns=COL)
    idx = np.where(np.random.binomial(1, a_prob, points) == 1)[0]
    df.iloc[idx] = df_dirty.iloc[idx]
    return df


def score_model(clf, df):
    '''Returns a series of measurements of the goodness of a model.'''
    n = len(df)
    X = df[COL[:2]].values
    predict = clf.predict(X)
    idx = df.loc[df[COL[2]] == 1].index.tolist()
    (good_data, bad_data) = split_good_bad_data(predict, idx)
    (metrics, metrics_string) = compute_metrics(good_data, bad_data, n)
    return (metrics, metrics_string)


def train_model(df):
    '''Based heavily on the sklearn docs.'''
    # http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#example-svm-plot-oneclass-py
    # Should be normalizing this data first
    nu = 0.05
    clf = svm.OneClassSVM(nu=nu, kernel="rbf")
    clf.fit(df[['x1', 'x2']])
    return clf


def visualize_detector(settings, clf, df):
    sns.set(style="darkgrid", palette="Set2")
    # plot the line, the points, and the nearest vectors to the plane
    (c_x, c_y) = (settings["mu1"], settings["mu2"])
    lim = 4.5
    xx, yy = np.meshgrid(np.linspace(c_x-lim, c_x+lim, 100*lim),
                         np.linspace(c_y-lim, c_y+lim, 100*lim))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    # Trying to force a 1:1 aspect ratio, without much success
    ax1 = fig.add_subplot(111, adjustable='box', aspect=1.0)
    plt.title("Novelty Detection")
    ax1.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                 cmap=plt.cm.Blues_r)
    a = ax1.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    ax1.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
    X_g = df[df.label == 0]
    X_a = df[df.label == 1]
    b1 = ax1.scatter(X_g.x1, X_g.x2, c='w',
                     alpha=0.5)
    b2 = ax1.scatter(X_a.x1, X_a.x2, c='r')
    # Uncomment these lines if you want to number the anomaly points
    # for i, txt in enumerate(range(len(X_a.x1))):
    #    ax1.annotate(txt, (X_a.x1.values[i], X_a.x2.values[i]))
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim((c_x-lim, c_x+lim))
    plt.ylim((c_y-lim, c_y+lim))
    plt.legend([a.collections[0], b1, b2],
               ["learned frontier", "regular observations",
                "abnormal observations"],
               loc="upper left",
               frameon=True,
               framealpha=1.0,
               prop=matplotlib.font_manager.FontProperties(size=14))
    plt.show()


def main():
    np.random.seed(42)
    # Bunch of setup stuff
    args = setup_parser().parse_args()
    settings_dict = load_settings(args.settings)
    # Get the signals
    df_clean = two_var_clean_signal(settings_dict)
    df_dirty = two_var_anomalies(settings_dict)
    # Train
    clf = train_model(df_clean)
    datasets = [(df_clean, 'Train Data, (clean)'),
                (df_dirty, 'Test Data Set')]
    # Calculate metrics for the training and test sets.  The training
    # set should score better.
    for (df, label) in datasets:
        (metrics, metrics_string) = score_model(clf, df)
        print("\n" + label + ":\n")
        print(metrics_string + "\n")
    # Plot results
    visualize_detector(settings_dict, clf, df_dirty)

if __name__ == '__main__':
    main()
