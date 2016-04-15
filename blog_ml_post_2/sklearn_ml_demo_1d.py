#!/usr/bin/python3
"""
sklearn_iot_ad.py
Copyright 2016 Snaplogic

sklearn_ml_demo_1d

First draft written by Shayne Hodge
Written for Python3, should work with 2.7
Requires seaborn, matplotlib, numpy, sklean

"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from lib_sklearn_demo import *

"""
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


def one_var_signal(settings_dict):
    '''Essentially a helper function around a bunch of other functions
    defined elsewhere to turn settings_dict into a useful set of data
    vectors.'''
    points = settings_dict["1-points"]
    (t, sig_clean) = one_clean_signal(settings_dict, points)
    (sig, anoms, a_idx) = one_dirty_signal(sig_clean, settings_dict)
    return (t, sig_clean, sig, a_idx)


def one_clean_signal(settings_dict, points):
    '''Generates points points of a 'constant' signal that is noisy in
    that it is actually samples of a normal random variable with mean
    and sigma normal-mu and normal-sigma.'''
    sig_mu = settings_dict["sig-mu"]
    sig_sig = settings_dict["sig-sig"]
    t = np.linspace(0, 20, num=points, endpoint=True)
    base_signal_amp = np.random.normal(loc=sig_mu, scale=sig_sig, size=points)
    clean_signal = base_signal_amp * np.ones(points)
    return (t, clean_signal)


def one_dirty_signal(sig_clean, settings_dict):
    '''Takes a clean signal and outputs one with anomalies.'''
    points = len(sig_clean)
    a_prob = settings_dict["1-var-anom-prob"]
    a_mu = settings_dict["1-anom-mu"]
    a_sig = settings_dict["1-anom-sig"]
    anom_indices = np.random.binomial(1, a_prob, points).astype(int)
    a_idx = np.where(anom_indices == 1)[0]
    base_add_anomaly = np.random.normal(loc=a_mu, scale=a_sig, size=points)
    anomaly_add_signal = base_add_anomaly*anom_indices
    sig_dirty = sig_clean + anomaly_add_signal
    return (sig_dirty, base_add_anomaly, a_idx)


def score_model(clf, sig, t, idx):
    '''Returns a series of measurements of the goodness of a model.'''
    n = len(sig)
    # The reshape is per a deprecation warning
    predict = clf.predict(sig.reshape(-1, 1))
    (good_data, bad_data) = split_good_bad_data(predict, idx)
    (metrics, metrics_string) = compute_metrics(good_data, bad_data, n)
    return (metrics, metrics_string)


def train_model(sig):
    '''Based heavily on the sklearn docs.'''
    # http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#example-svm-plot-oneclass-py
    # Note we really should be normalizing this data first
    nu = 0.1
    clf = svm.OneClassSVM(nu=nu, kernel="rbf")
    # The reshape is per a deprecation warning
    clf.fit(sig.reshape(-1, 1))
    return clf


def plot_signals(t, sig, sig_clean, a_idx, clf):
    def plot_decision_bounds(ax, t, bounds):
        ax.plot((t[0], t[-1]), (bounds, bounds), linestyle='--', color='black')

    sns.set(style="darkgrid", palette="Set2")
    anom_points = sig[a_idx]
    class_range = np.linspace(1.5*min(sig_clean), 1.5*max(sig_clean), num=100)
    Z_odd_shape = clf.decision_function(class_range.reshape(-1, 1))
    Z = np.ravel(Z_odd_shape)
    # Find the zero crossing
    # http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    # pdb.set_trace()
    zc_i = np.where(np.diff(np.sign(Z)))[0]
    zc = class_range[zc_i]
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, sig)
    ax1.plot(t[a_idx], anom_points, "ro", markersize=7.5)
    ax2.plot(t, sig_clean)
    plot_decision_bounds(ax1, t, zc)
    plot_decision_bounds(ax2, t, zc)
    ax1.set_title('1-var Dirty Signal')
    ax2.set_title('1-var Clean Signal')
    plt.show()


def main():
    np.random.seed(42)
    # Bunch of setup stuff
    args = setup_parser().parse_args()
    settings_dict = load_settings(args.settings)
    # One variable signal
    (t, sig_clean, sig, idx) = one_var_signal(settings_dict)
    clf = train_model(sig_clean)
    (metrics, metrics_string) = score_model(clf, sig, t, idx)
    print(metrics_string)
    plot_signals(t, sig, sig_clean, idx, clf)

if __name__ == '__main__':
    main()
