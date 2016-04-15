#!/usr/bin/python
"""
lib_sklearn_demo.py
Copyright 2016 Snaplogic

Helper library for iot_mqtt_demo
Original (bug-free) code by Shayne Hodge
"""

from __future__ import division, print_function
import json
import argparse
import numpy as np
from collections import OrderedDict


def fake_zip(x1, x2):
    return np.stack((x1, x2), axis=-1)


# This is basically a Python 2/3 compatibility shim
# http://legacy.python.org/dev/peps/pep-0469/
try:
    dict.iteritems
except AttributeError:
    # Python 3
    def itervalues(d):
        return iter(d.values())
    def iteritems(d):
        return iter(d.items())
else:
    # Python 2
    def itervalues(d):
        return d.itervalues()
    def iteritems(d):
        return d.iteritems()


def split_good_bad_data(data, idx):
    n = np.shape(data)[0]
    # We need to mask out the known anomalies
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    good_data = data[mask]
    bad_data = data[~mask]
    gu, gn = np.unique(good_data, return_counts=True)
    bu, bn = np.unique(bad_data, return_counts=True)
    return (good_data, bad_data)


def compute_metrics(good_data, bad_data, n):
    metrics = OrderedDict()
    metrics['true_pos'] = np.count_nonzero(good_data[good_data == 1])
    metrics['true_neg'] = np.count_nonzero(bad_data[bad_data == -1])
    metrics['false_pos'] = np.count_nonzero(bad_data[bad_data == 1])
    metrics['false_neg'] = np.count_nonzero(good_data[good_data == -1])
    # Some common measurements
    # Some common measurements
    metrics['accuracy'] = (metrics['true_pos'] + metrics['true_neg']) / n
    metrics['precision'] = metrics['true_pos'] / (metrics['true_pos'] +
                                                  metrics['false_pos'])
    metrics['recall'] = metrics['true_pos'] / (metrics['true_pos'] +
                                               metrics['false_neg'])
    metrics['F1'] = 2 * ((metrics['precision'] * metrics['recall']) /
                         (metrics['precision'] + metrics['recall']))
    metrics_array = ['{}:\t{}'.format(key, value) for (key, value) in
                     iteritems(metrics)]
    metrics_string = '\n'.join(metrics_array)
    return (metrics, metrics_string)


# Command line argument parser
def setup_parser():
    '''Set up argparse.'''
    default_input_json = 'sklearn-input.json'
    parser = argparse.ArgumentParser(
        description='sk-learn AD demo')
    parser.add_argument(
        '-s', '--settings',
        action='store',
        default=default_input_json,
        help='Path to input JSON file.')
    parser.add_argument(
        '-f', '--filename',
        action='store',
        help='Path to write clean signal output to, if desired.')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Turn on some debug messaging to stdout.')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Turn on the plot.')
    return parser


def fake_logging(verbose, message):
    if verbose:
        print(message)


def load_settings(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
    return settings
