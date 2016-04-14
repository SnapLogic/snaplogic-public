#!/usr/bin/python
"""
lib_apache_log_gen.py
Copyright 2016 Snaplogic

Helper library for apache_log_gen
"""

from __future__ import division, print_function
import argparse
import pandas as pd

CODES = ['200', '400', '401', '403', '404', '500']

def setup_parser():
    '''Set up argparse.'''
    batch_low = 1
    batch_high = 10001
    parser = argparse.ArgumentParser(
        description='NaturalLog Apache Log Generator')
    parser.add_argument(
        '-i', '--ip',
        action='store',
        default='./config/ip_list.txt',
        help='Path to list of incoming IP addresses.')
    parser.add_argument(
        '-r', '--referrers',
        action='store',
        default='./config/referrer.txt',
        help='Path to list of referrers.')
    parser.add_argument(
        '-u', '--useragents',
        action='store',
        default='./config/user_agent.txt',
        help='Path to list of user agents.')
    parser.add_argument(
        '-e', '--endpoints',
        action='store',
        default='./config/endpoints.json',
        help='Path to JSON of endpoints.')
    parser.add_argument(
        '-o', '--output',
        action='store',
        default='./output/naturallog.log',
        help=('Path to output log.  Will append to file if it already '
              'exists.'))
    parser.add_argument(
        '-b', '--batch',
        action='store',
        type=int,
        default=5,
        choices=range(batch_low, batch_high),
        help=('Log entries to generate per output file write, can be '
              'set between {} and {}.'.format(batch_low, batch_high)),
        metavar='BATCH')
    parser.add_argument(
        '-d', '--delay',
        action='store',
        type=float,
        default=0.1,
        help='Delay in seconds between batch generation.')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Turn on some debug messaging to stdout.')
    return parser.parse_args()


def fake_logging(verbose, message):
    if verbose:
        print(message)


def parse_endpoints_file(file_name):
    '''Load the JSON file into a pandas dataframe, process it a bit,
    and return a more usefully formatted structure.'''
    settings_df = pd.read_json(file_name)
    endpoint_weight_sum = settings_df['weight'].sum()
    # Normalize endpoint weights to percentages
    settings_df['weight'] /= endpoint_weight_sum
    # Normalize code weights to percentages
    # This is confusing, code borrowed from http://bit.ly/1TTzeqp
    settings_df[CODES] = settings_df[CODES].div(
                            settings_df[CODES].sum(axis=1), axis=0)
    return settings_df
