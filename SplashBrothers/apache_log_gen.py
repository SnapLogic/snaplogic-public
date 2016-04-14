#!/usr/bin/python
"""
SplashBrothers, Apache Log Generator
apache_log_gen.py
Copyright 2016 Snaplogic

Fake apache log generator.
"""

from __future__ import division, print_function
import numpy as np
from time import clock, sleep, strftime
from lib_apache_log_gen import *


def row_prob_conversion(s_df, labels):
    '''Return dictionary, keyed by endpoint label, where value is
    tuple of the return code probs.'''
    x = {}
    for ix, entry in s_df.iterrows():
        x[entry['endpoint']] = tuple(entry[labels])
    return x


def col_prob_conversion(s_df, col):
    '''Docstring coming Soon(TM)'''
    x = tuple(s_df[col])
    return x


def return_list_from_file(data_file):
    '''Docstring coming Soon(TM)'''
    with open(data_file) as f:
        lines = f.readlines()
    data_list = [x.strip() for x in lines]
    return data_list


def calc_bytes(mu, sigma):
    '''Return a random value for a normal PDF with mu and sigma,
    converted to an integer and coerced to be non-negative.'''
    byt = np.random.normal(loc=mu, scale=sigma)
    # Turn these into positive integers
    byt = np.ceil(np.abs(byt)).astype(int)
    return byt


def create_final_string(ip, user, time, end, code, byt, refr, agnt):
    '''Docstring coming Soon(TM)'''
    final_string = ("{} user-identifier {} [{} -0700] \"GET {} HTTP/1.0\" {} {} \"{}\" \"{}\""
                    .format(ip, user, time, end, code, byt, refr, agnt))
    return final_string


def rand_choose_cat(probs, labels):
    '''Docstring coming Soon(TM)'''
    one_hot_result = np.random.multinomial(1, probs)
    ix = np.nonzero(one_hot_result)[0][0]
    out = labels[ix]
    return out


def write_results(results, file):
    '''Write the results to the log file.'''
    with open(file, mode='a') as f:
        f.write(results)


def gen_cat_probs(some_list):
    """Small hack to allow IP list to be used with rand_choose_cat()"""
    return [(1/len(some_list))]*len(some_list)


def main():
    '''Docstring coming Soon(TM).  (Around the time the tests are
    implemented.)'''
    args = setup_parser()
    settings_df = parse_endpoints_file(args.endpoints)
    endpoint_probs = col_prob_conversion(settings_df, 'weight')
    code_probs_dict = row_prob_conversion(settings_df, CODES)
    endpoints = tuple(settings_df['endpoint'])
    ip_list = return_list_from_file(args.ip)
    ref_list = return_list_from_file(args.referrers)
    useragents_list = return_list_from_file(args.useragents)
    user_list = ['ChuckNorris', 'DonaldKnuth', 'l33th4x0r', 'HanSolo',
                 'BruceWayne', 'StephenCurry', 'KlayThompson', 'FordPrefect',
                 'CarlGauss', 'JohnTukey', 'TonyStark']
    ip_prob = gen_cat_probs(ip_list)
    ref_prob = gen_cat_probs(ref_list)
    useragent_prob = gen_cat_probs(useragents_list)
    user_prob = gen_cat_probs(user_list)
    while True:
        start_time = clock()
        string_list = []
        for i in range(args.batch):
            # This append won't scale well, but it shouldn't be a
            # problem in this application
            # TODO: preinit a struct with the batch size and fill it in
            # to avoid the repeated memory copies
            a_time = strftime("%d/%B/%Y:%H:%M:%S")
            ip = rand_choose_cat(ip_prob, ip_list)
            user = rand_choose_cat(user_prob, user_list)
            end = rand_choose_cat(endpoint_probs, endpoints)
            useragent = rand_choose_cat(useragent_prob, useragents_list)
            referrer = rand_choose_cat(ref_prob, ref_list)
            # Need to index based on endpoint
            row_df = settings_df[settings_df['endpoint'] == end]
            code = rand_choose_cat(code_probs_dict[end], CODES)
            byt = calc_bytes(int(row_df['mu_bytes']),
                             int(row_df['sigma_bytes']))

            log_string = create_final_string(ip, user, a_time, end, code, byt,
                                             referrer, useragent)
            string_list.append(log_string)
        final = "\n".join(string_list)
        write_results(final, args.output)
        end_time = clock()
        delta = end_time - start_time
        message = "Execution time was {} ms.".format(delta*1000)
        fake_logging(args.verbose, message)
        if sleep:
            sleep(args.delay)

if __name__ == '__main__':
    main()
