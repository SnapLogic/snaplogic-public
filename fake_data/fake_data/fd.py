#!/usr/bin/python3
"""
Copyright 2017 SnapLogic, Inc.
Created by Shayne Hodge
Create fake test data.
Tested with Python 3.6
TODO: May rely on PEP 468, but believe that got rewritten
"""

import csv
import click
import time
import json
import numpy as np
from os import stat
from math import ceil
from faker import Factory
from collections import OrderedDict
try:
    from lib_fake_data import setup_parser, fake_logging
except:
    from .lib_fake_data import setup_parser, fake_logging


def load_config_file(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as err:
        print("Could not parse config file:\n{}".format(err))
        raise err
    return config


def parse_config(args, fake):
    label = "non_vectorized_functions"
    config = load_config_file(args.config_file)
    # Can't do a dict comprehension here, because fake won't be in
    # scope.
    func_dict = OrderedDict()
    for key, f in config[label].items():
        try:
            func_dict[key] = eval(f)
        except Exception as err:
            print("Failed on key: {}, function: {} in creating function "
                  " dictionary.".format(key, f))
            raise err
    # Change the arguments to the ones from the config file unless
    # specified on the command line not to.
    if not args.use_cmd_args:
        try:
            for arg, value in config["command_line_args"].items():
                if arg not in ('verbose', 'output', 'use_cmd_args'):
                    setattr(args, arg, value)
        except Exception as err:
            print("Couldn't create argument dictionary.")
            raise err
    return args, func_dict


def default_funcs(fake):
    func_dict = OrderedDict({
        'address': fake.street_address,
        'state': fake.state_abbr,
        'zip': fake.postalcode,
        'color': fake.color_name,
        'phone': fake.phone_number,
        'browser': fake.firefox,
        'platform': fake.mac_platform_token,
        'ip': fake.ipv6,
        'email': fake.email,
        'cc': fake.credit_card_number,
        'job': fake.job,
        'uuid': fake.uuid4,
        'ssn': fake.ssn,
        'ean': fake.ean13,
        'nothing': fake.bs})
    return func_dict


def roundup(total_size, batch_size):
    # https://stackoverflow.com/questions/8866046/python-round-up-integer-to-next-hundred
    return int(ceil(total_size / batch_size)) * batch_size


def create_users(num_user, fake):
    return {n: fake.name() for n in range(num_user)}


def create_products(num_products, fake):
    (low, high) = (12500, 12500 + num_products)
    return {n: fake.catch_phrase() for n in range(low, high)}


def create_vector_data(set_size, gen_function):
    return np.array([gen_function() for _ in range(set_size)])


def create_rows(batch_size, data_vector):
    data_dict = OrderedDict()
    for (col_name, col) in data_vector.items():
        data_dict[col_name] = np.random.choice(col, batch_size)
    output = np.column_stack((data_dict.values()))
    return output


def write_dict_to_csv(data_dict, header, fname):
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar="'",
                            quoting=csv.QUOTE_ALL)
        rows = [x for x in data_dict.items()]
        writer.writerow(header)
        writer.writerows(rows)


def get_fake_factory():
    return Factory.create()


def create_file_names(base, ts):
    fname = '{}_{}.csv'.format(base, ts)
    fname_users = '{}_{}_users_lookup.csv'.format(base, ts)
    fname_products = '{}_{}_products_lookup.csv'.format(base, ts)
    return (fname, fname_users, fname_products)


def main():
    ts = time.time()
    parser = setup_parser()
    args = parser.parse_args()
    fake = get_fake_factory()
    if args.config_file:
        args, func_dict = parse_config(args, fake)
        fake_logging(args.verbose,
                     "Using config file: {}".format(args.config_file))
        fake_logging(args.verbose, "Arguments:")
        for k, v in vars(args).items():
            fake_logging(args.verbose, '{}: {}'.format(k, v))
    else:
        func_dict = default_funcs(fake)

    # Var assignments and some simple transforms
    num_products = args.users
    num_users = args.users
    batch_size = args.batch
    ver = args.verbose
    total_size = roundup(args.rows, batch_size)
    batches = int(total_size / batch_size)
    fake.seed(args.seed)
    np.random.seed(args.seed)
    (fname, fname_users, fname_products) = create_file_names(args.output, ts)
    # Creating some large sets to reuse, will speed up execution.
    # Creates repeats in every column, allowing meaningful aggregates
    # on any column
    data_vector = OrderedDict()
    user_dict = create_users(num_users, fake)
    product_dict = create_products(num_products, fake)
    data_vector['user'] = np.array(list(user_dict.keys()))
    data_vector['product'] = np.array(list(product_dict.keys()))
    for (key, f) in func_dict.items():
        data_vector[key] = create_vector_data(args.faker, f)
    data_vector['pct'] = np.linspace(0, 100, args.numeric)
    data_vector['total'] = np.linspace(200, 12000, args.numeric)
    fake_logging(ver, 'Done with initial vector creation.')
    fake_logging(ver, 'Starting data set generation.')
    header = list(data_vector.keys())
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar="'",
                            quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        with click.progressbar(range(batches), label='Percent Complete') as bar:
            for b in bar:
                temp = create_rows(batch_size, data_vector)
                writer.writerows(temp)
                #fake_logging(ver, '{:.2f}%'.format(100 * b / batches))
    write_dict_to_csv(user_dict, ['uid', 'user'], fname_users)
    write_dict_to_csv(product_dict, ['product_id', 'product'], fname_products)
    te = time.time()
    tdelta = te - ts
    fsize = stat(fname).st_size / 1024**2
    fake_logging(ver, "Wrote out {} rows to {}, size is {:.1f} MB".format(
                 total_size, fname, fsize))
    fake_logging(ver, "Elapsed time: {:.1f} seconds".format(tdelta))
    fake_logging(ver, "Throughput: {:.2f} MB/s".format(fsize / tdelta))


if __name__ == '__main__':
    main()
