"""
Tests for fake_data.py
Copyright 2017 SnapLogic, Inc. and Shayne Hodge
"""
import pytest
import sys
sys.path.insert(0, '../fake_data')
from fake_data import fd


@pytest.fixture()
def known_config():
    cfg = {
        'command_line_args': {
            'batch': 5,
            'faker': 5,
            'num_users': 10,
            'numeric': 10,
            'rows': 20,
            'seed': 8000,
            'output': './testdata/pytestdata'
        },
        'non_vectorized_functions': {
            'address': 'fake.street_address',
            'state': 'fake.state_abbr',
            'zip': 'fake.postalcode'}
    }
    return cfg


@pytest.fixture()
def filename():
    f = "./tests/test_config_fake_data.json"
    return f


@pytest.fixture()
def known_output_files():
    main_f = './ref_csvs/main_test.csv'
    plookup_f = './ref_csvs/plookup_test.csv'
    ulookup_f = './ref_csvs/ulookup_test.csv'
    main = []
    plookup = []
    ulookup = []
    return (main, plookup, ulookup)


def test_check_config_file_load(known_config, filename):
    config = fd.load_config_file(filename)
    assert known_config == config


def test_check_file_parse(known_config, filename):
    cmd_args = known_config['command_line_args']
    nvfs = known_config['non_vectorized_functions']
    arg_string = ['-c', filename, '-v']
    parser = fd.setup_parser()
    args = parser.parse_args(arg_string)
    fake = fd.get_fake_factory()
    args, func_dict = fd.parse_config(args, fake)
    assert args.config_file == filename
    assert args.batch == cmd_args['batch']
    assert args.faker == cmd_args['faker']
    assert args.num_users == cmd_args['num_users']
    assert args.numeric == cmd_args['numeric']
    assert args.rows == cmd_args['rows']
    assert args.seed == cmd_args['seed']
    for key in nvfs.keys():
        assert key in func_dict
    assert 'nothing' not in func_dict


def test_outfiles(known_output_files, known_config, filename):
    #TODO - capture output of script, compare to known files
    #need to find way to capture csv output to fixture
    assert True is True
