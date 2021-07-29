# -*- coding: utf-8 -*-
"""
desciption: system variables or other constant information
"""
import os
import requests
import argparse

CONFIG = {
    'url': 'jdbc:postgresql://166.111.121.62:5432/',
    'host': '166.111.121.62',
    'port': 5432,
    'driver': 'org.postgresql.Driver',
    'username': 'postgres',
    'password': 'postgres',
    'schema': 'tpch1x',
    'sqldir': 'tpch',
    'logdir': 'rewrite_results'
}

def parse_cmd_args():

    parser = argparse.ArgumentParser()

    # benchmark
    parser.add_argument('--workload', type=str, default='tpch', help='[tpch, user]')

    args = parser.parse_args()
    argus = vars(args)

    return argus
