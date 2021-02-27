#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import glob
import os
import re
import requests

DOC_FOLDER = "../docs/user_guide"
URL_PATTERN = re.compile(r"`[^<]+\<http[^>]+\>`_")


def get_all_links_from_file(rst_file_name):
    print(f"Reading {rst_file_name}")

    all_links = []

    # Force as extended ASCII to avoid decoding erors:
    # assume all urls are made of 8-bit chars only
    with open(rst_file_name, 'r', encoding="latin-1") as rst_file:
        for line in rst_file:
            matches = URL_PATTERN.findall(line)
            for match in matches:
                url = match.split("<")[1].split(">")[0]
                all_links.append(url)

    return all_links


def assert_url_works(url):
    print(f"Testing {url}")
    r = requests.head(url)
    code = r.status_code
    message = requests.status_codes._codes[code][0]  # pylint: disable=protected-access

    print(message + f" ({code})")

    if r.status_code == 302:
        assert_url_works(r.headers['Location'])
    else:
        assert r.status_code == 200
        print()


def test_all_links():
    user_guide_path = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), DOC_FOLDER))

    for rst_file in glob.glob(f"{user_guide_path}/*.rst"):
        for url in get_all_links_from_file(rst_file):
            assert_url_works(url)
        print()
