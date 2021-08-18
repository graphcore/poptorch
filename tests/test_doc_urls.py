#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import glob
import os
import re
import requests

DOC_FOLDER = "../docs/user_guide"
URL_PATTERN = re.compile(r"\bhttps?:[^\s>]+")

# URLs which don't exist yet (e.g documentation for a future release) can be
# added to the list of exceptions below.
#
# Make sure to add a TODO(TXXXX) comment to remove the exception once the link
# is fixed.
EXCEPTIONS = []


def get_all_links_from_file(rst_file_name):
    print(f"Reading {rst_file_name}")

    all_links = []

    # Force as extended ASCII to avoid decoding erors:
    # assume all urls are made of 8-bit chars only
    with open(rst_file_name, 'r', encoding="latin-1") as rst_file:
        for line in rst_file:
            matches = URL_PATTERN.findall(line)
            for match in matches:
                all_links.append(match)

    return all_links


def assert_url_works(url):
    print(f"Testing {url}")

    try:
        r = requests.head(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Allow the test to succeed with intermitent issues.
        # (TooManyReditects is not caught as could be a broken url.)
        return

    code = r.status_code
    message = requests.status_codes._codes[code][0]  # pylint: disable=protected-access

    print(message + f" ({code})")

    if r.status_code == 302:
        assert_url_works(r.headers['Location'])
    else:
        # Allow any non 4xx status code, as other failures could be temporary
        # and break the CI tests.
        assert r.status_code < 400 or r.status_code >= 500, (
            f"{url}: {message} ({code})")
        print()


def test_all_links():
    user_guide_path = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), DOC_FOLDER))

    for rst_file in glob.glob(f"{user_guide_path}/*.rst"):
        for url in get_all_links_from_file(rst_file):
            try:
                assert_url_works(url)
            except AssertionError as e:
                if url in EXCEPTIONS:
                    print(f"{url} found in exceptions: ignoring {e}")
                else:
                    raise
        print()
