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
EXCEPTIONS = [
    #TODO(T51629): remove exceptions after 2.4 release
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/feature_examples/pytorch",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/tutorials/pytorch",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/simple_applications/pytorch",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/feature_examples/popart/custom_operators",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/tutorials/pytorch/tut1_basics",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/tutorials/pytorch/tut3_mixed_precision",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/tutorials/pytorch/tut4_observing_tensors",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/feature_examples/pytorch/custom_op",
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.4/tutorials/pytorch/tut2_efficient_data_loading"
]


def get_all_links_from_file(rst_file_name):
    # Known issue: if a link is split over multiple lines, only the first line
    # (containing 'http') will be considered matched.

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


def check_url_works(url):
    print(f"Testing {url}")

    try:
        r = requests.head(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Allow the test to succeed with intermitent issues.
        # (TooManyRedirects is not caught as could be a broken url.)
        return None

    code = r.status_code
    message = requests.status_codes._codes[code][0]  # pylint: disable=protected-access

    print(message + f" ({code})")

    if r.status_code == 302:
        check_url_works(r.headers['Location'])
    else:
        # Allow any non 4xx status code, as other failures could be temporary
        # and break the CI tests.
        if r.status_code >= 400 and r.status_code < 500:
            return url, message, code
        print()

    return None


def test_all_links():
    user_guide_path = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), DOC_FOLDER))
    failed_urls = []

    for rst_file in glob.glob(f"{user_guide_path}/*.rst"):
        for url in get_all_links_from_file(rst_file):
            url_result = check_url_works(url)
            if url_result is not None:
                url, message, code = url_result
                if url in EXCEPTIONS:
                    print(f"{url} found in exceptions: ",
                          f"ignoring {message} ({code})")
                else:
                    failed_urls.append(f"{url}: {message} ({code})")
            print()

    assert not failed_urls, failed_urls
