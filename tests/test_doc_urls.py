#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import glob
import os
import re
import requests

DOC_FOLDER = "../docs/user_guide"
URL_PATTERN = re.compile(r"\bhttps?:[^\s>]+")

# URLs which don't exist yet (e.g documentation for a future release) can be
# added to the dictionary of exceptions:
PRE_RELEASE_URLS = {
    "https://github.com/graphcore/tutorials/tree/sdk-release-2.6": "https://phabricator.sourcevertex.net/diffusion/TUTORIALS/browse/sdk-release-2.6"
}


def get_all_links_from_file(rst_file_name):
    # Known issue: if a link is split over multiple lines, only the first line
    # (containing 'http') will be considered matched.

    print(f"Reading {rst_file_name}")

    all_links = []

    # Force as extended ASCII to avoid decoding errors:
    # assume all urls are made of 8-bit chars only
    with open(rst_file_name, "r", encoding="latin-1") as rst_file:
        for line in rst_file:
            matches = URL_PATTERN.findall(line)
            for match in matches:
                all_links.append(match)

    return all_links


def convert_to_internal(url):
    for forwarder in PRE_RELEASE_URLS:
        if url.startswith(forwarder):
            print("Will try pre-release URL:")
            return True, url.replace(forwarder, PRE_RELEASE_URLS[forwarder], 1)

    return False, url


def check_url_works(url):
    print(f"Testing {url}")

    try:
        r = requests.head(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Allow the test to succeed with intermittent issues.
        # (TooManyRedirects is not caught as could be a broken url.)
        return None

    code = r.status_code
    message = requests.status_codes._codes[code][0]  # pylint: disable=protected-access

    print(f"{message} ({code})")

    if r.status_code == 302:
        check_url_works(r.headers["Location"])
    else:
        # Allow any non 4xx status code, as other failures could be temporary
        # and break the CI tests.
        if r.status_code >= 400 and r.status_code < 500:
            return url, message, code
        print()

    return None


def test_all_links():
    user_guide_path = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), DOC_FOLDER)
    )
    failed_urls = []

    for rst_file in glob.glob(f"{user_guide_path}/*.rst"):
        for url in get_all_links_from_file(rst_file):
            url_result = check_url_works(url)

            # If URL didn't work, check internal repos for pending release
            if url_result is not None:
                is_pre_release, internal_url = convert_to_internal(url)
                if is_pre_release:
                    url_result = check_url_works(internal_url)

            if url_result is not None:
                url, message, code = url_result
                failed_urls.append(f"{url}: {message} ({code})")

            print()

    no_failures = not failed_urls
    assert no_failures, "\n".join(failed_urls)
