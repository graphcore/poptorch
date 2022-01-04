#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
ctest --timeout uses SIGKILL to kill processes after they time out
unfortunately this prevents Linux from generating a core dump.

So instead this script sends a SIGABRT to the process when it times out which
will create a core dump.

The second part of the script is so that the test appears as "Timeout" in the
ctest results (instead of Aborted): unfortunately there is no way to mark a
test as "Timeout in ctest, this can only be done if ctest detects the timeout
itself.

In order to achieve this we set TIMEOUT_AFTER_MATCH "1;TEST_TIMEOUT on all the
tests in ctest: it means ctest will consider a test to have timed out
(and kill it) if it doesn't complete within 1 second of printing the string
TEST_TIMEOUT.
"""

import subprocess
import signal
import sys
import time
import os

# Assuming the ctest --timeout argument is set to the same value: we want this one
# to kick in first, so remove 60 seconds from it.
timeout = int(os.environ.get("POPTORCH_TEST_TIMEOUT", "1000")) - 60
# Run the command passed
with subprocess.Popen(sys.argv[1:]) as p:
    try:
        print("Setting timeout to %d seconds" % timeout, flush=True)
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print("Timeout after %d seconds" % timeout, flush=True)
        # Timeout: send an segmentation fault signal to generate a core dump.
        p.send_signal(signal.SIGSEGV)
        print("Waiting for aborted process...", flush=True)
        # Wait for the process to exit cleanly
        p.wait()
        # Signal to ctest it was a timeout
        print("TEST_TIMEOUT", flush=True)
        # give ctest some time to process the timeout
        time.sleep(60)
        print("ERROR: Shouldn't have reached this point", flush=True)
        # Note: in theory ctest should kill this process 1 second after TEST_TIMEOUT was printed.
    sys.exit(p.returncode)
