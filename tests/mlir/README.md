# MLIR Pass Tests

This directory contains tests for PopTorch's custom MLIR passes.

The tests are written similarly to MLIR's standard [Check
tests](https://mlir.llvm.org/getting_started/TestingGuide/#check-tests), making
use of LLVM's [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html)
and [lit](https://llvm.org/docs/CommandGuide/lit.html) tools.

# Running

Make sure you're in a PopTorch buildenv. Then, to run the tests for one of the
MLIR files:

```sh
# Eg. to run the tests in remove_overwrite_test.mlir
poptorch_view/build $ ctest [--verbose] -R remove_overwrite
```

*(Note: `-R` is a regex match)*

Or, to run all of the MLIR pass tests:

```sh
poptorch_view/build $ ctest -L mlir
```

# Writing

In general, look to the existing tests for examples.

One important thing, however, is to make sure your test files end in
`_test.mlir`, so they're picked up by ctest.

Within this directory, you can create whatever subdirectory structure you want;
just make sure the filenames are globally unique.

As with the Python tests, cases are generated for ctest by
[generate_test_file.py](../generate_test_file.py).

