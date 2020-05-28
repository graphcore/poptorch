// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExceptionTest

#include <poptorch/error.hpp>

#include <boost/test/unit_test.hpp>

void test_function1() { throw poptorch::error("Test {} {} {}", 1, 2, 3); }

BOOST_AUTO_TEST_CASE(LoggingTest1) {

  try {
    test_function1();
    BOOST_TEST(false);
  } catch (const poptorch::error &e) {
    BOOST_CHECK(e.what() == std::string("Test 1 2 3"));
  }
}

void test_function2() { throw poptorch::error("Test {} {} {}", 1, 2); }

BOOST_AUTO_TEST_CASE(LoggingTest2) {

  try {
    test_function2();
    BOOST_TEST(false);
  } catch (const poptorch::error &e) {
    BOOST_CHECK(
        e.what() ==
        std::string(
            "Poptorch exception format error argument index out of range"));
  }
}
