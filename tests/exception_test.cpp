// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExceptionTest

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

// Use header only variant due to ABI issues
#include <boost/test/included/unit_test.hpp>

void testFunction1() {
  logging::setLogLevel(logging::Level::Trace);
  throw logging::Error("Test 1 2 3");
}

BOOST_AUTO_TEST_CASE(LoggingTest1) {
  try {
    testFunction1();
    BOOST_TEST(false);
  } catch (const logging::Error &e) {
    BOOST_CHECK(e.what() == std::string("Test 1 2 3"));
  }
}

void testFunction2() {
  logging::setLogLevel(logging::Level::Trace);
  throw logging::Error("Test");
}

BOOST_AUTO_TEST_CASE(LoggingTest2) {
  try {
    testFunction2();
    BOOST_TEST(false);
  } catch (const logging::Error &e) {
    BOOST_CHECK(
        e.what() ==
        std::string(
            "Poptorch exception format error argument index out of range"));
  }
}
