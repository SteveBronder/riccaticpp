#include <riccati/logger.hpp>
#include <gtest/gtest.h>
#include <sstream>

namespace {

using namespace riccati;

/**
 * @brief Test the to_string function for LogInfo enum.
 */
TEST(RiccatiLoggerTest, ToStringFunction) {
  EXPECT_STREQ(to_string(LogInfo::CHEBNODES), "chebyshev_nodes");
  EXPECT_STREQ(to_string(LogInfo::CHEBSTEP), "chebyshev_steps");
  EXPECT_STREQ(to_string(LogInfo::CHEBITS), "chebyshev_iterates");
  EXPECT_STREQ(to_string(LogInfo::LS), "linear_system_solves");
  EXPECT_STREQ(to_string(LogInfo::RICCSTEP), "riccati_steps");
}

/**
 * @brief Test the get_idx function for LogInfo enum.
 */
TEST(RiccatiLoggerTest, GetIdxFunction) {
  EXPECT_EQ(get_idx(LogInfo::CHEBNODES), 0);
  EXPECT_EQ(get_idx(LogInfo::CHEBSTEP), 1);
  EXPECT_EQ(get_idx(LogInfo::CHEBITS), 2);
  EXPECT_EQ(get_idx(LogInfo::LS), 3);
  EXPECT_EQ(get_idx(LogInfo::RICCSTEP), 4);
  EXPECT_THROW(get_idx(static_cast<LogInfo>(-1)),
               std::invalid_argument);  // Test for invalid input
}

/**
 * @brief Test the log_level function template for LogLevel enum.
 */
TEST(RiccatiLoggerTest, LogLevelFunction) {
#ifdef RICCATI_DEBUG
  EXPECT_STREQ(log_level<LogLevel::DEBUG>(), "[DEBUG]");
#endif
  EXPECT_STREQ(log_level<LogLevel::INFO>(), "[INFO]");
  EXPECT_STREQ(log_level<LogLevel::WARNING>(), "[WARNING]");
  EXPECT_STREQ(log_level<LogLevel::ERROR>(), "[ERROR]");
}

/**
 * @brief Test the EmptyLogger class.
 */
TEST(RiccatiLoggerTest, EmptyLoggerTest) {
  EmptyLogger logger;

  // Test that logging does not throw and does nothing
  EXPECT_NO_THROW(
      logger.log<LogLevel::INFO>(LogLevel::DEBUG, "This is a test"));
}

/**
 * @brief Test the DefaultLogger class for logging functionality.
 */
TEST(RiccatiLoggerTest, DefaultLoggerLoggingTest) {
  // Use a stringstream to capture the output
  std::stringstream ss;
  DefaultLogger<std::stringstream> logger{
      std::make_unique<std::stringstream>()};

  // Log an INFO message
  logger.log<LogLevel::INFO>(LogLevel::DEBUG, "Test message");

  // Check that the output contains the expected log level and message
  std::string output = logger.output_->str();
  EXPECT_NE(output.find("[INFO]"), std::string::npos);
  EXPECT_NE(output.find("Test message"), std::string::npos);

  // Clear the stream for the next test
  logger.output_->str("");
  logger.output_->clear();

  // Log an ERROR message
  logger.log<LogLevel::ERROR>(LogLevel::DEBUG, "Error occurred");

  // Check that the output contains the expected log level and message
  output = logger.output_->str();
  EXPECT_NE(output.find("[ERROR]"), std::string::npos);
  EXPECT_NE(output.find("Error occurred"), std::string::npos);
}

#ifndef RICCATI_DEBUG
/**
 * @brief Test that DEBUG logs are omitted when RICCATI_DEBUG is not defined.
 */
TEST(RiccatiLoggerTest, DefaultLoggerDebugLoggingDisabled) {
  // Use a stringstream to capture the output
  DefaultLogger<std::stringstream> logger{
      std::make_unique<std::stringstream>()};

  // Log a DEBUG message
  logger.log<LogLevel::DEBUG>(LogLevel::DEBUG, "Debug message");
  // Check that the output is empty (since RICCATI_DEBUG is not defined)
  std::string output = logger.output_->str();
  EXPECT_EQ(output.find("[DEBUG]"), std::string::npos);
}
#endif
#ifdef RICCATI_DEBUG
/**
 * @brief Test that DEBUG logs are included when RICCATI_DEBUG is defined.
 */
TEST(RiccatiLoggerTest, DefaultLoggerDebugLoggingEnabled) {
  // Use a stringstream to capture the output
  DefaultLogger<std::stringstream> logger{
      std::make_unique<std::stringstream>()};

  // Log a DEBUG message
  logger.log<LogLevel::DEBUG>(LogLevel::DEBUG, "Debug message");

  // Check that the output contains the DEBUG message
  std::string output = logger.output_->str();
  EXPECT_NE(output.find("[DEBUG]"), std::string::npos);
  EXPECT_NE(output.find("Debug message"), std::string::npos);
}
#endif

/**
 * @brief Test move assignment of DefaultLogger.
 */
TEST(RiccatiLoggerTest, DefaultLoggerMoveAssignment) {
  // Use stringstreams to capture outputs
  DefaultLogger<std::stringstream> logger1{
      std::make_unique<std::stringstream>()};
  DefaultLogger<std::stringstream> logger2{
      std::make_unique<std::stringstream>()};

  logger1.log<LogLevel::INFO>(LogLevel::DEBUG, "Logger1 message");

  // Move logger1 into logger2
  logger2 = std::move(logger1);

  std::string output = logger2.output_->str();
  EXPECT_NE(output.find("Logger1 message"), std::string::npos);
  EXPECT_FALSE(bool(logger1.output_));
}

/**
 * @brief Test that logging functions handle empty messages gracefully.
 */
TEST(RiccatiLoggerTest, DefaultLoggerEmptyMessage) {
  // Use a stringstream to capture the output
  DefaultLogger<std::stringstream> logger{
      std::make_unique<std::stringstream>()};

  // Log an empty message
  logger.log<LogLevel::INFO>(LogLevel::DEBUG, "");

  // Check that the output contains the log level but no message
  std::string output = logger.output_->str();
  EXPECT_NE(output.find("[INFO]"), std::string::npos);
}

}  // namespace
