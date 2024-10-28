#ifndef INCLUDE_RICCATI_LOGGER_HPP
#define INCLUDE_RICCATI_LOGGER_HPP

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <array>
#include <utility>

namespace riccati {

/**
 * @brief Enumeration of different log levels for logging.
 */
enum class LogLevel {
  DEBUG,    // Detailed debug information.
  INFO,     // General information messages.
  WARNING,  // Warning messages.
  ERROR,    // Error messages.
  CRITICAL  // Critical error messages.
};

/**
 * @brief Enumeration of different logging information keys.
 */
enum class LogInfo {
  CHEBNODES,  // Chebyshev nodes.
  CHEBSTEP,   // Chebyshev steps.
  CHEBITS,    // Chebyshev iterates.
  LS,         // Linear system solves.
  RICCSTEP    // Riccati steps.
};

/**
 * @brief Converts a LogInfo enum value to its string representation.
 *
 * @param log_info The LogInfo enum value to convert.
 * @return The string representation of the LogInfo value.
 */
inline constexpr auto to_string(const LogInfo& log_info) noexcept {
  if (log_info == LogInfo::CHEBNODES) {
    return "chebyshev_nodes";
  } else if (log_info == LogInfo::CHEBSTEP) {
    return "chebyshev_steps";
  } else if (log_info == LogInfo::CHEBITS) {
    return "chebyshev_iterates";
  } else if (log_info == LogInfo::LS) {
    return "linear_system_solves";
  } else if (log_info == LogInfo::RICCSTEP) {
    return "riccati_steps";
  } else {
    return "UNKNOWN_INFO";
  }
}
/**
 * @brief Get the index value for the log info in the default logger
 */
inline constexpr auto get_idx(const LogInfo& log_info) noexcept {
  if (log_info == LogInfo::CHEBNODES) {
    return 0;
  } else if (log_info == LogInfo::CHEBSTEP) {
    return 1;
  } else if (log_info == LogInfo::CHEBITS) {
    return 2;
  } else if (log_info == LogInfo::LS) {
    return 3;
  } else if (log_info == LogInfo::RICCSTEP) {
    return 4;
  } else {
    return -1;
  }
}

/**
 * @brief Retrieves the string representation of a LogLevel value.
 *
 * @tparam Level The LogLevel value.
 * @return The string representation of the LogLevel.
 */
template <LogLevel Level>
inline constexpr auto log_level() noexcept {
  if constexpr (Level == LogLevel::DEBUG) {
    return "[DEBUG]";
  } else if constexpr (Level == LogLevel::INFO) {
    return "[INFO]";
  } else if constexpr (Level == LogLevel::WARNING) {
    return "[WARNING]";
  } else if constexpr (Level == LogLevel::ERROR) {
    return "[ERROR]";
  } else if constexpr (Level == LogLevel::CRITICAL) {
    return "[CRITICAL]";
  } else {
    static_assert(1, "Invalid LogLevel!");
  }
}

/**
 * @brief Base class template for loggers.
 *
 * This class provides a common interface for logging and updating log
 * information. All loggers in Riccati should inherit from this logger
 * using the CRTP form `class Logger : public LoggerBase<Derived>`.
 * See \ref `riccati::DefaultLogger` for an example.
 *
 * @tparam Derived The derived logger class.
 */
template <typename Derived>
class LoggerBase {
  inline Derived& underlying() noexcept { return static_cast<Derived&>(*this); }
  inline Derived const& underlying() const noexcept {
    return static_cast<Derived const&>(*this);
  }

 public:
  /**
   * @brief Logs a message with a specified log level.
   *
   * @tparam Level The log level.
   * @tparam Types Variadic template parameter pack for message arguments.
   * @param arg Message arguments to log.
   */
  template <LogLevel Level, typename... Types>
  void log(Types&&... args) {
    this->underlying().template log<Level>(std::forward<Types>(args)...);
  }

  /**
   * @brief Updates the logging information with a key-value pair.
   *
   * @tparam Types Variadic template parameter pack for value arguments.
   * @param key The key of the log information.
   * @param x The value(s) to update.
   */
  template <typename... Types>
  void update_info(const LogInfo& key, Types&&... args) {
    this->underlying().update_info(key, std::forward<Types>(args)...);
  }

  /**
   * @brief Retrieves the logging information.
   *
   * @return The logging information.
   */
  auto& get_info() { return this->underlying().get_info(); }
  const auto& get_info() const { return this->underlying().get_info(); }
};

/**
 * @brief A deleter that performs no operation.
 *
 * This is used with smart pointers where no deletion is required.
 */
struct deleter_noop {
  /**
   * @brief No-op function call operator.
   *
   * @tparam T The type of the pointer.
   * @param arg The pointer to which the deleter is applied.
   */
  template <typename T>
  constexpr void operator()(T* arg) const {}
};

/**
 * A logger class where all the operations are noops
 */
class EmptyLogger : public LoggerBase<EmptyLogger> {
  public:
  template <LogLevel Level, typename... Types>
  inline constexpr void log(Types&&... args) const noexcept {}
  static constexpr int ret_code = -1;

  /**
   * @brief Updates the logging information with a key-value pair.
   *
   * @tparam Types Variadic template parameter pack for value arguments.
   * @param key The key of the log information.
   * @param x The value(s) to update.
   */
  template <typename... Types>
  inline constexpr void update_info(const LogInfo& key, Types&&... args) const noexcept {
    return;
  }

  /**
   * @brief Noope
   *
   * @return a -1 to indicate this should not be used.
   */
  constexpr auto& get_info() { return ret_code;}
  constexpr const auto& get_info() const { return ret_code; }
};

/**
 * @brief A simple logger class template.
 *
 * This class provides basic logging functionality with customizable output
 * streams.
 *
 * @tparam Stream The type of the output stream.
 * @tparam StreamDeleter The deleter type for the output stream.
 */
template <typename Stream, typename StreamDeleter>
class DefaultLogger : public LoggerBase<DefaultLogger<Stream, StreamDeleter>> {
 public:
  /**
   * @brief A map storing logging information.
   */
  std::array<std::pair<LogInfo, std::size_t>, 5> info_{std::make_pair(LogInfo::CHEBNODES, 0),
                                                 std::make_pair(LogInfo::CHEBSTEP, 0),
                                                 std::make_pair(LogInfo::CHEBITS, 0),
                                                 std::make_pair(LogInfo::LS, 0),
                                                 std::make_pair(LogInfo::RICCSTEP, 0)};

  /**
   * @brief The output stream for logging messages.
   */
  std::unique_ptr<Stream, deleter_noop> output_{new Stream{}};

  /**
   * @brief Default constructor.
   */
  DefaultLogger() = default;

  /**
   * @brief Copy constructor (deleted).
   *
   * @param other The other DefaultLogger to copy from.
   */
  DefaultLogger(const DefaultLogger& /* other */) = delete;

  /**
   * @brief Constructor with a custom output stream.
   *
   * @param output The output stream to use for logging.
   */
  explicit DefaultLogger(std::unique_ptr<Stream, StreamDeleter>&& output)
      : output_(std::move(output)),
        info_({std::make_pair(LogInfo::CHEBNODES, 0),
                                                 std::make_pair(LogInfo::CHEBSTEP, 0),
                                                 std::make_pair(LogInfo::CHEBITS, 0),
                                                 std::make_pair(LogInfo::LS, 0),
                                                 std::make_pair(LogInfo::RICCSTEP, 0)}) {}

  /**
   * @brief Copy assignment operator (deleted).
   *
   * @param other The other DefaultLogger to copy from.
   * @return Reference to this DefaultLogger.
   */
  auto& operator=(const DefaultLogger& /* other */) = delete;

  /**
   * @brief Move assignment operator.
   *
   * @param other The other DefaultLogger to move from.
   * @return Reference to this DefaultLogger.
   */
  auto& operator=(DefaultLogger&& other) noexcept {
    this->info_ = std::move(other.info_);
    this->output_ = std::move(other.output_);
    return *this;
  }
  /**
   * @brief Logs a message with a specified log level.
   *
   * @tparam Level The log level.
   * @param msg The message to log.
   */
  template <LogLevel Level>
  inline void log(std::string_view msg) {
    std::string full_msg = log_level<Level>();
    const std::time_t now = std::time(nullptr);
    char buf[66];
    if (strftime(buf, sizeof(buf), "[%c][", std::localtime(&now))) {
      full_msg += buf;
    }
    full_msg += msg;
    full_msg += std::string("]");
    *output_ << full_msg + "\n";
  }

  /**
   * @brief Updates the logging information with a key-value pair.
   *
   * @param key The key of the log information.
   * @param value The value to update.
   */
  inline void update_info(const LogInfo& key, std::size_t value) noexcept {
    info_[get_idx(key)].second += value;
  }

  /**
   * @brief Retrieves the logging information.
   *
   * @return Reference to the logging information map.
   */
  inline auto& get_info() noexcept { return info_; }

  /**
   * @brief Retrieves the logging information (const version).
   *
   * @return Const reference to the logging information map.
   */
  inline const auto& get_info() const noexcept { return info_; }
};

}  // namespace riccati

#endif
