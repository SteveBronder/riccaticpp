#ifndef INCLUDE_RICCATI_LOGGER_HPP
#define INCLUDE_RICCATI_LOGGER_HPP

#include <riccati/macros.hpp>
#include <riccati/utils.hpp>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <array>
#include <utility>
#include <stdexcept>
#include <iostream>

#ifdef RICCATI_DEBUG
#ifndef RICCATI_DEBUG_VAL
#define RICCATI_DEBUG_VAL true
#endif
#else
#ifndef RICCATI_DEBUG_VAL
#define RICCATI_DEBUG_VAL false
#endif
#endif
namespace riccati {

/**
 * @brief Enumeration of different log levels for logging.
 */
enum class LogLevel {
  ERROR = 0,    // Error messages.
  WARNING = 1,  // Warning messages.
  INFO = 2,     // General information messages.
  DEBUG = 3,    // Detailed debug information.
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
inline constexpr auto get_idx(const LogInfo& log_info) {
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
    throw std::invalid_argument("Invalid LogInfo key!");
    return 0;
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
  inline void log(Types&&... args) {
    this->underlying().template log<Level>(std::forward<Types>(args)...);
  }
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
template <typename Ptr>
class PtrLogger : public LoggerBase<PtrLogger<Ptr>> {
 public:
  /**
   * @brief The output stream for logging messages.
   */
  Ptr output_{};

  /**
   * @brief Default constructor.
   */
  PtrLogger() = default;

  /**
   * @brief Constructor with a custom output stream.
   *
   * @param output The output stream to use for logging.
   */
  template <typename Stream, typename StreamDeleter>
  RICCATI_NO_INLINE explicit PtrLogger(
      std::unique_ptr<Stream, StreamDeleter>&& output)
      : output_(std::move(output)) {}
  template <typename Stream>
  RICCATI_NO_INLINE explicit PtrLogger(const std::shared_ptr<Stream>& output)
      : output_(output) {}

  template <LogLevel Level, typename T, typename... Args>
  inline auto concat_string(std::stringstream& stream, T&& arg,
                            Args&&... args) {
    if constexpr (is_pair_v<T>) {
      if constexpr (std::is_same_v<LogInfo,
                                   std::decay_t<decltype(arg.first)>>) {
        stream << to_string(arg.first) << "=";
      } else {
        stream << "[" << arg.first << "=";
      }
      if constexpr (is_complex_v<T>) {
        stream << "(" << arg.second.real() << ", " << arg.second.imag() << ")";
      } else {
        stream << arg.second;
      }
      stream << "]";
    } else if constexpr (is_complex_v<T>) {
      stream << "(" << arg.real() << ", " << arg.imag() << ")";
    } else {
      stream << "[" << std::forward<T>(arg) << "]";
    }
    if constexpr (sizeof...(Args) == 0) {
      return stream.str();
    } else {
      return concat_string<Level>(stream, std::forward<Args>(args)...);
    }
  }

  template <LogLevel Level, typename T>
  inline void log(LogLevel UserLevel, T&& msg) {
    if constexpr (!RICCATI_DEBUG_VAL && Level == LogLevel::DEBUG) {
      return;
    }
    if (Level > UserLevel) {
      return;
    }
    std::string full_msg = log_level<Level>() + time_mi() + "[";
    if constexpr (std::is_convertible<std::decay_t<T>, std::string>::value) {
      full_msg += msg;
    } else {
      std::stringstream stream;
      stream << std::setprecision(18);
      full_msg += concat_string<Level>(stream, std::forward<T>(msg));
    }
    full_msg += std::string("]");
    *output_ << full_msg + "\n";
  }

  template <LogLevel Level, typename T, typename... Args,
            std::enable_if_t<sizeof...(Args) != 0>* = nullptr>
  inline void log(LogLevel UserLevel, T&& arg, Args&&... args) {
    if constexpr (!RICCATI_DEBUG_VAL && Level == LogLevel::DEBUG) {
      return;
    }
    if (Level > UserLevel) {
      return;
    }
    std::stringstream stream;
    stream << std::setprecision(18);
    this->log<Level>(UserLevel,
                     concat_string<Level>(stream, std::forward<T>(arg),
                                          std::forward<Args>(args)...));
  }
};

template <typename Stream, typename StreamDeleter = std::default_delete<Stream>>
using DefaultLogger = PtrLogger<std::unique_ptr<Stream, StreamDeleter>>;
template <typename Stream>
using SharedLogger = PtrLogger<std::shared_ptr<Stream>>;

}  // namespace riccati

#endif
