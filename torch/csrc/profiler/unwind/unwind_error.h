#pragma once
#include <c10/util/Optional.h>
#include <fmt/format.h>
#include <stdexcept>

namespace torch::unwind {

struct UnwindError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define UNWIND_CHECK(cond, fmtstring, ...)                          \
  do {                                                              \
    if (!(cond)) {                                                  \
      throw unwind::UnwindError(fmt::format(                        \
          "{}:{}: " fmtstring, __FILE__, __LINE__, ##__VA_ARGS__)); \
    }                                                               \
  } while (0)

// #define LOG_INFO(...) fmt::print(__VA_ARGS__)
#define LOG_INFO(...)

// #define PRINT_INST(...) LOG_INFO(__VA_ARGS__)
#define PRINT_INST(...)

// #define PRINT_LINE_TABLE(...) LOG_INFO(__VA_ARGS__)
#define PRINT_LINE_TABLE(...)

using c10::optional; // NOLINT

} // namespace torch::unwind
