cmake_minimum_required(VERSION 3.20.2)
project(
    riccati
    VERSION 1.3.0
    LANGUAGES C CXX)

include(FetchContent)
set(CMAKE_VERBOSE_MAKEFILE YES)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
cmake_policy(SET CMP0135 NEW)
cmake_policy(SET CMP0077 NEW)
if(MSVC)
    add_compile_options(/std:c++14)
endif()
set(CMAKE_CXX_EXTENSIONS NO)
if (CMAKE_BUILD_TYPE MATCHES Release)
#  set(CMAKE_VERBOSE_MAKEFILE NO)
else()
  set(CMAKE_VERBOSE_MAKEFILE YES)
endif()

# Build Types
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
    CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel tsan asan lsan msan ubsan"
    FORCE)

# ThreadSanitizer
set(CMAKE_C_FLAGS_TSAN
    "-fsanitize=thread -g -O1 -march=native -mtune=native"
    CACHE STRING "Flags used by the C compiler during ThreadSanitizer builds."
    FORCE)
set(CMAKE_CXX_FLAGS_TSAN
    "-fsanitize=thread -g -O1 -march=native -mtune=native"
    CACHE STRING "Flags used by the C++ compiler during ThreadSanitizer builds."
    FORCE)

# AddressSanitize
set(CMAKE_C_FLAGS_ASAN
    "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -Og -march=native -mtune=native"
    CACHE STRING "Flags used by the C compiler during AddressSanitizer builds."
    FORCE)
set(CMAKE_CXX_FLAGS_ASAN
    "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -Og -Wall -march=native -mtune=native"
    CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds."
    FORCE)

# LeakSanitizer
set(CMAKE_C_FLAGS_LSAN
    "-fsanitize=leak -fno-omit-frame-pointer -g -O1"
    CACHE STRING "Flags used by the C compiler during LeakSanitizer builds."
    FORCE)
set(CMAKE_CXX_FLAGS_LSAN
    "-fsanitize=leak -fno-omit-frame-pointer -ggdb3 -O1"
    CACHE STRING "Flags used by the C++ compiler during LeakSanitizer builds."
    FORCE)

set(CMAKE_CXX_FLAGS_DEBUG
    "-fno-omit-frame-pointer -ggdb3 -Og -DRICCATI_DEBUG=true"
    CACHE STRING "Flags used by the C++ compiler during Debug builds."
    FORCE)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE
      "-O3 -march=native -mtune=native -DRICCATI_DEBUG=false"
      CACHE STRING "Flags used by the C++ compiler during Release builds."
      FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake_deps)

set(RICCATI_BUILD_TESTING ${BUILD_TESTING} BOOL "Build the tests for the library")

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(
      -DNO_FPRINTF_OUTPUT
      -Wall)
endif()

set(riccati_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Allows upstream users to call fetch_content etc.
add_library(riccati INTERFACE)

target_include_directories(riccati PUBLIC INTERFACE
  ${riccati_INCLUDE_DIR}
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
