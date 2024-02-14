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

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -march=native -mtune=native -DRICCATI_DEBUG=false"
    CACHE STRING "Flags used by the C++ compiler during Release builds."
    FORCE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Externally provided libraries
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

option(RICCATI_BUILD_TESTING "Build the tests for Riccati" ON)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped

# Only needed for testing
if (RICCATI_BUILD_TESTING)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(
  boost
  URL https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
)

FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
  FetchContent_Populate(boost)
  set(boost_INCLUDE_DIR ${boost_SOURCE_DIR})
endif()
FetchContent_MakeAvailable(Eigen googletest)
endif()

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


# Define a function to simplify adding grouped tests
function(add_gtest_grouped_test test_directory)
  # Create a test target name based on the directory
  file(RELATIVE_PATH TEST_DIRECTORY_RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${test_directory})
  string(REPLACE "/" "_" TEST_TARGET_NAME ${TEST_DIRECTORY_RELATIVE})

  # Collect all .cpp files in the directory
  file(GLOB TEST_SOURCES "${TEST_DIRECTORY_RELATIVE}/*.cpp")
  if(TEST_SOURCES)
    message("STATUS" "Adding grouped test: ${TEST_TARGET_NAME}")
    # Add an executable for the grouped tests
    add_executable(${TEST_TARGET_NAME} ${TEST_SOURCES})
    # Add include directories, link libraries, and register tests as before
    target_include_directories(${TEST_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${boost_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests/riccati)
    target_link_libraries(${TEST_TARGET_NAME} gtest_main riccati Eigen3::Eigen)
    add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})
  endif()
  foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE) # Extract the filename without extension
    set(INDIVIDUAL_TEST_TARGET_NAME ${TEST_NAME}) # Create a unique target name

    # Add an executable for the individual test
    add_executable(${INDIVIDUAL_TEST_TARGET_NAME} ${TEST_SOURCE})

    # Add include directories, link libraries, and register the individual test
    target_include_directories(${INDIVIDUAL_TEST_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/riccati ${boost_INCLUDE_DIR})
    target_link_libraries(${INDIVIDUAL_TEST_TARGET_NAME} gtest_main riccati Eigen3::Eigen)
    add_test(NAME ${INDIVIDUAL_TEST_TARGET_NAME} COMMAND ${INDIVIDUAL_TEST_TARGET_NAME})
  endforeach()
endfunction()
message(STATUS "Building tests: ${RICCATI_BUILD_TESTING}")
enable_testing()
if (RICCATI_BUILD_TESTING)
  add_gtest_grouped_test(${CMAKE_CURRENT_SOURCE_DIR}/tests/riccati_test)
endif()
message(STATUS "Building tests: DONE")
