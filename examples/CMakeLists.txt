
cmake_minimum_required(VERSION 3.20.2)
project(
    riccatiex
    VERSION 0.1.0
    LANGUAGES C CXX)

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED NO)
cmake_policy(SET CMP0135 NEW)
cmake_policy(SET CMP0077 NEW)


set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(
  boost
  URL https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
)
FetchContent_MakeAvailable(boost)

include(FetchContent)

FetchContent_Declare(
  riccaticpp
  https://github.com/SteveBronder/riccaticpp
  examplecheck # Use a specific version or commit
)
FetchContent_MakeAvailable(riccaticpp)

# For your target

add_executable(airy_eq ${CMAKE_CURRENT_SOURCE_DIR}/airy.cpp)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(airy_eq PRIVATE -Wno-deprecated-declarations)
endif()
# Add include directories, link libraries, and register tests as before
target_include_directories(airy_eq PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${boost_SOURCE_DIR})
target_link_libraries(airy_eq riccati Eigen3::Eigen)
