set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(
  boost
  URL https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
)
FetchContent_Declare(googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG main)
FetchContent_MakeAvailable(googletest boost)

add_executable(airy_eq ${CMAKE_CURRENT_SOURCE_DIR}/airy.cpp)
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(airy_eq PRIVATE -Wno-deprecated-declarations)
endif()
# Add include directories, link libraries, and register tests as before
target_include_directories(airy_eq PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${boost_SOURCE_DIR})
target_link_libraries(airy_eq riccati Eigen3::Eigen)
