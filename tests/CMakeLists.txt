set(BUILD_TESTING ${RICCATI_BUILD_TESTING})

if (BUILD_TESTING)
# Only needed for testing
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
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      target_compile_options(${TEST_TARGET_NAME} PRIVATE -Wno-deprecated-declarations)
    endif()
    # Add include directories, link libraries, and register tests as before
    target_include_directories(${TEST_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${boost_SOURCE_DIR})
      target_link_libraries(${TEST_TARGET_NAME} gtest_main riccati
    Eigen3::Eigen)
    add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})
  endif()
  foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE) # Extract the filename without extension
    set(INDIVIDUAL_TEST_TARGET_NAME ${TEST_NAME}) # Create a unique target name

    # Add an executable for the individual test
    add_executable(${INDIVIDUAL_TEST_TARGET_NAME} ${TEST_SOURCE})
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${INDIVIDUAL_TEST_TARGET_NAME} PRIVATE -Wno-deprecated-declarations)
    endif()
    # Add include directories, link libraries, and register the individual test
    target_include_directories(${INDIVIDUAL_TEST_TARGET_NAME} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR} ${boost_SOURCE_DIR})
    target_link_libraries(${INDIVIDUAL_TEST_TARGET_NAME} gtest_main riccati Eigen3::Eigen)
    #add_test(NAME ${INDIVIDUAL_TEST_TARGET_NAME} COMMAND ${INDIVIDUAL_TEST_TARGET_NAME})
  endforeach()
endfunction()
message(STATUS "Building tests: ${BUILD_TESTING}")
enable_testing()
add_gtest_grouped_test(${CMAKE_CURRENT_SOURCE_DIR}/riccati_test)
endif()
message(STATUS "Building tests: DONE")
