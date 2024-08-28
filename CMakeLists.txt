cmake_minimum_required(VERSION 3.20.2)
project(
    riccati
    VERSION 0.1.0
    LANGUAGES C CXX)

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED NO)
cmake_policy(SET CMP0135 NEW)
cmake_policy(SET CMP0077 NEW)
set(CMAKE_CXX_EXTENSIONS NO)
if (CMAKE_BUILD_TYPE MATCHES Release)
#  set(CMAKE_VERBOSE_MAKEFILE NO)
else()
  set(CMAKE_VERBOSE_MAKEFILE YES)
endif()

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(-Wno-deprecated-declarations)
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
    "-fno-omit-frame-pointer -glldb -Og -DRICCATI_DEBUG=true"
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


install(DIRECTORY riccati DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Devel)

install(TARGETS riccati EXPORT riccatiTargets)

include (CMakePackageConfigHelpers)
configure_package_config_file (
${CMAKE_CURRENT_SOURCE_DIR}/cmake/riccatiConfig.cmake.in
${CMAKE_CURRENT_BINARY_DIR}/riccatiConfig.cmake
INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
NO_SET_AND_CHECK_MACRO # Eigen does not provide legacy style defines
NO_CHECK_REQUIRED_COMPONENTS_MACRO # Eigen does not provide components
)

# NOTE Remove the first code path once the minimum required CMake version is
# bumped to 3.14 or above.
if (CMAKE_VERSION VERSION_LESS 3.14)
# Remove CMAKE_SIZEOF_VOID_P from riccatiConfigVersion.cmake since Eigen does
# not depend on architecture specific settings or libraries. More
# specifically, an riccatiConfig.cmake generated from a 64 bit target can be
# used for 32 bit targets as well (and vice versa).
set (_riccati_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset (CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file (riccatiConfigVersion.cmake)
set (CMAKE_SIZEOF_VOID_P ${_riccati_CMAKE_SIZEOF_VOID_P})
else (CMAKE_VERSION VERSION_LESS 3.14)
write_basic_package_version_file (riccatiConfigVersion.cmake
                                    ARCH_INDEPENDENT)
endif (CMAKE_VERSION VERSION_LESS 3.14)

# The Eigen target will be located in the riccati namespace. Other CMake
# targets can refer to it using riccati::Eigen.
export (TARGETS eigen NAMESPACE riccati:: FILE riccatiTargets.cmake)
# Export riccati package to CMake registry such that it can be easily found by
# CMake even if it has not been installed to a standard directory.
export (PACKAGE riccati)

install (EXPORT riccatiTargets NAMESPACE riccati:: DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

install (FILES ${CMAKE_CURRENT_BINARY_DIR}/riccatiConfig.cmake
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})




if (RICCATI_BUILD_TESTING)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

if (RICCATI_BUILD_BENCHMARKS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
endif()

# Check if Doxygen is installed
if (RICCATI_BUILD_DOXYGEN)
    set(Eigen_INCLUDE_DIR "${riccati_SOURCE_DIR}")
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        # Set input and output files
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile)

        # Request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message(STATUS "Doxygen build started")

        # Note: do not put 'ALL' keyword since you probably don't want documentation
        # to be generated every time you build the project.
        add_custom_target( doc_doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )
    else()
        message("Doxygen need to be installed to generate the doxygen documentation")
    endif()
endif()

if (RICCATI_BUILD_PYTHON) 
    # Find the module development requirements (requires FindPython from 3.17 or
    # scikit-build-core's built-in backport)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    # Add a library using FindPython's tooling (pybind11 also provides a helper like
    # this)
    python_add_library(pyriccaticpp MODULE src/pymain.cpp WITH_SOABI)
    target_link_libraries(pyriccaticpp PRIVATE pybind11::headers riccati riccati::Eigen)

    # This is passing in the version as a define just as an example
    target_compile_definitions(pyriccaticpp PRIVATE VERSION_INFO=${PROJECT_VERSION})
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(pyriccaticpp PRIVATE -Wno-deprecated-declarations)
    endif()
    #target_compile_options(pyriccaticpp PRIVATE -Og -glldb)
    # The install directory is the output (wheel) directory
    install(TARGETS pyriccaticpp DESTINATION pyriccaticpp)
endif()
