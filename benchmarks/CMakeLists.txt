include(ExternalProject)

# Add an executable for the individual test
add_executable(brenner237 ./brenner237.cpp)

# Add include directories, link libraries, and register the individual test
target_include_directories(brenner237 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(brenner237 riccati Eigen3::Eigen)
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fig1.py
  ${CMAKE_CURRENT_BINARY_DIR}/fig1.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/eq237.csv
  ${CMAKE_CURRENT_BINARY_DIR}/data/eq237.txt COPYONLY)

ExternalProject_Add(
  Phase
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext-codes/Phase-functions
  GIT_REPOSITORY https://github.com/fruzsinaagocs/Phase-functions
  GIT_TAG b8393be4815f6b50e731c3be6eedb4c050e3172f
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      "")


ExternalProject_Add(
  WKB
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext-codes/adaptive-WKB-marching-method
  GIT_REPOSITORY https://github.com/fruzsinaagocs/adaptive-WKB-marching-method
  GIT_TAG 7973953162f052ec942c6f2c144c3a5cf83cfece
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      "")



add_custom_command(
  OUTPUT table.csv
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fig1.py > table.csv
  DEPENDS
    brenner237
    WKB
    Phase
  VERBATIM)
  add_custom_target(figgen DEPENDS table.csv)
