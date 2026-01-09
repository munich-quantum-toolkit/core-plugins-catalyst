# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Declare all external dependencies and make sure that they are available.

include(FetchContent)

# cmake-format: off
set(MQT_CORE_MINIMUM_VERSION 3.3.3
    CACHE STRING "MQT Core minimum version")
set(MQT_CORE_VERSION 3.3.3
    CACHE STRING "MQT Core version")
set(MQT_CORE_REV "8c9f6ab24968401e450812fc0ff7d05b5ae07a63"
    CACHE STRING "MQT Core identifier (tag, branch or commit hash)")
set(MQT_CORE_REPO_OWNER "munich-quantum-toolkit"
    CACHE STRING "MQT Core repository owner (change when using a fork)")
# cmake-format: on

# Configure mqt-core options before fetching
set(BUILD_MQT_CORE_TESTS
    OFF
    CACHE BOOL "Build MQT Core tests")
set(BUILD_MQT_CORE_SHARED_LIBS
    OFF
    CACHE BOOL "Build MQT Core shared libraries")
set(BUILD_MQT_CORE_MLIR
    ON
    CACHE BOOL "Build MQT Core MLIR support")
set(BUILD_MQT_CORE_BINDINGS
    OFF
    CACHE BOOL "Build MQT Core Python bindings")
set(MQT_CORE_INSTALL
    OFF
    CACHE BOOL "Generate installation instructions for MQT Core")
set(CMAKE_POSITION_INDEPENDENT_CODE
    ON
    CACHE BOOL "Enable position independent code (PIC)")

# Fetch mqt-core from GitHub
FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/${MQT_CORE_REPO_OWNER}/core.git
  GIT_TAG ${MQT_CORE_REV}
  FIND_PACKAGE_ARGS ${MQT_CORE_MINIMUM_VERSION})
list(APPEND FETCH_PACKAGES mqt-core)

# Do not try to find mqt-core via find_package by default
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE OPT_IN)

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})

# Exclude mqt-core directory from install target
if(mqt-core_SOURCE_DIR)
  set_property(DIRECTORY ${mqt-core_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
endif()

set(CATALYST_VERSION 0.13.0)
find_package(Catalyst ${CATALYST_VERSION} QUIET)
if(Catalyst_FOUND)
  message(STATUS "Found Catalyst ${Catalyst_VERSION} via CMake find_package.")
  return()
endif()

if(NOT DEFINED Python_EXECUTABLE OR NOT Python_EXECUTABLE)
  message(FATAL_ERROR "Catalyst not found via find_package and no Python interpreter found.")
endif()

# Check if the pennylane-catalyst package is installed in the python environment.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import catalyst; print(catalyst.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE FOUND_CATALYST_VERSION)

if(NOT FOUND_CATALYST_VERSION)
  message(
    FATAL_ERROR
      "pennylane-catalyst package not found in the python environment. Please install pennylane-catalyst >= ${CATALYST_VERSION}."
  )
endif()

if(FOUND_CATALYST_VERSION VERSION_LESS ${CATALYST_VERSION})
  message(
    FATAL_ERROR
      "pennylane-catalyst version ${FOUND_CATALYST_VERSION} found in the python environment is not compatible. Please install pennylane-catalyst >= ${CATALYST_VERSION}."
  )
endif()

# Detect the installed catalyst include files.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import catalyst.utils.runtime_environment as c; print(c.get_include_path())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE CATALYST_INCLUDE_DIRS)

if(NOT CATALYST_INCLUDE_DIRS)
  message(
    FATAL_ERROR
      "The include directory of the pennylane-catalyst package could not be retrieved. Please ensure that the catalyst is installed correctly."
  )
endif()

message(STATUS "Catalyst include path resolved to: ${CATALYST_INCLUDE_DIRS}")
