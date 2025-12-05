# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Declare all external dependencies and make sure that they are available.

include(FetchContent)

# Find Python first - scikit-build-core will set the hints for us
find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)

set(CATALYST_VERSION 0.13.0)

if(DEFINED Python3_EXECUTABLE AND Python3_EXECUTABLE)
  # Check if the pennylane-catalyst package is installed in the python environment.
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import catalyst; print(catalyst.__version__)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE FOUND_CATALYST_VERSION)
  if(FOUND_CATALYST_VERSION)
    message(STATUS "Found pennylane-catalyst ${FOUND_CATALYST_VERSION} in python environment.")
    # Check if the version is compatible.
    if(FOUND_CATALYST_VERSION VERSION_LESS ${CATALYST_VERSION})
      message(
        WARNING
          "pennylane-catalyst version ${FOUND_CATALYST_VERSION} in python environment is not compatible."
      )
    else()
      # Detect the installed catalyst include files.
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import catalyst.utils.runtime_environment as c; print(c.get_include_path())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE CATALYST_INCLUDE_DIRS)

      message(STATUS "Catalyst include path resolved to: ${CATALYST_INCLUDE_DIRS}")

      string(FIND "${CATALYST_INCLUDE_DIRS}" "site-packages" SITEPKG_IDX)

      if(SITEPKG_IDX EQUAL -1)
        # In case of a installation from source Assume include path looks like: <root>/mlir/include
        # Derive <root>/mlir/build/include and <root>/mlir/build/lib/cmake/catalyst
        get_filename_component(CATALYST_MLIR_ROOT "${CATALYST_INCLUDE_DIRS}/.." ABSOLUTE)
        set(CATALYST_BUILD_DIR "${CATALYST_MLIR_ROOT}/build")
        set(CATALYST_BUILD_INCLUDE_DIR "${CATALYST_BUILD_DIR}/include")
        set(Catalyst_DIR "${CATALYST_BUILD_DIR}/lib/cmake/catalyst")

        include_directories("${CATALYST_INCLUDE_DIRS}")
        include_directories("${CATALYST_BUILD_INCLUDE_DIR}")
      else()
        # In case of an installation from PyPI, the include path looks like:
        # <root>/site-packages/catalyst/include Derive the site-packages root and add it to the
        # CMAKE_PREFIX_PATH.
        get_filename_component(CATALYST_SP_ROOT "${CATALYST_INCLUDE_DIRS}/../.." ABSOLUTE)
        list(APPEND CMAKE_PREFIX_PATH "${CATALYST_SP_ROOT}")
        message(STATUS "Adding Catalyst site-packages to CMAKE_PREFIX_PATH: ${CATALYST_SP_ROOT}")
      endif()

    endif()
  else()
    # Unfortunately, the download for an individual package cannot be turned off. To avoid
    # downloading the entire package, we use `find_package` instead.
    find_package(Catalyst ${CATALYST_VERSION} REQUIRED)
  endif()

  if(NOT CATALYST_INCLUDE_DIRS)
    message(
      FATAL_ERROR
        "The include directory of the pennylane-catalyst package could not be retrieved. Please ensure that the catalyst is installed correctly."
    )
  endif()

else()
  find_package(Catalyst ${CATALYST_VERSION} QUIET)
  if(NOT Catalyst_FOUND)
    message(
      FATAL_ERROR
        "Python3 interpreter not found and Catalyst not discoverable. Either set Python3_EXECUTABLE for in-env detection or provide Catalyst via CMAKE_PREFIX_PATH."
    )
  endif()
endif()

# cmake-format: off
set(MQT_CORE_MINIMUM_VERSION 3.1.0
    CACHE STRING "MQT Core minimum version")
set(MQT_CORE_VERSION 3.3.3
    CACHE STRING "MQT Core version")
set(MQT_CORE_REV "v3.3.3"
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

# Exclude mqt-core from install to avoid trying to install libraries we don't build
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE OPT_IN)

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})

# Exclude mqt-core directory from install target
if(mqt-core_SOURCE_DIR)
  set_property(DIRECTORY ${mqt-core_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
endif()
