# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Declare all external dependencies and make sure that they are available.

include(FetchContent)

# Verify that MLIR came from the exact setup-mlir installation used by the bootstrap script. LLVM
# does not encode its Git revision in LLVMConfig.cmake, so setup writes this marker after
# downloading the pinned archive.
get_filename_component(MQT_MLIR_INSTALL_PREFIX "${MLIR_DIR}/../../.." REALPATH)
set(MQT_LLVM_REVISION_FILE "${MQT_MLIR_INSTALL_PREFIX}/.llvm-revision")
if(NOT EXISTS "${MQT_LLVM_REVISION_FILE}")
  message(
    FATAL_ERROR
      "The MLIR installation has no revision marker. Run scripts/bootstrap.sh to install the pinned toolchain."
  )
endif()
file(READ "${MQT_LLVM_REVISION_FILE}" FOUND_LLVM_REVISION)
string(STRIP "${FOUND_LLVM_REVISION}" FOUND_LLVM_REVISION)
if(NOT FOUND_LLVM_REVISION STREQUAL "${MQT_LLVM_REVISION}")
  message(
    FATAL_ERROR
      "LLVM revision mismatch: expected ${MQT_LLVM_REVISION}, found ${FOUND_LLVM_REVISION}.")
endif()

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

# Fetch MQT Core from the exact compatible revision. With no FIND_PACKAGE_ARGS on this declaration,
# FetchContent cannot substitute an installed Core package.
FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
  GIT_TAG ${MQT_CORE_REVISION})
FetchContent_MakeAvailable(mqt-core)

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${mqt-core_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE MQT_CORE_REVISION_RESULT
  OUTPUT_VARIABLE FOUND_MQT_CORE_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT MQT_CORE_REVISION_RESULT EQUAL 0 OR NOT FOUND_MQT_CORE_REVISION STREQUAL
                                           "${MQT_CORE_REVISION}")
  message(
    FATAL_ERROR
      "MQT Core revision mismatch: expected ${MQT_CORE_REVISION}, found ${FOUND_MQT_CORE_REVISION}."
  )
endif()

# Exclude mqt-core directory from install target
if(mqt-core_SOURCE_DIR)
  set_property(DIRECTORY ${mqt-core_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
endif()

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import catalyst; from catalyst.utils.runtime_environment import get_include_path; print(catalyst.__version__); print(catalyst.__revision__); print(get_include_path())"
  RESULT_VARIABLE CATALYST_METADATA_RESULT
  OUTPUT_VARIABLE CATALYST_METADATA
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT CATALYST_METADATA_RESULT EQUAL 0)
  message(
    FATAL_ERROR "The exact Catalyst source build is not installed. Run scripts/bootstrap.sh first.")
endif()

string(REPLACE "\n" ";" CATALYST_METADATA "${CATALYST_METADATA}")
list(LENGTH CATALYST_METADATA CATALYST_METADATA_LENGTH)
if(NOT CATALYST_METADATA_LENGTH EQUAL 3)
  message(FATAL_ERROR "Could not read Catalyst version, revision, and include path.")
endif()
list(GET CATALYST_METADATA 0 FOUND_CATALYST_VERSION)
list(GET CATALYST_METADATA 1 FOUND_CATALYST_REVISION)
list(GET CATALYST_METADATA 2 CATALYST_INCLUDE_DIRS)

if(NOT FOUND_CATALYST_VERSION STREQUAL "${MQT_CATALYST_VERSION}")
  message(
    FATAL_ERROR
      "Catalyst version mismatch: expected ${MQT_CATALYST_VERSION}, found ${FOUND_CATALYST_VERSION}."
  )
endif()
if(NOT FOUND_CATALYST_REVISION STREQUAL "${MQT_CATALYST_REVISION}")
  message(
    FATAL_ERROR
      "Catalyst revision mismatch: expected ${MQT_CATALYST_REVISION}, found ${FOUND_CATALYST_REVISION}."
  )
endif()
set(CATALYST_SOURCE_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.cache/catalyst/${MQT_CATALYST_REVISION}/mlir/include")
set(CATALYST_BUILD_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.cache/catalyst/${MQT_CATALYST_REVISION}/mlir/build/include")
if(NOT IS_DIRECTORY "${CATALYST_INCLUDE_DIRS}"
   OR NOT IS_DIRECTORY "${CATALYST_SOURCE_INCLUDE_DIR}"
   OR NOT IS_DIRECTORY "${CATALYST_BUILD_INCLUDE_DIR}")
  message(
    FATAL_ERROR
      "Catalyst installed, source, and generated include directories must all exist. Run scripts/bootstrap.sh first."
  )
endif()

# Catalyst's wheel contains the public Quantum headers, while some of their transitive Catalyst
# headers and generated interface files remain in the exact source build. All three directories
# belong to the verified revision.
set(CATALYST_INCLUDE_DIRS ${CATALYST_INCLUDE_DIRS} ${CATALYST_SOURCE_INCLUDE_DIR}
                          ${CATALYST_BUILD_INCLUDE_DIR})

message(
  STATUS
    "Using Catalyst ${FOUND_CATALYST_VERSION} (${FOUND_CATALYST_REVISION}), MQT Core ${FOUND_MQT_CORE_REVISION}, and LLVM ${FOUND_LLVM_REVISION}"
)
