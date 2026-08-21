# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set required cmake version
cmake_minimum_required(VERSION 3.24...4.4)

project(
  mqt-core-plugins-catalyst
  LANGUAGES C CXX
  DESCRIPTION "An MLIR-plugin for connecting MQT Core with Xanadu's Catalyst")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# prefer venv Python
set(Python_FIND_VIRTUALENV
    FIRST
    CACHE STRING "Give precedence to virtualenvs when searching for Python")
# Find the target Python before bootstrapping the native dependencies.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
                                             ${SKBUILD_SABI_COMPONENT})

# MLIR must be installed by scripts/bootstrap.sh.
include(cmake/ToolchainPins.cmake)
if(NOT MLIR_DIR)
  set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.cache/mlir/${MQT_LLVM_REVISION}/lib/cmake/mlir")
endif()
get_filename_component(MQT_PINNED_LLVM_DIR "${MLIR_DIR}/../llvm" ABSOLUTE)
if(LLVM_DIR AND NOT LLVM_DIR STREQUAL MQT_PINNED_LLVM_DIR)
  message(
    FATAL_ERROR "LLVM_DIR must match MLIR_DIR. Expected ${MQT_PINNED_LLVM_DIR}, found ${LLVM_DIR}.")
endif()
set(LLVM_DIR
    "${MQT_PINNED_LLVM_DIR}"
    CACHE PATH "LLVM package matching the pinned MLIR package" FORCE)

# Let the embedded MQT Core project discover MLIR first. Its pinned revision attaches documentation
# commands to the `mlir-doc` target in the directory where MLIR is discovered, as required by CMake.
include(cmake/ExternalDependencies.cmake)

# Import MLIR in this directory as well for the plugin's own targets.
find_package(MLIR REQUIRED CONFIG)

set(MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

# Add the paths to the MLIR and LLVM CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(LLVM_ENABLE_RTTI ON)
set(LLVM_ENABLE_EH ON)

# Include the TableGen, LLVM and MLIR CMake modules.
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
