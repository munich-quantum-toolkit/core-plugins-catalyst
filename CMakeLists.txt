# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set required cmake version
cmake_minimum_required(VERSION 3.24...4.2)

project(
  mqt-core-plugins-catalyst
  LANGUAGES C CXX
  DESCRIPTION "An MLIR-plugin for connecting MQT Core with Xanadu's Catalyst")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# top-level call to find Python
find_package(Python 3.11 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

include(cmake/ExternalDependencies.cmake)

set(MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")

# LLVM / MLIR must be installed on the system
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# LLVM specific setup required for the plugin to work properly
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_subdirectory(include)
add_subdirectory(lib)
