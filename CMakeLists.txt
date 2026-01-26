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
set(LLVM_ENABLE_RTTI ON)
set(LLVM_ENABLE_EH ON)

# MLIR must be installed on the system
find_package(MLIR REQUIRED CONFIG)

# Add the paths to the MLIR and LLVM CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

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
