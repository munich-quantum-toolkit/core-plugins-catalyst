# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set required cmake version
cmake_minimum_required(VERSION 3.24...4.2)

project(
  core-plugins-catalyst
  LANGUAGES C CXX
  DESCRIPTION "An MLIR-plugin for connecting MQT Core with Xanadu's Catalyst")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_custom_target(core-plugins-catalyst COMMAND ${CMAKE_COMMAND} -E echo
                                                "Building core-plugins-catalyst")
