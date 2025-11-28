# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set required cmake version
cmake_minimum_required(VERSION 3.24...4.0)

project(
  core-plugins-catalyst
  LANGUAGES C CXX
  DESCRIPTION "An MLIR-plugin for connecting MQT Core with Xanadu's Catalyst")

add_custom_target(core-plugins-catalyst COMMAND ${CMAKE_COMMAND} -E echo "Building core-plugins-catalyst")


