# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  MQTOptToCatalystQuantum
  MQTOptToCatalystQuantum.cpp
  LINK_LIBS
  MLIRMQTOpt
  MLIRTransforms
  MLIRFuncDialect
  DEPENDS
  MQTOptToCatalystQuantumIncGen)

target_compile_features(MQTOptToCatalystQuantum PUBLIC cxx_std_20)
target_compile_options(MQTOptToCatalystQuantum
                       PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fexceptions>)
target_include_directories(MQTOptToCatalystQuantum PRIVATE ${CATALYST_INCLUDE_DIRS})

file(GLOB_RECURSE CONVERSION_HEADERS_SOURCE
     "${MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR}/mlir/Conversion/MQTOptToCatalystQuantum/*.h")
file(GLOB_RECURSE CONVERSION_HEADERS_BUILD
     "${MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR}/mlir/Conversion/MQTOptToCatalystQuantum/*.inc")

# add public headers using file sets
target_sources(
  MQTOptToCatalystQuantum
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR}
         FILES
         ${CONVERSION_HEADERS_SOURCE}
         FILE_SET
         HEADERS
         BASE_DIRS
         ${MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR}
         FILES
         ${CONVERSION_HEADERS_BUILD})

if(ENABLE_COVERAGE)
  target_compile_options(MQTOptToCatalystQuantum PRIVATE --coverage -O0)
  target_link_options(MQTOptToCatalystQuantum PRIVATE --coverage)
endif()
