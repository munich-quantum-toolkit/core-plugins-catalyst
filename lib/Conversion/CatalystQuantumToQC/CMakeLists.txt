# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  CatalystQuantumToQC
  CatalystQuantumToQC.cpp
  LINK_LIBS
  MLIRQC
  MLIRTransforms
  MLIRFuncDialect
  DEPENDS
  CatalystQuantumToQCIncGen)

target_compile_features(CatalystQuantumToQC PUBLIC cxx_std_20)
target_compile_options(CatalystQuantumToQC PRIVATE -fexceptions)
target_include_directories(CatalystQuantumToQC PRIVATE ${CATALYST_INCLUDE_DIRS})

file(GLOB_RECURSE CONVERSION_HEADERS_SOURCE
     "${MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR}/mlir/Conversion/CatalystQuantumToQC/*.h")
file(GLOB_RECURSE CONVERSION_HEADERS_BUILD
     "${MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR}/mlir/Conversion/CatalystQuantumToQC/*.inc")

# add public headers using file sets
target_sources(
  CatalystQuantumToQC
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
  target_compile_options(CatalystQuantumToQC PRIVATE --coverage -O0)
  target_link_options(CatalystQuantumToQC PRIVATE --coverage)
endif()
