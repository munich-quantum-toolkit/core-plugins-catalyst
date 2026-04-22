# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_library(
  QCToCatalystQuantum
  QCToCatalystQuantum.cpp
  LINK_LIBS
  MLIRQCDialect
  MLIRTransforms
  MLIRFuncDialect
  DEPENDS
  QCToCatalystQuantumIncGen)

target_compile_features(QCToCatalystQuantum PUBLIC cxx_std_20)
target_compile_options(QCToCatalystQuantum
                       PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fexceptions>)
target_include_directories(QCToCatalystQuantum PRIVATE ${CATALYST_INCLUDE_DIRS})

file(GLOB_RECURSE CONVERSION_HEADERS_SOURCE
     "${MQT_MLIR_PLUGIN_SOURCE_INCLUDE_DIR}/mlir/Conversion/QCToCatalystQuantum/*.h")
file(GLOB_RECURSE CONVERSION_HEADERS_BUILD
     "${MQT_MLIR_PLUGIN_BUILD_INCLUDE_DIR}/mlir/Conversion/QCToCatalystQuantum/*.inc")

# add public headers using file sets
target_sources(
  QCToCatalystQuantum
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
  target_compile_options(QCToCatalystQuantum PRIVATE --coverage -O0)
  target_link_options(QCToCatalystQuantum PRIVATE --coverage)
endif()
