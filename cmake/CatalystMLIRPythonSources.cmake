# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# setup-mlir deliberately ships MLIR without its native Python extension. Catalyst uses jaxlib's
# extension, but its standalone build still embeds MLIR's pure Python and generated dialect sources.
# Declare those sources from Catalyst's exact LLVM submodule before Catalyst declares its own Python
# package.
if(PROJECT_NAME STREQUAL "Catalyst" AND NOT TARGET MLIRPythonSources.Dialects)
  if(NOT MQT_CATALYST_LLVM_SOURCE_DIR)
    message(FATAL_ERROR "MQT_CATALYST_LLVM_SOURCE_DIR is required")
  endif()

  find_package(MLIR REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(AddMLIRPython)

  set(MLIR_SOURCE_DIR "${MQT_CATALYST_LLVM_SOURCE_DIR}/mlir")
  set(MLIR_MAIN_SRC_DIR "${MLIR_SOURCE_DIR}")
  set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")
  set(MLIR_PYTHON_PACKAGE_PREFIX "mlir_quantum")
  set(MLIR_PYTHON_SOURCES_ONLY ON)
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  foreach(target IN ITEMS acc_common_td LinalgOdsGen omp_common_td)
    if(NOT TARGET ${target})
      add_custom_target(${target})
    endif()
  endforeach()
  add_subdirectory("${MLIR_SOURCE_DIR}/python" "${CMAKE_BINARY_DIR}/mlir-python-sources")

  # The portable MLIR distribution excludes upstream test libraries. Catalyst lists these test-only
  # targets for its opt tool and test aggregate even when they are not required by the compiler
  # driver packaged in the wheel.
  if(NOT TARGET MLIRTestDialect)
    add_library(MLIRTestDialect STATIC "${CMAKE_CURRENT_LIST_DIR}/CatalystMLIRTestDialectStub.cpp")
    target_compile_features(MLIRTestDialect PRIVATE cxx_std_17)
    target_link_libraries(MLIRTestDialect PUBLIC MLIRIR)
  endif()
  if(NOT TARGET CatalystUnitTests)
    add_custom_target(CatalystUnitTests)
  endif()
endif()
