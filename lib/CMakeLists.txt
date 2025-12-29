# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Q: Why is it add_llvm_library and not add_mlir_library? A: According to Erick from Xanadu, who
# talked with the MLIR team, the reason is probably, that the MLIR plugin uses the same
# infrastructure as the LLVM plugins. The MLIR-specific CMake convenience macros for plugins are not
# (yet) consistently used/upstreamed across toolchains.
#
# Note: On DLL platforms, the tool that will use this plugin must be linked, see
# https://github.com/llvm/llvm-project/blob/2acecfe65397c162958ab305dc44614ff51e748c/llvm/cmake/modules/AddLLVM.cmake#L770
# and
# https://github.com/llvm/llvm-project/blob/2acecfe65397c162958ab305dc44614ff51e748c/llvm/cmake/modules/AddLLVM.cmake#L517
# This would mean that it is not enough to have the corresponding header at hand, we would need to
# have the catalyst target available to link against it.

add_subdirectory(Conversion)

set(TARGET_NAME mqt-core-plugins-catalyst)

add_llvm_library(${TARGET_NAME} MODULE mqt-plugin.cpp LINK_LIBS CatalystQuantumToMQTOpt
                 MQTOptToCatalystQuantum)

# set required C++ standard
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_20)
target_compile_options(${TARGET_NAME}
                       PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fexceptions>)

# On macOS, allow undefined symbols to be resolved at plugin load time
if(APPLE)
  target_link_options(${TARGET_NAME} PRIVATE "-undefined" "dynamic_lookup")
endif()

if(ENABLE_COVERAGE)
  target_compile_options(${TARGET_NAME} PRIVATE --coverage -O0)
  target_link_options(${TARGET_NAME} PRIVATE --coverage)
endif()

# install directive for scikit-build-core
install(
  TARGETS mqt-core-plugins-catalyst
  DESTINATION .
  COMPONENT mqt-core-plugins-catalyst_Python)
