# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(READ "${CMAKE_CURRENT_LIST_DIR}/../toolchain.json" MQT_TOOLCHAIN_PINS)
string(JSON MQT_CATALYST_VERSION GET "${MQT_TOOLCHAIN_PINS}" catalyst_version)
string(JSON MQT_CATALYST_REVISION GET "${MQT_TOOLCHAIN_PINS}" catalyst_revision)
string(JSON MQT_CORE_REVISION GET "${MQT_TOOLCHAIN_PINS}" core_revision)
string(JSON MQT_LLVM_REVISION GET "${MQT_TOOLCHAIN_PINS}" llvm_revision)
