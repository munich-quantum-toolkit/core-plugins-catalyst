# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS CatalystQuantumToQC.td)
mlir_tablegen(CatalystQuantumToQC.h.inc -gen-pass-decls -name CatalystQuantumToQC)
add_public_tablegen_target(CatalystQuantumToQCIncGen)

add_mlir_doc(CatalystQuantumToQC CatalystQuantumToQC ./ -gen-pass-doc)
