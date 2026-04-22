# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QCToCatalystQuantum.td)
mlir_tablegen(QCToCatalystQuantum.h.inc -gen-pass-decls -name QCToCatalystQuantum)
add_public_tablegen_target(QCToCatalystQuantumIncGen)

add_mlir_doc(QCToCatalystQuantum QCToCatalystQuantum ./ -gen-pass-doc)
