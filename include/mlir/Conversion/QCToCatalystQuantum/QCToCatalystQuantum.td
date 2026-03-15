// Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCToCatalystQuantum : Pass<"qc-to-catalystquantum"> {
  let summary = "Convert MQT's `QC` to Catalyst's `Quantum` dialect.";

  let description = [{
    This pass converts MQT's `QC` to Catalyst's `Quantum` dialect.
  }];
  let dependentDialects = [
    "::catalyst::quantum::QuantumDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::qc::QCDialect"
  ];
}
