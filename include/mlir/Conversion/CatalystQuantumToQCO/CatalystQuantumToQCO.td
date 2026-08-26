// Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def CatalystQuantumToQCO : Pass<"catalystquantum-to-qco", "::mlir::ModuleOp"> {
  let summary = "Convert static Catalyst Quantum circuits to MQT's QCO dialect.";

  let description = [{
    Converts fixed Catalyst registers to scalar QCO qubits and records native
    QCO register metadata so that the original registers can be reconstructed.
    Catalyst device, observable, and terminal measurement-process operations
    remain legal. High-level operators and quantum control-flow must be lowered
    before this pass.
  }];

  let dependentDialects = [
    "::catalyst::quantum::QuantumDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::qco::QCODialect"
  ];
}
