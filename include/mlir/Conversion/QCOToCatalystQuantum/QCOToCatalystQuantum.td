// Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCOToCatalystQuantum : Pass<"qco-to-catalystquantum", "::mlir::ModuleOp"> {
  let summary = "Convert MQT's QCO dialect to Catalyst's Quantum dialect.";

  let description = [{
    Reconstructs fixed Catalyst quantum registers from the scalar qubits and
    register metadata used by QCO and converts supported QCO operations back to
    Catalyst's value-semantic Quantum dialect.
  }];
  let dependentDialects = [
    "::catalyst::quantum::QuantumDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::qco::QCODialect"
  ];
}
