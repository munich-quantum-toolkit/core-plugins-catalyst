// Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def CatalystQuantumToQC : Pass<"catalystquantum-to-qc"> {
  let summary = "Convert Catalyst's `Quantum` to MQT's `QC` dialect.";

  let description = [{
    This pass converts Catalyst's `Quantum` to MQT's `QC` dialect.
    The following operations are currently NOT converted (and instead marked legal):
    - DeviceInitOp
    - DeviceReleaseOp
    - NamedObsOp
    - ExpvalOp
    - FinalizeOp
    - ComputationalBasisOp
    - StateOp
    - InitializeOp
  }];

  // Define dependent dialects
  let dependentDialects = [
    "catalyst::quantum::QuantumDialect",
    "::mlir::qc"
  ];
}
