// Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %s | FileCheck %s --implicit-check-not='qco.'

module {
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumEntanglingGates
  func.func @testQCOToCatalystQuantumEntanglingGates() {
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // CHECK: quantum.custom "SWAP"
    // CHECK: quantum.custom "ISWAP"
    // CHECK: quantum.custom "ISWAP"{{.*}} adj
    // ECR and DCX are decomposed to runtime-supported Catalyst gates.
    // CHECK: quantum.custom "PauliZ"
    // CHECK: quantum.custom "CNOT"
    // CHECK: quantum.custom "RX"
    // CHECK: quantum.custom "CNOT"
    // CHECK: quantum.custom "CNOT"
    %swap0, %swap1 = qco.swap %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %iswap0, %iswap1 = qco.iswap %swap0, %swap1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %iswapdg0, %iswapdg1 = qco.inv (%arg0 = %iswap0, %arg1 = %iswap1) {
      %out0, %out1 = qco.iswap %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
    %ecr0, %ecr1 = qco.ecr %iswapdg0, %iswapdg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %dcx0, %dcx1 = qco.dcx %ecr0, %ecr1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // CHECK: quantum.custom "CSWAP"
    %control, %target0, %target1 = qco.ctrl(%q2) targets(%arg0 = %dcx0, %arg1 = %dcx1) {
      %out0, %out1 = qco.swap %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    qco.dealloc %target0 : !qco.qubit
    qco.dealloc %target1 : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: quantum.dealloc
    return
  }
}
