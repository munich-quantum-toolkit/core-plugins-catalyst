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
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumParameterizedGates(
  // CHECK-SAME: %[[PHI:.*]]: f64)
  func.func @testQCOToCatalystQuantumParameterizedGates(%phi: f64) {
    %q0 = qco.alloc("input", 2, 0) : !qco.qubit
    %q1 = qco.alloc("input", 2, 1) : !qco.qubit

    // CHECK: quantum.custom "RX"(%[[PHI]])
    // CHECK: quantum.custom "RY"(%[[PHI]])
    // CHECK: quantum.custom "RZ"(%[[PHI]])
    // CHECK: quantum.custom "PhaseShift"(%[[PHI]])
    // QCO and Catalyst use opposite global-phase signs.
    // CHECK: quantum.gphase(%[[PHI]]) adj
    %rx = qco.rx(%phi) %q0 : !qco.qubit -> !qco.qubit
    %ry = qco.ry(%phi) %rx : !qco.qubit -> !qco.qubit
    %rz = qco.rz(%phi) %ry : !qco.qubit -> !qco.qubit
    %phase = qco.p(%phi) %rz : !qco.qubit -> !qco.qubit
    qco.gphase(%phi)

    // CHECK: quantum.custom "CRX"(%[[PHI]])
    %control, %target = qco.ctrl(%q1) targets(%arg = %phase) {
      %out = qco.rx(%phi) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    qco.dealloc %target : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: quantum.dealloc
    return
  }
}
