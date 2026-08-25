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
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}2)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    %q0 = qco.alloc("input", 2, 0) : !qco.qubit
    %q1 = qco.alloc("input", 2, 1) : !qco.qubit

    // CHECK: %[[RX:.*]] = quantum.custom "RX"(%[[PHI]]) %[[Q0]] : !quantum.bit
    // CHECK: %[[RY:.*]] = quantum.custom "RY"(%[[PHI]]) %[[RX]] : !quantum.bit
    // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%[[PHI]]) %[[RY]] : !quantum.bit
    // CHECK: %[[P:.*]] = quantum.custom "PhaseShift"(%[[PHI]]) %[[RZ]] : !quantum.bit
    // QCO and Catalyst use opposite global-phase signs.
    // CHECK: quantum.gphase(%[[PHI]]) adj
    %rx = qco.rx(%phi) %q0 : !qco.qubit -> !qco.qubit
    %ry = qco.ry(%phi) %rx : !qco.qubit -> !qco.qubit
    %rz = qco.rz(%phi) %ry : !qco.qubit -> !qco.qubit
    %phase = qco.p(%phi) %rz : !qco.qubit -> !qco.qubit
    qco.gphase(%phi)

    // CHECK: %[[OUTPUTS:.*]]:2 = quantum.custom "CRX"(%[[PHI]]) %[[Q1]], %[[P]] : !quantum.bit, !quantum.bit
    %control, %target = qco.ctrl(%q1) targets(%arg = %phase) {
      %out = qco.rx(%phi) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    qco.dealloc %target : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[OUTPUTS]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[OUTPUTS]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG1]] : !quantum.reg
    return
  }
}
