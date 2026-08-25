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
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumCliffordT
  func.func @testQCOToCatalystQuantumCliffordT() {
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}2)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    %q0 = qco.alloc("input", 2, 0) : !qco.qubit
    %q1 = qco.alloc("input", 2, 1) : !qco.qubit

    // CHECK: %[[ID:.*]] = quantum.custom "Identity"() %[[Q0]] : !quantum.bit
    // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[ID]] : !quantum.bit
    // SX and SXdg are lowered to runtime-supported RX and global phase operations.
    // CHECK: %[[PI_HALF:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[MINUS_PI_QUARTER:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[SX:.*]] = quantum.custom "RX"(%[[PI_HALF]]) %[[H]] : !quantum.bit
    // CHECK: quantum.gphase(%[[MINUS_PI_QUARTER]])
    // CHECK: %[[PI_HALF_ADJ:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[MINUS_PI_QUARTER_ADJ:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[SXDG:.*]] = quantum.custom "RX"(%[[PI_HALF_ADJ]]) %[[SX]] adj : !quantum.bit
    // CHECK: quantum.gphase(%[[MINUS_PI_QUARTER_ADJ]]) adj
    // CHECK: %[[S:.*]] = quantum.custom "S"() %[[SXDG]] : !quantum.bit
    // CHECK: %[[SDG:.*]] = quantum.custom "S"() %[[S]] adj : !quantum.bit
    // CHECK: %[[T:.*]] = quantum.custom "T"() %[[SDG]] : !quantum.bit
    // CHECK: %[[TDG:.*]] = quantum.custom "T"() %[[T]] adj : !quantum.bit
    %id = qco.id %q0 : !qco.qubit -> !qco.qubit
    %h = qco.h %id : !qco.qubit -> !qco.qubit
    %sx = qco.sx %h : !qco.qubit -> !qco.qubit
    %sxdg = qco.sxdg %sx : !qco.qubit -> !qco.qubit
    %s = qco.s %sxdg : !qco.qubit -> !qco.qubit
    %sdg = qco.sdg %s : !qco.qubit -> !qco.qubit
    %t = qco.t %sdg : !qco.qubit -> !qco.qubit
    %tdg = qco.tdg %t : !qco.qubit -> !qco.qubit

    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[TARGET:.*]], %[[CONTROL:.*]] = quantum.custom "Hadamard"() %[[TDG]] ctrls(%[[Q1]]) ctrlvals(%[[TRUE]]) : !quantum.bit ctrls !quantum.bit
    %control, %target = qco.ctrl(%q1) targets(%arg = %tdg) {
      %out = qco.h %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    qco.dealloc %target : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[TARGET]] : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[CONTROL]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG1]] : !quantum.reg
    return
  }
}
