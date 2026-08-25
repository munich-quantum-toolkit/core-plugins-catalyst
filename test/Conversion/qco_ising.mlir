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
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumIsingGates(
  // CHECK-SAME: %[[THETA:.*]]: f64, %[[BETA:.*]]: f64)
  func.func @testQCOToCatalystQuantumIsingGates(%theta: f64, %beta: f64) {
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}3)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    // CHECK: %[[Q2:.*]] = quantum.extract %[[REG]][{{ *}}2]
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // CHECK: %[[XX:.*]]:2 = quantum.custom "IsingXX"(%[[THETA]]) %[[Q0]], %[[Q1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[YY:.*]]:2 = quantum.custom "IsingYY"(%[[THETA]]) %[[XX]]#0, %[[XX]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ZZ:.*]]:2 = quantum.custom "IsingZZ"(%[[THETA]]) %[[YY]]#0, %[[YY]]#1 : !quantum.bit, !quantum.bit
    // RZX and XX±YY are decomposed through supported Catalyst gates.
    // CHECK: %[[H0:.*]] = quantum.custom "Hadamard"() %[[ZZ]]#1 : !quantum.bit
    // CHECK: %[[RZX:.*]]:2 = quantum.custom "IsingZZ"(%[[THETA]]) %[[ZZ]]#0, %[[H0]] : !quantum.bit, !quantum.bit
    // CHECK: %[[H1:.*]] = quantum.custom "Hadamard"() %[[RZX]]#1 : !quantum.bit
    // CHECK: %[[PI:.*]] = arith.constant 3.1415926535897931
    // CHECK: %[[PI_MINUS_BETA:.*]] = arith.subf %[[PI]], %[[BETA]]
    // CHECK: %[[BETA_MINUS_PI:.*]] = arith.subf %[[BETA]], %[[PI]]
    // CHECK: %[[RZ0:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA]]) %[[H1]] : !quantum.bit
    // CHECK: %[[PLUS:.*]]:2 = quantum.custom "IsingXY"(%[[THETA]]) %[[RZX]]#0, %[[RZ0]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI]]) %[[PLUS]]#1 : !quantum.bit
    // CHECK: %[[PI_2:.*]] = arith.constant 3.1415926535897931
    // CHECK: %[[PI_MINUS_BETA_2:.*]] = arith.subf %[[PI_2]], %[[BETA]]
    // CHECK: %[[BETA_MINUS_PI_2:.*]] = arith.subf %[[BETA]], %[[PI_2]]
    // CHECK: %[[X0:.*]] = quantum.custom "PauliX"() %[[PLUS]]#0 : !quantum.bit
    // CHECK: %[[RZ2:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA_2]]) %[[RZ1]] : !quantum.bit
    // CHECK: %[[MINUS:.*]]:2 = quantum.custom "IsingXY"(%[[THETA]]) %[[X0]], %[[RZ2]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ3:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI_2]]) %[[MINUS]]#1 : !quantum.bit
    // CHECK: %[[X1:.*]] = quantum.custom "PauliX"() %[[MINUS]]#0 : !quantum.bit
    %xx0, %xx1 = qco.rxx(%theta) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %yy0, %yy1 = qco.ryy(%theta) %xx0, %xx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %zz0, %zz1 = qco.rzz(%theta) %yy0, %yy1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %rzx0, %rzx1 = qco.rzx(%theta) %zz0, %zz1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %plus0, %plus1 = qco.xx_plus_yy(%theta, %beta) %rzx0, %rzx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %minus0, %minus1 = qco.xx_minus_yy(%theta, %beta) %plus0, %plus1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[TARGETS:.*]]:2, %[[CONTROL:.*]] = quantum.custom "IsingXX"(%[[THETA]]) %[[X1]], %[[RZ3]] ctrls(%[[Q2]]) ctrlvals(%[[TRUE]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %control, %target0, %target1 = qco.ctrl(%q2) targets(%arg0 = %minus0, %arg1 = %minus1) {
      %out0, %out1 = qco.rxx(%theta) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    qco.dealloc %target0 : !qco.qubit
    qco.dealloc %target1 : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[TARGETS]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[TARGETS]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG2:.*]] = quantum.insert %[[REG1]][{{ *}}2], %[[CONTROL]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG2]] : !quantum.reg
    return
  }
}
