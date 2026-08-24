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
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // CHECK: quantum.custom "IsingXX"(%[[THETA]])
    // CHECK: quantum.custom "IsingYY"(%[[THETA]])
    // CHECK: quantum.custom "IsingZZ"(%[[THETA]])
    // RZX and XX±YY are decomposed through supported Catalyst gates.
    // CHECK: quantum.custom "Hadamard"
    // CHECK: quantum.custom "IsingZZ"(%[[THETA]])
    // CHECK: quantum.custom "Hadamard"
    // CHECK: quantum.custom "IsingXY"(%[[THETA]])
    // CHECK: quantum.custom "PauliX"
    // CHECK: quantum.custom "IsingXY"(%[[THETA]])
    %xx0, %xx1 = qco.rxx(%theta) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %yy0, %yy1 = qco.ryy(%theta) %xx0, %xx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %zz0, %zz1 = qco.rzz(%theta) %yy0, %yy1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %rzx0, %rzx1 = qco.rzx(%theta) %zz0, %zz1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %plus0, %plus1 = qco.xx_plus_yy(%theta, %beta) %rzx0, %rzx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %minus0, %minus1 = qco.xx_minus_yy(%theta, %beta) %plus0, %plus1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // CHECK: quantum.custom "IsingXX"(%[[THETA]]){{.*}}ctrls(
    %control, %target0, %target1 = qco.ctrl(%q2) targets(%arg0 = %minus0, %arg1 = %minus1) {
      %out0, %out1 = qco.rxx(%theta) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    qco.dealloc %target0 : !qco.qubit
    qco.dealloc %target1 : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: quantum.dealloc
    return
  }
}
