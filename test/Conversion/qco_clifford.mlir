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

    // CHECK: quantum.custom "Identity"
    // CHECK: quantum.custom "Hadamard"
    // SX and SXdg are lowered to runtime-supported RX and global phase operations.
    // CHECK: quantum.custom "RX"
    // CHECK: quantum.gphase
    // CHECK: quantum.custom "RX"{{.*}} adj
    // CHECK: quantum.gphase{{.*}} adj
    // CHECK: quantum.custom "S"
    // CHECK: quantum.custom "S"{{.*}} adj
    // CHECK: quantum.custom "T"
    // CHECK: quantum.custom "T"{{.*}} adj
    %id = qco.id %q0 : !qco.qubit -> !qco.qubit
    %h = qco.h %id : !qco.qubit -> !qco.qubit
    %sx = qco.sx %h : !qco.qubit -> !qco.qubit
    %sxdg = qco.sxdg %sx : !qco.qubit -> !qco.qubit
    %s = qco.s %sxdg : !qco.qubit -> !qco.qubit
    %sdg = qco.sdg %s : !qco.qubit -> !qco.qubit
    %t = qco.t %sdg : !qco.qubit -> !qco.qubit
    %tdg = qco.tdg %t : !qco.qubit -> !qco.qubit

    // CHECK: quantum.custom "Hadamard"{{.*}}ctrls(
    %control, %target = qco.ctrl(%q1) targets(%arg = %tdg) {
      %out = qco.h %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    qco.dealloc %target : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: quantum.dealloc
    return
  }
}
