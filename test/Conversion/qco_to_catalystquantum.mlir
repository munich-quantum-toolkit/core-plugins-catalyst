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
// RUN:   %s | FileCheck %s

module {
  func.func private @__mqt_catalyst_qco_qubit_bridge(!qco.qubit) -> !quantum.bit attributes {catalyst.qco_qubit_bridge}

  // CHECK-LABEL: func.func @reconstruct_register(
  // CHECK-SAME: %[[THETA:.*]]: f64)
  func.func @reconstruct_register(%theta: f64) -> i1 {
    // Complete register metadata reconstructs one fixed Catalyst register.
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}3){{.*}}: !quantum.reg
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q2:.*]] = quantum.extract %[[REG]][{{ *}}2] : !quantum.reg -> !quantum.bit
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit
    // CHECK: %[[RX:.*]] = quantum.custom "RX"(%[[THETA]]) %[[H]] : !quantum.bit
    %h = qco.h %q0 : !qco.qubit -> !qco.qubit
    %rx = qco.rx(%theta) %h : !qco.qubit -> !qco.qubit

    // Native QCO modifiers reconstruct Catalyst controls and adjoints.
    // CHECK: %[[CNOT:.*]]:2 = quantum.custom "CNOT"() %[[Q1]], %[[RX]] : !quantum.bit, !quantum.bit
    %controlled, %target = qco.ctrl(%q1) targets(%arg = %rx) {
      %out = qco.x %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    // CHECK: %[[SDG:.*]] = quantum.custom "S"() %[[CNOT]]#1 adj : !quantum.bit
    %sdg = qco.inv (%arg = %target) {
      %out = qco.s %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : {!qco.qubit} -> {!qco.qubit}

    // CHECK: %[[SWAP:.*]]:2 = quantum.custom "SWAP"() %[[SDG]], %[[Q2]] : !quantum.bit, !quantum.bit
    // CHECK: quantum.gphase(%[[THETA]])
    %swap0, %swap1 = qco.swap %sdg, %q2 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    qco.gphase(%theta)

    // QCO's value/result order is restored to Catalyst's result/qubit order.
    // CHECK: %[[RESULT:.*]], %[[MEASURED:.*]] = quantum.measure %[[SWAP]]#0
    // CHECK-SAME: mqt.qco_measure_register_index = 2 : i64
    // CHECK-SAME: mqt.qco_measure_register_name = "c"
    // CHECK-SAME: mqt.qco_measure_register_size = 3 : i64
    %measured, %result = qco.measure("c", 3, 2) %swap0 : !qco.qubit
    qco.dealloc %measured : !qco.qubit
    qco.dealloc %controlled : !qco.qubit
    qco.dealloc %swap1 : !qco.qubit

    // The latest scalar values are inserted before the reconstructed register is released.
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[MEASURED]] : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG2:.*]] = quantum.insert %[[REG1]][{{ *}}2], %[[SWAP]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG2]] : !quantum.reg
    // CHECK: return %[[RESULT]] : i1
    return %result : i1
  }

  // A bridge call resolves the latest scalar value in the operand's lineage.
  // CHECK-LABEL: func.func @reconstruct_scalar_bridge
  func.func @reconstruct_scalar_bridge() -> f64 {
    // CHECK: %[[Q:.*]] = quantum.alloc_qb : !quantum.bit
    // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[Q]] : !quantum.bit
    // CHECK: %[[OBS:.*]] = quantum.namedobs %[[H]][{{ *}}PauliZ]
    // CHECK: %[[EXPVAL:.*]] = quantum.expval %[[OBS]] : f64
    // CHECK: quantum.dealloc_qb %[[H]] : !quantum.bit
    %q = qco.alloc : !qco.qubit
    %h = qco.h %q : !qco.qubit -> !qco.qubit
    %bridge = call @__mqt_catalyst_qco_qubit_bridge(%h) : (!qco.qubit) -> !quantum.bit
    %obs = quantum.namedobs %bridge[PauliZ] : !quantum.obs
    %expval = quantum.expval %obs : f64
    qco.dealloc %h : !qco.qubit
    return %expval : f64
  }
}
