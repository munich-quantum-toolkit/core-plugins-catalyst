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
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}3)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    // CHECK: %[[Q2:.*]] = quantum.extract %[[REG]][{{ *}}2]
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // CHECK: %[[SWAP:.*]]:2 = quantum.custom "SWAP"() %[[Q0]], %[[Q1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[ISWAP:.*]]:2 = quantum.custom "ISWAP"() %[[SWAP]]#0, %[[SWAP]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ISWAPDG:.*]]:2 = quantum.custom "ISWAP"() %[[ISWAP]]#0, %[[ISWAP]]#1 adj : !quantum.bit, !quantum.bit
    // ECR and DCX are decomposed to runtime-supported Catalyst gates.
    // CHECK: %[[PI_HALF:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[ISWAPDG]]#0 : !quantum.bit
    // CHECK: %[[CNOT0:.*]]:2 = quantum.custom "CNOT"() %[[Z]], %[[ISWAPDG]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[SX_ANGLE:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[SX_PHASE:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[SX:.*]] = quantum.custom "RX"(%[[SX_ANGLE]]) %[[CNOT0]]#1 : !quantum.bit
    // CHECK: quantum.gphase(%[[SX_PHASE]])
    // CHECK: %[[RX:.*]] = quantum.custom "RX"(%[[PI_HALF]]) %[[CNOT0]]#0 : !quantum.bit
    // CHECK: %[[RY:.*]] = quantum.custom "RY"(%[[PI_HALF]]) %[[RX]] : !quantum.bit
    // CHECK: %[[RX2:.*]] = quantum.custom "RX"(%[[PI_HALF]]) %[[RY]] : !quantum.bit
    // CHECK: %[[CNOT1:.*]]:2 = quantum.custom "CNOT"() %[[RX2]], %[[SX]] : !quantum.bit, !quantum.bit
    // CHECK: %[[CNOT2:.*]]:2 = quantum.custom "CNOT"() %[[CNOT1]]#1, %[[CNOT1]]#0 : !quantum.bit, !quantum.bit
    %swap0, %swap1 = qco.swap %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %iswap0, %iswap1 = qco.iswap %swap0, %swap1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %iswapdg0, %iswapdg1 = qco.inv (%arg0 = %iswap0, %arg1 = %iswap1) {
      %out0, %out1 = qco.iswap %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
    %ecr0, %ecr1 = qco.ecr %iswapdg0, %iswapdg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %dcx0, %dcx1 = qco.dcx %ecr0, %ecr1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // CHECK: %[[CSWAP:.*]]:3 = quantum.custom "CSWAP"() %[[Q2]], %[[CNOT2]]#1, %[[CNOT2]]#0 : !quantum.bit, !quantum.bit, !quantum.bit
    %control, %target0, %target1 = qco.ctrl(%q2) targets(%arg0 = %dcx0, %arg1 = %dcx1) {
      %out0, %out1 = qco.swap %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    qco.dealloc %target0 : !qco.qubit
    qco.dealloc %target1 : !qco.qubit
    qco.dealloc %control : !qco.qubit
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[CSWAP]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[CSWAP]]#2 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG2:.*]] = quantum.insert %[[REG1]][{{ *}}2], %[[CSWAP]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG2]] : !quantum.reg
    return
  }
}
