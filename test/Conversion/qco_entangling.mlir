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

// ============================================================================
// Entangling gates (SWAP, ISWAP, ECR, DCX) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumEntanglingGates
  func.func @testQCOToCatalystQuantumEntanglingGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}3)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    // CHECK: %[[Q2:.*]] = quantum.extract %[[REG]][{{ *}}2]
    // Prepare qubits
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // --- Uncontrolled -------------------------------------------------------------------------
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
    // CHECK: quantum.gphase(%[[SX_PHASE]]){{$}}
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

    // --- Controlled ---------------------------------------------------------------------------
    // CHECK: %[[CSWAP:.*]]:3 = quantum.custom "CSWAP"() %[[Q2]], %[[CNOT2]]#1, %[[CNOT2]]#0 : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: %[[TRUE_CISWAP:.*]] = arith.constant true
    // CHECK: %[[CISWAP_T:.*]]:2, %[[CISWAP_C:.*]] = quantum.custom "ISWAP"() %[[CSWAP]]#1, %[[CSWAP]]#2 ctrls(%[[CSWAP]]#0) ctrlvals(%[[TRUE_CISWAP]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CISWAPDG:.*]] = arith.constant true
    // CHECK: %[[CISWAPDG_T:.*]]:2, %[[CISWAPDG_C:.*]] = quantum.custom "ISWAP"() %[[CISWAP_T]]#0, %[[CISWAP_T]]#1 adj ctrls(%[[CISWAP_C]]) ctrlvals(%[[TRUE_CISWAPDG]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // ECR is lowered to the runtime-supported PennyLane decomposition.
    // CHECK: %[[CECR_PI_HALF:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[TRUE_CECR_Z:.*]] = arith.constant true
    // CHECK: %[[CECR_Z_T:.*]], %[[CECR_Z_C:.*]] = quantum.custom "PauliZ"() %[[CISWAPDG_T]]#0 ctrls(%[[CISWAPDG_C]]) ctrlvals(%[[TRUE_CECR_Z]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CECR_CNOT:.*]] = arith.constant true
    // CHECK: %[[CECR_CNOT_T:.*]]:2, %[[CECR_CNOT_C:.*]] = quantum.custom "CNOT"() %[[CECR_Z_T]], %[[CISWAPDG_T]]#1 ctrls(%[[CECR_Z_C]]) ctrlvals(%[[TRUE_CECR_CNOT]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CECR_SX_ANGLE:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[CECR_SX_PHASE:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[TRUE_CECR_SX:.*]] = arith.constant true
    // CHECK: %[[CECR_SX_T:.*]], %[[CECR_SX_C0:.*]] = quantum.custom "RX"(%[[CECR_SX_ANGLE]]) %[[CECR_CNOT_T]]#1 ctrls(%[[CECR_CNOT_C]]) ctrlvals(%[[TRUE_CECR_SX]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CECR_PHASE:.*]] = arith.constant true
    // CHECK: %[[CECR_SX_C:.*]] = quantum.gphase(%[[CECR_SX_PHASE]]) ctrls(%[[CECR_SX_C0]]) ctrlvals(%[[TRUE_CECR_PHASE]]) : ctrls !quantum.bit
    // CHECK: %[[TRUE_CECR_RX0:.*]] = arith.constant true
    // CHECK: %[[CECR_RX0_T:.*]], %[[CECR_RX0_C:.*]] = quantum.custom "RX"(%[[CECR_PI_HALF]]) %[[CECR_CNOT_T]]#0 ctrls(%[[CECR_SX_C]]) ctrlvals(%[[TRUE_CECR_RX0]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CECR_RY:.*]] = arith.constant true
    // CHECK: %[[CECR_RY_T:.*]], %[[CECR_RY_C:.*]] = quantum.custom "RY"(%[[CECR_PI_HALF]]) %[[CECR_RX0_T]] ctrls(%[[CECR_RX0_C]]) ctrlvals(%[[TRUE_CECR_RY]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CECR_RX1:.*]] = arith.constant true
    // CHECK: %[[CECR_RX1_T:.*]], %[[CECR_C:.*]] = quantum.custom "RX"(%[[CECR_PI_HALF]]) %[[CECR_RY_T]] ctrls(%[[CECR_RY_C]]) ctrlvals(%[[TRUE_CECR_RX1]]) : !quantum.bit ctrls !quantum.bit
    // DCX is lowered to two controlled CNOT operations.
    // CHECK: %[[TRUE_CDCX0:.*]] = arith.constant true
    // CHECK: %[[CDCX0_T:.*]]:2, %[[CDCX0_C:.*]] = quantum.custom "CNOT"() %[[CECR_RX1_T]], %[[CECR_SX_T]] ctrls(%[[CECR_C]]) ctrlvals(%[[TRUE_CDCX0]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CDCX1:.*]] = arith.constant true
    // CHECK: %[[CDCX1_T:.*]]:2, %[[CDCX1_C:.*]] = quantum.custom "CNOT"() %[[CDCX0_T]]#1, %[[CDCX0_T]]#0 ctrls(%[[CDCX0_C]]) ctrlvals(%[[TRUE_CDCX1]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %cswapc, %cswap0, %cswap1 = qco.ctrl(%q2) targets(%arg0 = %dcx0, %arg1 = %dcx1) {
      %out0, %out1 = qco.swap %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %ciswapc, %ciswap0, %ciswap1 = qco.ctrl(%cswapc) targets(%arg0 = %cswap0, %arg1 = %cswap1) {
      %out0, %out1 = qco.iswap %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %ciswapdgc, %ciswapdg0, %ciswapdg1 = qco.ctrl(%ciswapc) targets(%arg0 = %ciswap0, %arg1 = %ciswap1) {
      %out0, %out1 = qco.inv (%inv0 = %arg0, %inv1 = %arg1) {
        %invout0, %invout1 = qco.iswap %inv0, %inv1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        qco.yield %invout0, %invout1
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %cecrc, %cecr0, %cecr1 = qco.ctrl(%ciswapdgc) targets(%arg0 = %ciswapdg0, %arg1 = %ciswapdg1) {
      %out0, %out1 = qco.ecr %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %cdcxc, %cdcx0, %cdcx1 = qco.ctrl(%cecrc) targets(%arg0 = %cecr0, %arg1 = %cecr1) {
      %out0, %out1 = qco.dcx %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    // --- Reinsertion ---------------------------------------------------------------------------
    qco.dealloc %cdcx0 : !qco.qubit
    qco.dealloc %cdcx1 : !qco.qubit
    qco.dealloc %cdcxc : !qco.qubit
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[CDCX1_T]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[CDCX1_T]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG2:.*]] = quantum.insert %[[REG1]][{{ *}}2], %[[CDCX1_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG2]] : !quantum.reg
    // Release qubits
    return
  }
}
