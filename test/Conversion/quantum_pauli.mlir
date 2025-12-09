// Copyright (c) 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --catalyst-pipeline="builtin.module(catalystquantum-to-mqtopt)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// Pauli family (X, Y, Z) and controlled variants
// Tests both static and dynamic allocation/extraction
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptPauliGates
  func.func @testCatalystQuantumToMQTOptPauliGates(%n : i64, %idx : i64) {
    // --- Dynamic allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[SIZE_CAST:.*]] = arith.index_cast %arg0 : i64 to index
    // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[SIZE_CAST]]) : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][%[[C0]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][%[[C1]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[Q2:.*]] = memref.load %[[ALLOC]][%[[C2]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[IDX_CAST:.*]] = arith.index_cast %arg1 : i64 to index
    // CHECK: %[[Q3:.*]] = memref.load %[[ALLOC]][%[[IDX_CAST]]] : memref<?x!mqtopt.Qubit>

    // --- Uncontrolled Pauli gates --------------------------------------------------------------
    // CHECK: %[[X1:.*]] = mqtopt.x(static [] mask []) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[Y1:.*]] = mqtopt.y(static [] mask []) %[[X1]] : !mqtopt.Qubit
    // CHECK: %[[Z1:.*]] = mqtopt.z(static [] mask []) %[[Y1]] : !mqtopt.Qubit
    // CHECK: %[[I1:.*]] = mqtopt.i(static [] mask []) %[[Z1]] : !mqtopt.Qubit

    // --- Controlled Pauli gates ----------------------------------------------------------------
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[T0_1:.*]], %[[C1_0:.*]] = mqtopt.x(static [] mask []) %[[I1]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T0_2:.*]], %[[C1_1:.*]] = mqtopt.y(static [] mask []) %[[T0_1]] ctrl %[[C1_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T0_3:.*]], %[[C1_2:.*]] = mqtopt.z(static [] mask []) %[[T0_2]] ctrl %[[C1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T0_4:.*]], %[[C1_3:.*]] = mqtopt.i(static [] mask []) %[[T0_3]] ctrl %[[C1_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Two-qubit controlled gates ------------------------------------------------------------
    // CHECK: %[[T0_5:.*]], %[[C1_4:.*]] = mqtopt.x(static [] mask []) %[[T0_4]] ctrl %[[C1_3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T0_6:.*]], %[[C1_5:.*]] = mqtopt.y(static [] mask []) %[[T0_5]] ctrl %[[C1_4]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T0_7:.*]], %[[C1_6:.*]] = mqtopt.z(static [] mask []) %[[T0_6]] ctrl %[[C1_5]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    
    // --- Toffoli (2 controls + 1 target) -------------------------------------------------------
    // CHECK: %[[T0_8:.*]], %[[C12_0:.*]]:2 = mqtopt.x(static [] mask []) %[[T0_7]] ctrl %[[C1_6]], %[[Q2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // --- Controlled two-qubit controlled gates: (q0 controlled by q1) controlled by q2 ---------------------------------------------------------------------------
    // CHECK: %[[T0_9:.*]], %[[C12_1:.*]]:2 = mqtopt.x(static [] mask []) %[[T0_8]] ctrl %[[C12_0]]#0, %[[C12_0]]#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T0_10:.*]], %[[C12_2:.*]]:2 = mqtopt.y(static [] mask []) %[[T0_9]] ctrl %[[C12_1]]#0, %[[C12_1]]#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T0_11:.*]], %[[C12_3:.*]]:2 = mqtopt.z(static [] mask []) %[[T0_10]] ctrl %[[C12_2]]#0, %[[C12_2]]#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T0_12:.*]], %[[C123:.*]]:3 = mqtopt.x(static [] mask []) %[[T0_11]] ctrl %[[C12_3]]#0, %[[C12_3]]#1, %[[Q3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // Release qubits
    // CHECK: %[[IDX_CAST_FINAL:.*]] = arith.index_cast %arg1 : i64 to index
    // CHECK: memref.store %[[C123]]#2, %[[ALLOC]][%[[IDX_CAST_FINAL]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C_2:.*]] = arith.constant 2 : index
    // CHECK: memref.store %[[C123]]#1, %[[ALLOC]][%[[C_2]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C_1:.*]] = arith.constant 1 : index
    // CHECK: memref.store %[[C123]]#0, %[[ALLOC]][%[[C_1]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C_0:.*]] = arith.constant 0 : index
    // CHECK: memref.store %[[T0_12]], %[[ALLOC]][%[[C_0]]] : memref<?x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<?x!mqtopt.Qubit>

    // Prepare qubits with dynamic allocation
    %qreg = quantum.alloc(%n) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit
    %q3 = quantum.extract %qreg[%idx] : !quantum.reg -> !quantum.bit

    // Non-controlled Pauli gates
    %q0_x = quantum.custom "PauliX"() %q0 : !quantum.bit
    %q0_y = quantum.custom "PauliY"() %q0_x : !quantum.bit
    %q0_z = quantum.custom "PauliZ"() %q0_y : !quantum.bit
    %q0_i = quantum.custom "Identity"() %q0_z : !quantum.bit

    %true = arith.constant true

    // Controlled Pauli gates: q0 controlled by q1
    %q0_ctrlx, %q1_ctrlx = quantum.custom "PauliX"() %q0_i ctrls(%q1) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrly, %q1_ctrly = quantum.custom "PauliY"() %q0_ctrlx ctrls(%q1_ctrlx) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrlz, %q1_ctrlz = quantum.custom "PauliZ"() %q0_ctrly ctrls(%q1_ctrly) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrli, %q1_ctrli = quantum.custom "Identity"() %q0_ctrlz ctrls(%q1_ctrlz) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit

    // C gates: q0 controlled by q1
    %q0_cx, %q1_cx = quantum.custom "CNOT"() %q1_ctrli, %q0_ctrli : !quantum.bit, !quantum.bit
    %q0_cy, %q1_cy = quantum.custom "CY"() %q1_cx, %q0_cx : !quantum.bit, !quantum.bit
    %q0_cz, %q1_cz = quantum.custom "CZ"() %q1_cy, %q0_cy : !quantum.bit, !quantum.bit
    %q0_ct, %q1_ct, %q2_ct = quantum.custom "Toffoli"() %q1_cz, %q2, %q0_cz : !quantum.bit, !quantum.bit, !quantum.bit

    // Controlled-C gates: (q0 controlled by q1) controlled by q2
    %q0_ccx, %q1_ccx, %q2_ccx = quantum.custom "CNOT"() %q1_ct, %q0_ct ctrls(%q2_ct) ctrlvals(%true) :!quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_ccy, %q1_ccy, %q2_ccy = quantum.custom "CY"() %q1_ccx, %q0_ccx ctrls(%q2_ccx) ctrlvals(%true) :!quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_ccz, %q1_ccz, %q2_ccz = quantum.custom "CZ"() %q1_ccy, %q0_ccy ctrls(%q2_ccy) ctrlvals(%true) :!quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cccx, %q1_cccx, %q2_cccx, %q3_cccx = quantum.custom "Toffoli"() %q1_ccz, %q2_ccz, %q0_ccz ctrls(%q3) ctrlvals(%true) :!quantum.bit, !quantum.bit, !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[%idx], %q3_cccx : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 2], %q2_cccx : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 1], %q1_cccx : !quantum.reg, !quantum.bit
    %qreg4 = quantum.insert %qreg3[ 0], %q0_cccx : !quantum.reg, !quantum.bit

    quantum.dealloc %qreg4 : !quantum.reg

    return
  }
}
