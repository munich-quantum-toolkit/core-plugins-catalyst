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

    // --- Uncontrolled Pauli gates --------------------------------------------------------------
    // CHECK: %[[X1:.*]] = mqtopt.x(static [] mask []) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[Y1:.*]] = mqtopt.y(static [] mask []) %[[X1]] : !mqtopt.Qubit
    // CHECK: %[[Z1:.*]] = mqtopt.z(static [] mask []) %[[Y1]] : !mqtopt.Qubit
    // CHECK: %[[I1:.*]] = mqtopt.i(static [] mask []) %[[Z1]] : !mqtopt.Qubit

    // --- Controlled Pauli gates ----------------------------------------------------------------
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[T1:.*]], %[[C1_:.*]] = mqtopt.x(static [] mask []) %[[I1]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T2:.*]], %[[C2_:.*]] = mqtopt.y(static [] mask []) %[[T1]] ctrl %[[C1_]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T3:.*]], %[[C3_:.*]] = mqtopt.z(static [] mask []) %[[T2]] ctrl %[[C2_]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T4:.*]], %[[C4_:.*]] = mqtopt.i(static [] mask []) %[[T3]] ctrl %[[C3_]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Two-qubit controlled gates ------------------------------------------------------------
    // CHECK: %[[T5:.*]], %[[C5:.*]] = mqtopt.x(static [] mask []) %[[T4]] ctrl %[[C4_]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T6:.*]], %[[C6:.*]] = mqtopt.y(static [] mask []) %[[T5]] ctrl %[[C5]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T7:.*]], %[[C7:.*]] = mqtopt.z(static [] mask []) %[[T6]] ctrl %[[C6]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T8:.*]], %[[C8:.*]]:2 = mqtopt.x(static [] mask []) %[[T7]] ctrl %[[C7]], %[[Q2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // Release qubits
    // CHECK: %[[C_2:.*]] = arith.constant 2 : index
    // CHECK: memref.store %[[C8]]#1, %[[ALLOC]][%[[C_2]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C_1:.*]] = arith.constant 1 : index
    // CHECK: memref.store %[[C8]]#0, %[[ALLOC]][%[[C_1]]] : memref<?x!mqtopt.Qubit>
    // CHECK: %[[C_0:.*]] = arith.constant 0 : index
    // CHECK: memref.store %[[T8]], %[[ALLOC]][%[[C_0]]] : memref<?x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<?x!mqtopt.Qubit>

    // Prepare qubits with dynamic allocation
    %qreg = quantum.alloc(%n) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    // Non-controlled Pauli gates
    %q0_x = quantum.custom "PauliX"() %q0 : !quantum.bit
    %q0_y = quantum.custom "PauliY"() %q0_x : !quantum.bit
    %q0_z = quantum.custom "PauliZ"() %q0_y : !quantum.bit
    %q0_i = quantum.custom "Identity"() %q0_z : !quantum.bit

    %true = arith.constant true

    // Controlled Pauli gates, NOTE: %target', %control' = quantum.custom "PauliX"() %target ctrls(%control)
    %q0_ctrlx, %q1_ctrlx = quantum.custom "PauliX"() %q0_i ctrls(%q1) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrly, %q1_ctrly = quantum.custom "PauliY"() %q0_ctrlx ctrls(%q1_ctrlx) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrlz, %q1_ctrlz = quantum.custom "PauliZ"() %q0_ctrly ctrls(%q1_ctrly) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrli, %q1_ctrli = quantum.custom "Identity"() %q0_ctrlz ctrls(%q1_ctrlz) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit

    // C gates, NOTE: %control', %target' = quantum.custom "CNOT"() %control, %target
    %q1_cx, %q0_cx = quantum.custom "CNOT"() %q1_ctrli, %q0_ctrli : !quantum.bit, !quantum.bit
    %q1_cy, %q0_cy = quantum.custom "CY"()   %q1_cx,    %q0_cx : !quantum.bit, !quantum.bit
    %q1_cz, %q0_cz = quantum.custom "CZ"()   %q1_cy,    %q0_cy : !quantum.bit, !quantum.bit
    %q1_ct, %q2_ct, %q0_ct = quantum.custom "Toffoli"() %q1_cz, %q2, %q0_cz : !quantum.bit, !quantum.bit, !quantum.bit

    // Controlled-C gates (become multi-contolled Pauli gates in Catalyst)

    // Release qubits
    %qreg2 = quantum.insert %qreg[ 2], %q2_ct : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 1], %q1_ct : !quantum.reg, !quantum.bit
    %qreg4 = quantum.insert %qreg3[ 0], %q0_ct : !quantum.reg, !quantum.bit

    quantum.dealloc %qreg4 : !quantum.reg

    return
  }
}
