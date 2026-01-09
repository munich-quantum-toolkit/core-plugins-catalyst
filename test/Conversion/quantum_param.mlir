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
// RUN:   --catalyst-pipeline="builtin.module(catalystquantum-to-mqtopt)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// Parameterized gates RX/RY/RZ, PhaseShift and controlled variants
// Tests both static (compile-time constant) and dynamic (runtime) parameters
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptParameterizedGates
  func.func @testCatalystQuantumToMQTOptParameterizedGates(%dynAngle : f64) {
    // --- Static allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x!mqtopt.Qubit>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][%[[C0]]] : memref<2x!mqtopt.Qubit>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][%[[C1]]] : memref<2x!mqtopt.Qubit>

    // --- Uncontrolled with static parameter (converted to dynamic) ---------------------------------------------------
    // CHECK: %[[RX:.*]] = mqtopt.rx(%cst static [] mask [false]) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[RY:.*]] = mqtopt.ry(%cst static [] mask [false]) %[[RX]] : !mqtopt.Qubit
    // CHECK: %[[RZ:.*]] = mqtopt.rz(%cst static [] mask [false]) %[[RY]] : !mqtopt.Qubit
    // CHECK: %[[PS:.*]] = mqtopt.p(%cst static [] mask [false]) %[[RZ]] : !mqtopt.Qubit

    // --- Controlled with static parameter (converted to dynamic) ------------------------------------------------------
    // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = mqtopt.rx(%cst static [] mask [false]) %[[PS]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CRY_T:.*]], %[[CRY_C:.*]] = mqtopt.ry(%cst static [] mask [false]) %[[CRX_T]] ctrl %[[CRX_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Uncontrolled with dynamic parameter -------------------------------------------------------------------------
    // CHECK: %[[RX2:.*]] = mqtopt.rx(%arg0 static [] mask [false]) %[[CRY_T]] : !mqtopt.Qubit
    // CHECK: %[[RY2:.*]] = mqtopt.ry(%arg0 static [] mask [false]) %[[RX2]] : !mqtopt.Qubit

    // --- Controlled with dynamic parameter ----------------------------------------------------------------------------
    // CHECK: %[[CRZ_T:.*]], %[[CRZ_C:.*]] = mqtopt.rz(%arg0 static [] mask [false]) %[[RY2]] ctrl %[[CRY_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.constant 0 : index
    // CHECK: memref.store %[[CRZ_T]], %[[ALLOC]][%[[C0_FINAL]]] : memref<2x!mqtopt.Qubit>
    // CHECK: %[[C1_FINAL:.*]] = arith.constant 1 : index
    // CHECK: memref.store %[[CRZ_C]], %[[ALLOC]][%[[C1_FINAL]]] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<2x!mqtopt.Qubit>

    // Prepare qubits
    %staticAngle = arith.constant 3.000000e-01 : f64
    %qreg = quantum.alloc( 2) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit

    // Uncontrolled parameterized gates with static parameter
    %q0_rx = quantum.custom "RX"(%staticAngle) %q0 : !quantum.bit
    %q0_ry = quantum.custom "RY"(%staticAngle) %q0_rx : !quantum.bit
    %q0_rz = quantum.custom "RZ"(%staticAngle) %q0_ry : !quantum.bit
    %q0_p = quantum.custom "PhaseShift"(%staticAngle) %q0_rz : !quantum.bit

    // Controlled parameterized gates with static parameter
    %true = arith.constant true
    %q0_crx, %q1_crx = quantum.custom "RX"(%staticAngle) %q0_p ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_cry, %q1_cry = quantum.custom "RY"(%staticAngle) %q0_crx ctrls(%q1_crx) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // Uncontrolled parameterized gates with dynamic parameter
    %q0_rx2 = quantum.custom "RX"(%dynAngle) %q0_cry : !quantum.bit
    %q0_ry2 = quantum.custom "RY"(%dynAngle) %q0_rx2 : !quantum.bit

    // Controlled parameterized gates with dynamic parameter
    %q0_crz, %q1_crz = quantum.custom "RZ"(%dynAngle) %q0_ry2 ctrls(%q1_cry) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 0], %q0_crz : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_crz : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg2 : !quantum.reg
    return
  }
}
