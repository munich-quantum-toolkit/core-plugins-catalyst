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
// RUN:   --catalyst-pipeline="builtin.module(mqtopt-to-catalystquantum)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// Parameterized gates RX/RY/RZ, PhaseShift and controlled variants
// Tests both static constants and dynamic parameters
// Groups: Allocation & extraction / Static params / Dynamic params / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumParameterizedGates
  func.func @testMQTOptToCatalystQuantumParameterizedGates(%phi : f64) {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[THETA:.*]] = arith.constant 3.000000e-01 : f64
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: %[[IDX0:.*]] = arith.index_cast %[[C0]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][%[[IDX0]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX1:.*]] = arith.index_cast %[[C1]] : index to i64
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][%[[IDX1]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX2:.*]] = arith.index_cast %[[C2]] : index to i64
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][%[[IDX2]]] : !quantum.reg -> !quantum.bit

    // --- Static parameters ---------------------------------------------------
    // CHECK: %[[RX:.*]] = quantum.custom "RX"(%[[THETA]]) %[[Q0]] : !quantum.bit
    // CHECK: %[[RY:.*]] = quantum.custom "RY"(%[[THETA]]) %[[RX]] : !quantum.bit
    // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%[[THETA]]) %[[RY]] : !quantum.bit
    // CHECK: %[[PS:.*]] = quantum.custom "PhaseShift"(%[[THETA]]) %[[RZ]] : !quantum.bit
    // CHECK: quantum.gphase(%[[THETA]]) :

    // --- Dynamic parameters (runtime values) ---------------------------------------------------
    // CHECK: %[[DRX:.*]] = quantum.custom "RX"(%arg0) %[[PS]] : !quantum.bit
    // CHECK: %[[DRY:.*]] = quantum.custom "RY"(%arg0) %[[DRX]] : !quantum.bit
    // CHECK: %[[DRZ:.*]] = quantum.custom "RZ"(%arg0) %[[DRY]] : !quantum.bit
    // CHECK: %[[DPS:.*]] = quantum.custom "PhaseShift"(%arg0) %[[DRZ]] : !quantum.bit
    // CHECK: quantum.gphase(%arg0) :

    // --- Controlled with static parameters -----------------------------------------------------
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CRX_T_:.*]], %[[CRX_C_:.*]]:2 = quantum.custom "RX"(%[[THETA]]) %[[DPS]] ctrls(%[[Q1]], %[[Q2]]) ctrlvals(%[[TRUE]]{{.*}}, %[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = quantum.custom "CRX"(%[[THETA]]) %[[CRX_T_]] ctrls(%[[CRX_C_]]#0) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRY_T:.*]], %[[CRY_C:.*]] = quantum.custom "CRY"(%[[THETA]]) %[[CRX_T]] ctrls(%[[CRX_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ_T:.*]], %[[CRZ_C:.*]] = quantum.custom "CRZ"(%[[THETA]]) %[[CRY_T]] ctrls(%[[CRY_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CPS_T:.*]], %[[CPS_C:.*]] = quantum.custom "ControlledPhaseShift"(%[[THETA]]) %[[CRZ_T]] ctrls(%[[CRZ_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Controlled with dynamic parameters ----------------------------------------------------
    // CHECK: %[[DCRX_T:.*]], %[[DCRX_C:.*]] = quantum.custom "CRX"(%arg0) %[[CPS_T]] ctrls(%[[CPS_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[DCRY_T:.*]], %[[DCRY_C:.*]] = quantum.custom "CRY"(%arg0) %[[DCRX_T]] ctrls(%[[DCRX_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[DCRZ_T:.*]], %[[DCRZ_C:.*]] = quantum.custom "CRZ"(%arg0) %[[DCRY_T]] ctrls(%[[DCRY_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[DCPS_T:.*]], %[[DCPS_C:.*]] = quantum.custom "ControlledPhaseShift"(%arg0) %[[DCRZ_T]] ctrls(%[[DCRZ_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.index_cast %[[C0]] : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C0_FINAL]]], %[[DCPS_T]] : !quantum.reg, !quantum.bit
    // CHECK: %[[C1_FINAL:.*]] = arith.index_cast %[[C1]] : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C1_FINAL]]], %[[DCPS_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg

    // Prepare qubits
    %cst = arith.constant 3.000000e-01 : f64
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index

    %r0_0 = memref.alloc() : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %r0_0[%i2] : memref<3x!mqtopt.Qubit>

    // Static parameter rotations (constant)
    %q0_1 = mqtopt.rx(%cst) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%cst) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.rz(%cst) %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.p(%cst) %q0_3 : !mqtopt.Qubit
    mqtopt.gphase(%cst) : ()

    // Dynamic parameter rotations (runtime)
    %q0_5 = mqtopt.rx(%phi) %q0_4 : !mqtopt.Qubit
    %q0_6 = mqtopt.ry(%phi) %q0_5 : !mqtopt.Qubit
    %q0_7 = mqtopt.rz(%phi) %q0_6 : !mqtopt.Qubit
    %q0_8 = mqtopt.p(%phi) %q0_7 : !mqtopt.Qubit
    mqtopt.gphase(%phi) : ()

    // Controlled rotations with static parameters
    %q0_9_, %q1_1_, %q2_1 = mqtopt.rx(%cst) %q0_8 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_9, %q1_1  = mqtopt.rx(%cst) %q0_9_ ctrl %q1_1_ : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_10, %q1_2 = mqtopt.ry(%cst) %q0_9 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_11, %q1_3 = mqtopt.rz(%cst) %q0_10 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_12, %q1_4 = mqtopt.p(%cst) %q0_11 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled rotations with dynamic parameters
    %q0_13, %q1_5 = mqtopt.rx(%phi) %q0_12 ctrl %q1_4 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_14, %q1_6 = mqtopt.ry(%phi) %q0_13 ctrl %q1_5 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_15, %q1_7 = mqtopt.rz(%phi) %q0_14 ctrl %q1_6 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_16, %q1_8 = mqtopt.p(%phi) %q0_15 ctrl %q1_7 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Release qubits
    memref.store %q0_16, %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q1_8, %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q2_1, %r0_0[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %r0_0 : memref<3x!mqtopt.Qubit>
    return
  }
}
