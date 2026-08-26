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
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %s | FileCheck %s --implicit-check-not='quantum.'

// ============================================================================
// Pauli family (X, Y, Z, Identity) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToQCOPauliGates
  func.func @testCatalystQuantumToQCOPauliGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[Q0:.*]] = qco.alloc("qreg0", 3, 0) : !qco.qubit
    // CHECK: %[[Q1:.*]] = qco.alloc("qreg0", 3, 1) : !qco.qubit
    // CHECK: %[[Q2:.*]] = qco.alloc("qreg0", 3, 2) : !qco.qubit
    // Prepare qubits
    %qreg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[2] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled Pauli gates --------------------------------------------------------------
    // CHECK: %[[X:.*]] = qco.x %[[Q0]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[Y:.*]] = qco.y %[[X]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[Z:.*]] = qco.z %[[Y]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[ID:.*]] = qco.id %[[Z]] {{.*}} : !qco.qubit -> !qco.qubit
    %x = quantum.custom "PauliX"() %q0 : !quantum.bit
    %y = quantum.custom "PauliY"() %x : !quantum.bit
    %z = quantum.custom "PauliZ"() %y : !quantum.bit
    %id = quantum.custom "Identity"() %z : !quantum.bit

    // --- Controlled Pauli gates ---------------------------------------------------------------
    // CHECK: %[[CX_C:.*]], %[[CX_T:.*]] = qco.ctrl(%[[Q1]]) targets (%[[CX_ARG:.*]] = %[[ID]]) {
    // CHECK: %[[CX_OUT:.*]] = qco.x %[[CX_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CX_OUT]]
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}}catalyst.gate_name = "PauliX"{{.*}}catalyst.native_control_count = 0 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CY_C:.*]], %[[CY_T:.*]] = qco.ctrl(%[[CX_C]]) targets (%[[CY_ARG:.*]] = %[[CX_T]]) {
    // CHECK: %[[CY_OUT:.*]] = qco.y %[[CY_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CY_OUT]]
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}}catalyst.gate_name = "PauliY"{{.*}}catalyst.native_control_count = 0 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CZ_C:.*]], %[[CZ_T:.*]] = qco.ctrl(%[[CY_C]]) targets (%[[CZ_ARG:.*]] = %[[CY_T]]) {
    // CHECK: %[[CZ_OUT:.*]] = qco.z %[[CZ_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CZ_OUT]]
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}}catalyst.gate_name = "PauliZ"{{.*}}catalyst.native_control_count = 0 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CI_C:.*]], %[[CI_T:.*]] = qco.ctrl(%[[CZ_C]]) targets (%[[CI_ARG:.*]] = %[[CZ_T]]) {
    // CHECK: %[[CI_OUT:.*]] = qco.id %[[CI_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CI_OUT]]
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}}catalyst.gate_name = "Identity"{{.*}}catalyst.native_control_count = 0 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[NEG_PRE:.*]] = qco.x %[[CI_C]] {{.*}}catalyst.negative_control_wrapper{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[NCX_C:.*]], %[[NCX_T:.*]] = qco.ctrl(%[[NEG_PRE]]) targets (%[[NCX_ARG:.*]] = %[[CI_T]]) {
    // CHECK: %[[NCX_OUT:.*]] = qco.x %[[NCX_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[NCX_OUT]]
    // CHECK: } {{.*}}catalyst.control_values = array<i1: false>{{.*}}catalyst.gate_name = "PauliX"{{.*}}catalyst.native_control_count = 0 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[NEG_POST:.*]] = qco.x %[[NCX_C]] {{.*}}catalyst.negative_control_wrapper{{.*}} : !qco.qubit -> !qco.qubit
    %true = arith.constant true
    %false = arith.constant false
    %cx, %cxc = quantum.custom "PauliX"() %id ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %cy, %cyc = quantum.custom "PauliY"() %cx ctrls(%cxc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %cz, %czc = quantum.custom "PauliZ"() %cy ctrls(%cyc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %ci, %cic = quantum.custom "Identity"() %cz ctrls(%czc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %ncx, %ncxc = quantum.custom "PauliX"() %ci ctrls(%cic) ctrlvals(%false) : !quantum.bit ctrls !quantum.bit

    // --- Native controlled gates ---------------------------------------------------------------
    // CHECK: %[[CNOT_C:.*]], %[[CNOT_T:.*]] = qco.ctrl(%[[NEG_POST]]) targets (%[[CNOT_ARG:.*]] = %[[NCX_T]]) {
    // CHECK: %[[CNOT_OUT:.*]] = qco.x %[[CNOT_ARG]] {{.*}}catalyst.gate_name = "CNOT"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CNOT_OUT]]
    // CHECK: } {{.*}}catalyst.gate_name = "CNOT"{{.*}}catalyst.native_control_count = 1 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CYN_C:.*]], %[[CYN_T:.*]] = qco.ctrl(%[[CNOT_C]]) targets (%[[CYN_ARG:.*]] = %[[CNOT_T]]) {
    // CHECK: %[[CYN_OUT:.*]] = qco.y %[[CYN_ARG]] {{.*}}catalyst.gate_name = "CY"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CYN_OUT]]
    // CHECK: } {{.*}}catalyst.gate_name = "CY"{{.*}}catalyst.native_control_count = 1 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CZN_C:.*]], %[[CZN_T:.*]] = qco.ctrl(%[[CYN_C]]) targets (%[[CZN_ARG:.*]] = %[[CYN_T]]) {
    // CHECK: %[[CZN_OUT:.*]] = qco.z %[[CZN_ARG]] {{.*}}catalyst.gate_name = "CZ"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CZN_OUT]]
    // CHECK: } {{.*}}catalyst.gate_name = "CZ"{{.*}}catalyst.native_control_count = 1 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[TOFFOLI_C:.*]]:2, %[[TOFFOLI_T:.*]] = qco.ctrl(%[[CZN_C]], %[[Q2]]) targets (%[[TOFFOLI_ARG:.*]] = %[[CZN_T]]) {
    // CHECK: %[[TOFFOLI_OUT:.*]] = qco.x %[[TOFFOLI_ARG]] {{.*}}catalyst.gate_name = "Toffoli"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[TOFFOLI_OUT]]
    // CHECK: } {{.*}}catalyst.gate_name = "Toffoli"{{.*}}catalyst.native_control_count = 2 : i64{{.*}} : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit, !qco.qubit}, {!qco.qubit})
    %cnot:2 = quantum.custom "CNOT"() %ncxc, %ncx : !quantum.bit, !quantum.bit
    %cyn:2 = quantum.custom "CY"() %cnot#0, %cnot#1 : !quantum.bit, !quantum.bit
    %czn:2 = quantum.custom "CZ"() %cyn#0, %cyn#1 : !quantum.bit, !quantum.bit
    %toffoli:3 = quantum.custom "Toffoli"() %czn#0, %q2, %czn#1 : !quantum.bit, !quantum.bit, !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: qco.dealloc %[[TOFFOLI_T]] : !qco.qubit
    // CHECK: qco.dealloc %[[TOFFOLI_C]]#0 : !qco.qubit
    // CHECK: qco.dealloc %[[TOFFOLI_C]]#1 : !qco.qubit
    // Release qubits
    %reg0 = quantum.insert %qreg[0], %toffoli#2 : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %toffoli#0 : !quantum.reg, !quantum.bit
    %reg2 = quantum.insert %reg1[2], %toffoli#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %reg2 : !quantum.reg
    return
  }

  // CHECK-LABEL: func.func @testModifiedPauliRot(
  // CHECK-SAME: %[[ANGLE:.*]]: f64)
  // CHECK: %[[CONTROL:.*]] = qco.alloc : !qco.qubit
  // CHECK: %[[Q0:.*]] = qco.alloc : !qco.qubit
  // CHECK: %[[Q1:.*]] = qco.alloc : !qco.qubit
  // CHECK: %[[Q2:.*]] = qco.alloc : !qco.qubit
  // CHECK: %[[PI_HALF:.*]] = arith.constant 1.5707963267948966 : f64
  // CHECK: %[[H:.*]] = qco.h %[[Q0]] {{.*}} : !qco.qubit -> !qco.qubit
  // CHECK: %[[RX:.*]] = qco.rx(%[[PI_HALF]]) %[[Q1]] {{.*}} : !qco.qubit -> !qco.qubit
  // CHECK: %[[CNOT0_C:.*]], %[[CNOT0_T:.*]] = qco.ctrl(%[[H]]) targets (%{{.*}} = %[[RX]]) {
  // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}}catalyst.gate_name = "CNOT"{{.*}}catalyst.native_control_count = 1 : i64
  // CHECK: %[[CNOT1_C:.*]], %[[CNOT1_T:.*]] = qco.ctrl(%[[CNOT0_T]]) targets (%{{.*}} = %[[Q2]]) {
  // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}}catalyst.gate_name = "CNOT"{{.*}}catalyst.native_control_count = 1 : i64
  // CHECK: %[[PAULI_PRE:.*]] = qco.x %{{.*}} {{.*}}catalyst.negative_control_wrapper{{.*}} : !qco.qubit -> !qco.qubit
  // CHECK: %[[PAULI_C:.*]], %[[PAULI_T:.*]] = qco.ctrl(%[[PAULI_PRE]]) targets (%{{.*}} = %[[CNOT1_T]]) {
  // CHECK: qco.inv
  // CHECK: qco.rz(%[[ANGLE]])
  // CHECK: } {{.*}}catalyst.control_values = array<i1: false>{{.*}}catalyst.gate_name = "RZ"{{.*}}catalyst.native_control_count = 0 : i64
  // CHECK: qco.inv
  // CHECK: qco.rx(%[[PI_HALF]])
  // CHECK: qco.h
  func.func @testModifiedPauliRot(%angle: f64) {
    %false = arith.constant false
    %control = quantum.alloc_qb : !quantum.bit
    %q0 = quantum.alloc_qb : !quantum.bit
    %q1 = quantum.alloc_qb : !quantum.bit
    %q2 = quantum.alloc_qb : !quantum.bit
    %pauli:3, %control0 = quantum.paulirot ["X", "Y", "Z"](%angle) %q0, %q1, %q2 adj ctrls(%control) ctrlvals(%false) : !quantum.bit, !quantum.bit, !quantum.bit ctrls !quantum.bit
    quantum.dealloc_qb %pauli#0 : !quantum.bit
    quantum.dealloc_qb %pauli#1 : !quantum.bit
    quantum.dealloc_qb %pauli#2 : !quantum.bit
    quantum.dealloc_qb %control0 : !quantum.bit
    return
  }

}
