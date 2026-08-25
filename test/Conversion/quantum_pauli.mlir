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
// RUN:   %s | FileCheck %s --implicit-check-not='quantum.custom'

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
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CY_C:.*]], %[[CY_T:.*]] = qco.ctrl(%[[CX_C]]) targets (%[[CY_ARG:.*]] = %[[CX_T]]) {
    // CHECK: %[[CY_OUT:.*]] = qco.y %[[CY_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CZ_C:.*]], %[[CZ_T:.*]] = qco.ctrl(%[[CY_C]]) targets (%[[CZ_ARG:.*]] = %[[CY_T]]) {
    // CHECK: %[[CZ_OUT:.*]] = qco.z %[[CZ_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CI_C:.*]], %[[CI_T:.*]] = qco.ctrl(%[[CZ_C]]) targets (%[[CI_ARG:.*]] = %[[CZ_T]]) {
    // CHECK: %[[CI_OUT:.*]] = qco.id %[[CI_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true>{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %true = arith.constant true
    %cx, %cxc = quantum.custom "PauliX"() %id ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %cy, %cyc = quantum.custom "PauliY"() %cx ctrls(%cxc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %cz, %czc = quantum.custom "PauliZ"() %cy ctrls(%cyc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %ci, %cic = quantum.custom "Identity"() %cz ctrls(%czc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // --- Native controlled gates ---------------------------------------------------------------
    // CHECK: %[[CNOT_C:.*]], %[[CNOT_T:.*]] = qco.ctrl(%[[CI_C]]) targets (%[[CNOT_ARG:.*]] = %[[CI_T]]) {
    // CHECK: %[[CNOT_OUT:.*]] = qco.x %[[CNOT_ARG]] {{.*}}catalyst.gate_name = "CNOT"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: } {{.*}}catalyst.gate_name = "CNOT"{{.*}}catalyst.native_control_count = 1 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CYN_C:.*]], %[[CYN_T:.*]] = qco.ctrl(%[[CNOT_C]]) targets (%[[CYN_ARG:.*]] = %[[CNOT_T]]) {
    // CHECK: %[[CYN_OUT:.*]] = qco.y %[[CYN_ARG]] {{.*}}catalyst.gate_name = "CY"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: } {{.*}}catalyst.gate_name = "CY"{{.*}}catalyst.native_control_count = 1 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CZN_C:.*]], %[[CZN_T:.*]] = qco.ctrl(%[[CYN_C]]) targets (%[[CZN_ARG:.*]] = %[[CYN_T]]) {
    // CHECK: %[[CZN_OUT:.*]] = qco.z %[[CZN_ARG]] {{.*}}catalyst.gate_name = "CZ"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: } {{.*}}catalyst.gate_name = "CZ"{{.*}}catalyst.native_control_count = 1 : i64{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[TOFFOLI_C:.*]]:2, %[[TOFFOLI_T:.*]] = qco.ctrl(%[[CZN_C]], %[[Q2]]) targets (%[[TOFFOLI_ARG:.*]] = %[[CZN_T]]) {
    // CHECK: %[[TOFFOLI_OUT:.*]] = qco.x %[[TOFFOLI_ARG]] {{.*}}catalyst.gate_name = "Toffoli"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: } {{.*}}catalyst.gate_name = "Toffoli"{{.*}}catalyst.native_control_count = 2 : i64{{.*}} : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit, !qco.qubit}, {!qco.qubit})
    %cnot:2 = quantum.custom "CNOT"() %cic, %ci : !quantum.bit, !quantum.bit
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
}
