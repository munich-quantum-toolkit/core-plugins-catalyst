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
// Clifford + T and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToQCOCliffordT
  func.func @testCatalystQuantumToQCOCliffordT() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[Q0:.*]] = qco.alloc("qreg0", 2, 0) : !qco.qubit
    // CHECK: %[[Q1:.*]] = qco.alloc("qreg0", 2, 1) : !qco.qubit
    // Prepare qubits
    %qreg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled Clifford+T gates ---------------------------------------------------------
    // CHECK: %[[H:.*]] = qco.h %[[Q0]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[SX:.*]] = qco.sx %[[H]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[SXDG:.*]] = qco.inv (%[[SX_ARG:.*]] = %[[SX]]) {
    // CHECK: %[[SX_OUT:.*]] = qco.sx %[[SX_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[SX_OUT]]
    // CHECK: } {{.*}} : {!qco.qubit} -> {!qco.qubit}
    // CHECK: %[[S:.*]] = qco.s %[[SXDG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[SDG:.*]] = qco.inv (%[[S_ARG:.*]] = %[[S]]) {
    // CHECK: %[[S_OUT:.*]] = qco.s %[[S_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[S_OUT]]
    // CHECK: } {{.*}} : {!qco.qubit} -> {!qco.qubit}
    // CHECK: %[[T:.*]] = qco.t %[[SDG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[TDG:.*]] = qco.inv (%[[T_ARG:.*]] = %[[T]]) {
    // CHECK: %[[T_OUT:.*]] = qco.t %[[T_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[T_OUT]]
    // CHECK: } {{.*}} : {!qco.qubit} -> {!qco.qubit}
    %h = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %sx = quantum.custom "SX"() %h : !quantum.bit
    %sxdg = quantum.custom "SX"() %sx adj : !quantum.bit
    %s = quantum.custom "S"() %sxdg : !quantum.bit
    %sdg = quantum.custom "S"() %s adj : !quantum.bit
    %t = quantum.custom "T"() %sdg : !quantum.bit
    %tdg = quantum.custom "T"() %t adj : !quantum.bit

    // --- Controlled Clifford+T gates -----------------------------------------------------------
    // CHECK: %[[CH_C:.*]], %[[CH_T:.*]] = qco.ctrl(%[[Q1]]) targets (%[[CH_ARG:.*]] = %[[TDG]]) {
    // CHECK: %[[CH_OUT:.*]] = qco.h %[[CH_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CH_OUT]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CSX_C:.*]], %[[CSX_T:.*]] = qco.ctrl(%[[CH_C]]) targets (%[[CSX_ARG:.*]] = %[[CH_T]]) {
    // CHECK: %[[CSX_OUT:.*]] = qco.sx %[[CSX_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CSX_OUT]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CSXDG_C:.*]], %[[CSXDG_T:.*]] = qco.ctrl(%[[CSX_C]]) targets (%[[CSXDG_ARG:.*]] = %[[CSX_T]]) {
    // CHECK: %[[CSXDG_INV:.*]] = qco.inv (%[[CSXDG_INV_ARG:.*]] = %[[CSXDG_ARG]]) {
    // CHECK: %[[CSXDG_OUT:.*]] = qco.sx %[[CSXDG_INV_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CSXDG_OUT]]
    // CHECK: } {{.*}} : {!qco.qubit} -> {!qco.qubit}
    // CHECK: qco.yield %[[CSXDG_INV]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CS_C:.*]], %[[CS_T:.*]] = qco.ctrl(%[[CSXDG_C]]) targets (%[[CS_ARG:.*]] = %[[CSXDG_T]]) {
    // CHECK: %[[CS_OUT:.*]] = qco.s %[[CS_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CS_OUT]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CSDG_C:.*]], %[[CSDG_T:.*]] = qco.ctrl(%[[CS_C]]) targets (%[[CSDG_ARG:.*]] = %[[CS_T]]) {
    // CHECK: %[[CSDG_INV:.*]] = qco.inv (%[[CSDG_INV_ARG:.*]] = %[[CSDG_ARG]]) {
    // CHECK: %[[CSDG_OUT:.*]] = qco.s %[[CSDG_INV_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CSDG_OUT]]
    // CHECK: } {{.*}} : {!qco.qubit} -> {!qco.qubit}
    // CHECK: qco.yield %[[CSDG_INV]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CT_C:.*]], %[[CT_T:.*]] = qco.ctrl(%[[CSDG_C]]) targets (%[[CT_ARG:.*]] = %[[CSDG_T]]) {
    // CHECK: %[[CT_OUT:.*]] = qco.t %[[CT_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CT_OUT]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CTDG_C:.*]], %[[CTDG_T:.*]] = qco.ctrl(%[[CT_C]]) targets (%[[CTDG_ARG:.*]] = %[[CT_T]]) {
    // CHECK: %[[CTDG_INV:.*]] = qco.inv (%[[CTDG_INV_ARG:.*]] = %[[CTDG_ARG]]) {
    // CHECK: %[[CTDG_OUT:.*]] = qco.t %[[CTDG_INV_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CTDG_OUT]]
    // CHECK: } {{.*}} : {!qco.qubit} -> {!qco.qubit}
    // CHECK: qco.yield %[[CTDG_INV]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %true = arith.constant true
    %ch, %chc = quantum.custom "Hadamard"() %tdg ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %csx, %csxc = quantum.custom "SX"() %ch ctrls(%chc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %csxdg, %csxdgc = quantum.custom "SX"() %csx adj ctrls(%csxc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %cs, %csc = quantum.custom "S"() %csxdg ctrls(%csxdgc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %csdg, %csdgc = quantum.custom "S"() %cs adj ctrls(%csc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %ct, %ctc = quantum.custom "T"() %csdg ctrls(%csdgc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %ctdg, %ctdgc = quantum.custom "T"() %ct adj ctrls(%ctc) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: qco.dealloc %[[CTDG_T]] : !qco.qubit
    // CHECK: qco.dealloc %[[CTDG_C]] : !qco.qubit
    // Release qubits
    %reg0 = quantum.insert %qreg[0], %ctdg : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %ctdgc : !quantum.reg, !quantum.bit
    quantum.dealloc %reg1 : !quantum.reg
    return
  }
}
