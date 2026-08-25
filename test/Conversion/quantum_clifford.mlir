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

module {
  // CHECK-LABEL: func.func @testCatalystQuantumToQCOCliffordT
  func.func @testCatalystQuantumToQCOCliffordT() {
    // CHECK: %[[Q0:.*]] = qco.alloc("qreg0", 2, 0) : !qco.qubit
    // CHECK: %[[Q1:.*]] = qco.alloc("qreg0", 2, 1) : !qco.qubit
    %qreg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit

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

    // CHECK: %[[CONTROLS:.*]], %[[TARGETS:.*]] = qco.ctrl(%[[Q1]]) targets (%[[TARGET_ARG:.*]] = %[[TDG]]) {
    // CHECK: %[[CONTROLLED_H:.*]] = qco.h %[[TARGET_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CONTROLLED_H]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %true = arith.constant true
    %controlled, %control = quantum.custom "Hadamard"() %tdg ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // CHECK: qco.dealloc %[[TARGETS]] : !qco.qubit
    // CHECK: qco.dealloc %[[CONTROLS]] : !qco.qubit
    %reg0 = quantum.insert %qreg[0], %controlled : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %control : !quantum.reg, !quantum.bit
    quantum.dealloc %reg1 : !quantum.reg
    return
  }
}
