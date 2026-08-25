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
  // CHECK-LABEL: func.func @testCatalystQuantumToQCOPauliGates
  func.func @testCatalystQuantumToQCOPauliGates() {
    // CHECK: %[[Q0:.*]] = qco.alloc("qreg0", 3, 0) : !qco.qubit
    // CHECK: %[[Q1:.*]] = qco.alloc("qreg0", 3, 1) : !qco.qubit
    // CHECK: %[[Q2:.*]] = qco.alloc("qreg0", 3, 2) : !qco.qubit
    %qreg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[2] : !quantum.reg -> !quantum.bit

    // CHECK: %[[X:.*]] = qco.x %[[Q0]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[Y:.*]] = qco.y %[[X]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[Z:.*]] = qco.z %[[Y]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[ID:.*]] = qco.id %[[Z]] {{.*}} : !qco.qubit -> !qco.qubit
    %x = quantum.custom "PauliX"() %q0 : !quantum.bit
    %y = quantum.custom "PauliY"() %x : !quantum.bit
    %z = quantum.custom "PauliZ"() %y : !quantum.bit
    %id = quantum.custom "Identity"() %z : !quantum.bit

    // CHECK: %[[WRAPPED_CONTROL:.*]] = qco.x %[[Q1]] {{.*}}catalyst.negative_control_wrapper{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[CONTROL_OUT:.*]], %[[TARGET_OUT:.*]] = qco.ctrl(%[[WRAPPED_CONTROL]]) targets (%[[TARGET_ARG:.*]] = %[[ID]]) {
    // CHECK: %[[CONTROLLED_X:.*]] = qco.x %[[TARGET_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[CONTROLLED_X]]
    // CHECK: } {{.*}}catalyst.control_values = array<i1: false>{{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[CONTROL:.*]] = qco.x %[[CONTROL_OUT]] {{.*}}catalyst.negative_control_wrapper{{.*}} : !qco.qubit -> !qco.qubit
    %false = arith.constant false
    %controlled, %control = quantum.custom "PauliX"() %id ctrls(%q1) ctrlvals(%false) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[TOFFOLI_CONTROLS:.*]]:2, %[[TOFFOLI_TARGET:.*]] = qco.ctrl(%[[CONTROL]], %[[Q2]]) targets (%[[TOFFOLI_ARG:.*]] = %[[TARGET_OUT]]) {
    // CHECK: %[[TOFFOLI_X:.*]] = qco.x %[[TOFFOLI_ARG]] {{.*}}catalyst.gate_name = "Toffoli"{{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[TOFFOLI_X]]
    // CHECK: } {{.*}}catalyst.control_values = array<i1: true, true>{{.*}}catalyst.native_control_count = 2 : i64{{.*}} : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit, !qco.qubit}, {!qco.qubit})
    %toffoli:3 = quantum.custom "Toffoli"() %control, %q2, %controlled : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: qco.dealloc %[[TOFFOLI_TARGET]] : !qco.qubit
    // CHECK: qco.dealloc %[[TOFFOLI_CONTROLS]]#0 : !qco.qubit
    // CHECK: qco.dealloc %[[TOFFOLI_CONTROLS]]#1 : !qco.qubit
    %reg0 = quantum.insert %qreg[0], %toffoli#2 : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %toffoli#0 : !quantum.reg, !quantum.bit
    %reg2 = quantum.insert %reg1[2], %toffoli#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %reg2 : !quantum.reg
    return
  }
}
