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
// RUN:   %s | FileCheck %s

module {
  // A private call carries preserved quantum reads through Core conversions
  // without exposing a pure, operand-free value that CSE could merge.
  // CHECK: func.func private @[[BRIDGE:__mqt_catalyst_qco_qubit_bridge.*]](!qco.qubit) -> !quantum.bit attributes {catalyst.qco_qubit_bridge}
  // CHECK: func.func private @[[GATE_HINT:__mqt_catalyst_qco_gate_hint_bridge.*]]() attributes
  // CHECK-SAME: catalyst.gate_name = "PauliX"
  // CHECK-SAME: catalyst.native_control_count = 0 : i64
  // CHECK-SAME: catalyst.qco_gate_hint_bridge

  // CHECK-LABEL: func.func @convert_static_register(
  // CHECK-SAME: %[[THETA:.*]]: f64)
  func.func @convert_static_register(%theta: f64) -> i1 {
    // One Catalyst register becomes one metadata-tagged QCO allocation per qubit.
    // CHECK: %[[Q0:.*]] = qco.alloc("qreg0", 3, 0) : !qco.qubit
    // CHECK: %[[Q1:.*]] = qco.alloc("qreg0", 3, 1) : !qco.qubit
    // CHECK: %[[Q2:.*]] = qco.alloc("qreg0", 3, 2) : !qco.qubit
    // CHECK-NOT: quantum.alloc
    // CHECK-NOT: quantum.extract
    %reg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit

    // Gates and dynamic parameters retain their SSA order.
    // CHECK: %[[H:.*]] = qco.h %[[Q0]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: %[[RX:.*]] = qco.inv (%[[RX_ARG:.*]] = %[[H]]) {
    // CHECK: %[[RX_BODY:.*]] = qco.rx(%[[THETA]]) %[[RX_ARG]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[RX_BODY]]
    // CHECK: } {{.*}} : {!qco.qubit} -> {!qco.qubit}
    %h = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %rx = quantum.custom "RX"(%theta) %h adj : !quantum.bit

    // A negative control is represented without losing semantics by conjugating
    // the native positive QCO control with Pauli-X.
    // CHECK: %[[NEG:.*]] = qco.x %[[Q1]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK-NEXT: call @[[GATE_HINT]]() : () -> ()
    // CHECK: %[[CTRL:.*]], %[[TARGET:.*]] = qco.ctrl(%[[NEG]]) targets {{ *}}(%[[ALIAS:.*]] = %[[RX]]) {
    // CHECK: %[[X:.*]] = qco.x %[[ALIAS]] {{.*}} : !qco.qubit -> !qco.qubit
    // CHECK: qco.yield %[[X]]
    // CHECK: } {{.*}} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    // CHECK: %[[RESTORED:.*]] = qco.x %[[CTRL]] {{.*}} : !qco.qubit -> !qco.qubit
    %false = arith.constant false
    %x, %control = quantum.custom "PauliX"() %rx ctrls(%q1) ctrlvals(%false) : !quantum.bit ctrls !quantum.bit

    // Two-qubit gates, barriers, and global phases are preserved.
    // CHECK: %[[SWAP0:.*]], %[[SWAP1:.*]] = qco.swap %[[TARGET]], %[[Q2]] {{.*}} : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    // CHECK: %[[BARRIER:.*]]:2 = qco.barrier %[[SWAP0]], %[[SWAP1]] {{.*}} : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    // CHECK: qco.gphase(%[[THETA]]) {{.*}}
    %swap:2 = quantum.custom "SWAP"() %x, %q2 : !quantum.bit, !quantum.bit
    %barrier:2 = quantum.custom "Barrier"() %swap#0, %swap#1 : !quantum.bit, !quantum.bit
    quantum.gphase(%theta)

    // Measurement returns the updated qubit first in QCO and the classical bit second.
    // CHECK: %[[MEASURED:.*]], %[[RESULT:.*]] = qco.measure %[[BARRIER]]#0 : !qco.qubit
    // CHECK: qco.dealloc %[[MEASURED]] : !qco.qubit
    // CHECK: qco.dealloc %[[RESTORED]] : !qco.qubit
    // CHECK: qco.dealloc %[[BARRIER]]#1 : !qco.qubit
    // CHECK: return %[[RESULT]] : i1
    %result, %measured = quantum.measure %barrier#0 : i1, !quantum.bit
    %reg0 = quantum.insert %reg[0], %measured : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %control : !quantum.reg, !quantum.bit
    %reg2 = quantum.insert %reg1[2], %barrier#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %reg2 : !quantum.reg
    return %result : i1
  }

  // Catalyst operations without a QCO equivalent remain available to the
  // surrounding Catalyst program.
  // CHECK-LABEL: func.func @preserve_device_and_observable_ops
  func.func @preserve_device_and_observable_ops() {
    // CHECK: quantum.device ["", "lightning.qubit", ""]
    // CHECK: %[[OBS_QUBIT:.*]] = call @[[BRIDGE]](%{{.*}}) : (!qco.qubit) -> !quantum.bit
    // CHECK: quantum.namedobs %[[OBS_QUBIT]][{{ *}}PauliZ]
    // CHECK: quantum.device_release
    quantum.device ["", "lightning.qubit", ""]
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %obs = quantum.namedobs %q[PauliZ] : !quantum.obs
    %updated = quantum.insert %reg[0], %q : !quantum.reg, !quantum.bit
    quantum.dealloc %updated : !quantum.reg
    quantum.device_release
    return
  }

  // Preserved names reserve their namespace before generated names are chosen.
  // CHECK-LABEL: func.func @unique_generated_register_name
  // CHECK: qco.alloc("qreg1", 1, 0)
  // CHECK: qco.alloc("qreg0", 1, 0)
  func.func @unique_generated_register_name() {
    %generated = quantum.alloc(1) : !quantum.reg
    %named = quantum.alloc(1) {mqt.qco_register_name = "qreg0"} : !quantum.reg
    quantum.dealloc %named : !quantum.reg
    quantum.dealloc %generated : !quantum.reg
    return
  }
}
