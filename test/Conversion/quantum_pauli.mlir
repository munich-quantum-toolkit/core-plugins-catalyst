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
  // CHECK-LABEL: func.func @testCatalystQuantumToQCOPauliGates
  func.func @testCatalystQuantumToQCOPauliGates() {
    // CHECK: qco.alloc("qreg0", 3, 0)
    // CHECK: qco.alloc("qreg0", 3, 1)
    // CHECK: qco.alloc("qreg0", 3, 2)
    %qreg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[2] : !quantum.reg -> !quantum.bit

    // CHECK: qco.x
    // CHECK: qco.y
    // CHECK: qco.z
    // CHECK: qco.id
    %x = quantum.custom "PauliX"() %q0 : !quantum.bit
    %y = quantum.custom "PauliY"() %x : !quantum.bit
    %z = quantum.custom "PauliZ"() %y : !quantum.bit
    %id = quantum.custom "Identity"() %z : !quantum.bit

    // CHECK: qco.ctrl
    // CHECK: qco.x
    // CHECK: catalyst.control_values = array<i1: false>
    %false = arith.constant false
    %controlled, %control = quantum.custom "PauliX"() %id ctrls(%q1) ctrlvals(%false) : !quantum.bit ctrls !quantum.bit

    // CHECK: qco.ctrl
    // CHECK: qco.x {{.*}}catalyst.gate_name = "Toffoli"
    %toffoli:3 = quantum.custom "Toffoli"() %control, %q2, %controlled : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: qco.dealloc
    // CHECK: qco.dealloc
    // CHECK: qco.dealloc
    %reg0 = quantum.insert %qreg[0], %toffoli#2 : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %toffoli#0 : !quantum.reg, !quantum.bit
    %reg2 = quantum.insert %reg1[2], %toffoli#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %reg2 : !quantum.reg
    return
  }
}
