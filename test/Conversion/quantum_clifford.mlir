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
    // CHECK: qco.alloc("qreg0", 2, 0)
    // CHECK: qco.alloc("qreg0", 2, 1)
    %qreg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit

    // CHECK: qco.h
    // CHECK: qco.sx
    // CHECK: qco.inv
    // CHECK: qco.sx
    // CHECK: qco.s
    // CHECK: qco.inv
    // CHECK: qco.s
    // CHECK: qco.t
    // CHECK: qco.inv
    // CHECK: qco.t
    %h = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %sx = quantum.custom "SX"() %h : !quantum.bit
    %sxdg = quantum.custom "SX"() %sx adj : !quantum.bit
    %s = quantum.custom "S"() %sxdg : !quantum.bit
    %sdg = quantum.custom "S"() %s adj : !quantum.bit
    %t = quantum.custom "T"() %sdg : !quantum.bit
    %tdg = quantum.custom "T"() %t adj : !quantum.bit

    // CHECK: qco.ctrl
    // CHECK: qco.h
    %true = arith.constant true
    %controlled, %control = quantum.custom "Hadamard"() %tdg ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // CHECK: qco.dealloc
    // CHECK: qco.dealloc
    %reg0 = quantum.insert %qreg[0], %controlled : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %control : !quantum.reg, !quantum.bit
    quantum.dealloc %reg1 : !quantum.reg
    return
  }
}
