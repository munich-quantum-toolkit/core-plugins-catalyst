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
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %s | FileCheck %s --implicit-check-not='qco.'

// ============================================================================
// Clifford + T and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumCliffordT
  func.func @testQCOToCatalystQuantumCliffordT() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}2)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    // Prepare qubits
    %q0 = qco.alloc("input", 2, 0) : !qco.qubit
    %q1 = qco.alloc("input", 2, 1) : !qco.qubit

    // --- Uncontrolled Clifford+T gates ---------------------------------------------------------
    // CHECK: %[[ID:.*]] = quantum.custom "Identity"() %[[Q0]] : !quantum.bit
    // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[ID]] : !quantum.bit
    // SX and SXdg are lowered to runtime-supported RX and global phase operations.
    // CHECK: %[[PI_HALF:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[MINUS_PI_QUARTER:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[SX:.*]] = quantum.custom "RX"(%[[PI_HALF]]) %[[H]] : !quantum.bit
    // CHECK: quantum.gphase(%[[MINUS_PI_QUARTER]]){{$}}
    // CHECK: %[[PI_HALF_ADJ:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[MINUS_PI_QUARTER_ADJ:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[SXDG:.*]] = quantum.custom "RX"(%[[PI_HALF_ADJ]]) %[[SX]] adj : !quantum.bit
    // CHECK: quantum.gphase(%[[MINUS_PI_QUARTER_ADJ]]) adj
    // CHECK: %[[S:.*]] = quantum.custom "S"() %[[SXDG]] : !quantum.bit
    // CHECK: %[[SDG:.*]] = quantum.custom "S"() %[[S]] adj : !quantum.bit
    // CHECK: %[[T:.*]] = quantum.custom "T"() %[[SDG]] : !quantum.bit
    // CHECK: %[[TDG:.*]] = quantum.custom "T"() %[[T]] adj : !quantum.bit
    %id = qco.id %q0 : !qco.qubit -> !qco.qubit
    %h = qco.h %id : !qco.qubit -> !qco.qubit
    %sx = qco.sx %h : !qco.qubit -> !qco.qubit
    %sxdg = qco.sxdg %sx : !qco.qubit -> !qco.qubit
    %s = qco.s %sxdg : !qco.qubit -> !qco.qubit
    %sdg = qco.sdg %s : !qco.qubit -> !qco.qubit
    %t = qco.t %sdg : !qco.qubit -> !qco.qubit
    %tdg = qco.tdg %t : !qco.qubit -> !qco.qubit

    // --- Controlled Clifford+T gates -----------------------------------------------------------
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CH_T:.*]], %[[CH_C:.*]] = quantum.custom "Hadamard"() %[[TDG]] ctrls(%[[Q1]]) ctrlvals(%[[TRUE]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CSX_ANGLE:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[CSX_PHASE:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[TRUE_CSX:.*]] = arith.constant true
    // CHECK: %[[CSX_T:.*]], %[[CSX_C0:.*]] = quantum.custom "RX"(%[[CSX_ANGLE]]) %[[CH_T]] ctrls(%[[CH_C]]) ctrlvals(%[[TRUE_CSX]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CSX_PHASE:.*]] = arith.constant true
    // CHECK: %[[CSX_C:.*]] = quantum.gphase(%[[CSX_PHASE]]) ctrls(%[[CSX_C0]]) ctrlvals(%[[TRUE_CSX_PHASE]]) : ctrls !quantum.bit
    // CHECK: %[[CSXDG_ANGLE:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK: %[[CSXDG_PHASE:.*]] = arith.constant -0.78539816339744828 : f64
    // CHECK: %[[TRUE_CSXDG:.*]] = arith.constant true
    // CHECK: %[[CSXDG_T:.*]], %[[CSXDG_C0:.*]] = quantum.custom "RX"(%[[CSXDG_ANGLE]]) %[[CSX_T]] adj ctrls(%[[CSX_C]]) ctrlvals(%[[TRUE_CSXDG]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CSXDG_PHASE:.*]] = arith.constant true
    // CHECK: %[[CSXDG_C:.*]] = quantum.gphase(%[[CSXDG_PHASE]]) adj ctrls(%[[CSXDG_C0]]) ctrlvals(%[[TRUE_CSXDG_PHASE]]) : ctrls !quantum.bit
    // CHECK: %[[TRUE_CS:.*]] = arith.constant true
    // CHECK: %[[CS_T:.*]], %[[CS_C:.*]] = quantum.custom "S"() %[[CSXDG_T]] ctrls(%[[CSXDG_C]]) ctrlvals(%[[TRUE_CS]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CSDG:.*]] = arith.constant true
    // CHECK: %[[CSDG_T:.*]], %[[CSDG_C:.*]] = quantum.custom "S"() %[[CS_T]] adj ctrls(%[[CS_C]]) ctrlvals(%[[TRUE_CSDG]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CT:.*]] = arith.constant true
    // CHECK: %[[CT_T:.*]], %[[CT_C:.*]] = quantum.custom "T"() %[[CSDG_T]] ctrls(%[[CSDG_C]]) ctrlvals(%[[TRUE_CT]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE_CTDG:.*]] = arith.constant true
    // CHECK: %[[CTDG_T:.*]], %[[CTDG_C:.*]] = quantum.custom "T"() %[[CT_T]] adj ctrls(%[[CT_C]]) ctrlvals(%[[TRUE_CTDG]]) : !quantum.bit ctrls !quantum.bit
    %chc, %ch = qco.ctrl(%q1) targets(%arg = %tdg) {
      %out = qco.h %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %csxc, %csx = qco.ctrl(%chc) targets(%arg = %ch) {
      %out = qco.sx %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %csxdgc, %csxdg = qco.ctrl(%csxc) targets(%arg = %csx) {
      %out = qco.sxdg %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %csc, %cs = qco.ctrl(%csxdgc) targets(%arg = %csxdg) {
      %out = qco.s %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %csdgc, %csdg = qco.ctrl(%csc) targets(%arg = %cs) {
      %out = qco.sdg %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %ctc, %ct = qco.ctrl(%csdgc) targets(%arg = %csdg) {
      %out = qco.t %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %ctdgc, %ctdg = qco.ctrl(%ctc) targets(%arg = %ct) {
      %out = qco.tdg %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    // --- Barrier, measurement, and reinsertion -------------------------------------------------
    %barrier:2 = qco.barrier %ctdg, %ctdgc : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %measured, %mres = qco.measure %barrier#0 : !qco.qubit
    qco.dealloc %measured : !qco.qubit
    qco.dealloc %barrier#1 : !qco.qubit
    // CHECK-NOT: quantum.custom "Barrier"
    // CHECK: %[[MRES:.*]], %[[MEASURED:.*]] = quantum.measure %[[CTDG_T]] : i1, !quantum.bit
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[MEASURED]] : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[CTDG_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG1]] : !quantum.reg
    // Release qubits
    return
  }
}
