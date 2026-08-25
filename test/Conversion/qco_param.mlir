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
// Parameterized gates RX/RY/RZ, PhaseShift and controlled variants
// Tests both static constants and dynamic parameters
// Groups: Allocation & extraction / Static params / Dynamic params / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumParameterizedGates(
  // CHECK-SAME: %[[PHI:.*]]: f64)
  func.func @testQCOToCatalystQuantumParameterizedGates(%phi: f64) {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[THETA:.*]] = arith.constant 3.000000e-01 : f64
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}3)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    // CHECK: %[[Q2:.*]] = quantum.extract %[[REG]][{{ *}}2]
    // Prepare qubits
    %theta = arith.constant 3.000000e-01 : f64
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // --- Static parameters --------------------------------------------------------------------
    // CHECK: %[[RX:.*]] = quantum.custom "RX"(%[[THETA]]) %[[Q0]] : !quantum.bit
    // CHECK: %[[RY:.*]] = quantum.custom "RY"(%[[THETA]]) %[[RX]] : !quantum.bit
    // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%[[THETA]]) %[[RY]] : !quantum.bit
    // CHECK: %[[P:.*]] = quantum.custom "PhaseShift"(%[[THETA]]) %[[RZ]] : !quantum.bit
    // CHECK: quantum.gphase(%[[THETA]]) adj
    %rx = qco.rx(%theta) %q0 : !qco.qubit -> !qco.qubit
    %ry = qco.ry(%theta) %rx : !qco.qubit -> !qco.qubit
    %rz = qco.rz(%theta) %ry : !qco.qubit -> !qco.qubit
    %phase = qco.p(%theta) %rz : !qco.qubit -> !qco.qubit
    qco.gphase(%theta)

    // --- Dynamic parameters (runtime values) --------------------------------------------------
    // CHECK: %[[DRX:.*]] = quantum.custom "RX"(%[[PHI]]) %[[P]] : !quantum.bit
    // CHECK: %[[DRY:.*]] = quantum.custom "RY"(%[[PHI]]) %[[DRX]] : !quantum.bit
    // CHECK: %[[DRZ:.*]] = quantum.custom "RZ"(%[[PHI]]) %[[DRY]] : !quantum.bit
    // CHECK: %[[DP:.*]] = quantum.custom "PhaseShift"(%[[PHI]]) %[[DRZ]] : !quantum.bit
    // CHECK: quantum.gphase(%[[PHI]]) adj
    %drx = qco.rx(%phi) %phase : !qco.qubit -> !qco.qubit
    %dry = qco.ry(%phi) %drx : !qco.qubit -> !qco.qubit
    %drz = qco.rz(%phi) %dry : !qco.qubit -> !qco.qubit
    %dphase = qco.p(%phi) %drz : !qco.qubit -> !qco.qubit
    qco.gphase(%phi)

    // --- Controlled with static parameters ----------------------------------------------------
    // CHECK: %[[TRUE0:.*]] = arith.constant true
    // CHECK: %[[TRUE1:.*]] = arith.constant true
    // CHECK: %[[CRX2_T:.*]], %[[CRX2_C:.*]]:2 = quantum.custom "RX"(%[[THETA]]) %[[DP]] ctrls(%[[Q1]], %[[Q2]]) ctrlvals(%[[TRUE0]], %[[TRUE1]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    // CHECK: %[[CRX:.*]]:2 = quantum.custom "CRX"(%[[THETA]]) %[[CRX2_C]]#0, %[[CRX2_T]] : !quantum.bit, !quantum.bit
    // CHECK: %[[CRY:.*]]:2 = quantum.custom "CRY"(%[[THETA]]) %[[CRX]]#0, %[[CRX]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[CRZ:.*]]:2 = quantum.custom "CRZ"(%[[THETA]]) %[[CRY]]#0, %[[CRY]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[CP:.*]]:2 = quantum.custom "ControlledPhaseShift"(%[[THETA]]) %[[CRZ]]#0, %[[CRZ]]#1 : !quantum.bit, !quantum.bit
    %controls:2, %target = qco.ctrl(%q1, %q2) targets(%arg = %dphase) {
      %out = qco.rx(%theta) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit, !qco.qubit}, {!qco.qubit}) -> ({!qco.qubit, !qco.qubit}, {!qco.qubit})
    %crxc, %crx = qco.ctrl(%controls#0) targets(%arg = %target) {
      %out = qco.rx(%theta) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %cryc, %cry = qco.ctrl(%crxc) targets(%arg = %crx) {
      %out = qco.ry(%theta) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %crzc, %crz = qco.ctrl(%cryc) targets(%arg = %cry) {
      %out = qco.rz(%theta) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %cpc, %cp = qco.ctrl(%crzc) targets(%arg = %crz) {
      %out = qco.p(%theta) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    // --- Controlled with dynamic parameters ---------------------------------------------------
    // CHECK: %[[DCRX:.*]]:2 = quantum.custom "CRX"(%[[PHI]]) %[[CP]]#0, %[[CP]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[DCRY:.*]]:2 = quantum.custom "CRY"(%[[PHI]]) %[[DCRX]]#0, %[[DCRX]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[DCRZ:.*]]:2 = quantum.custom "CRZ"(%[[PHI]]) %[[DCRY]]#0, %[[DCRY]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[DCP:.*]]:2 = quantum.custom "ControlledPhaseShift"(%[[PHI]]) %[[DCRZ]]#0, %[[DCRZ]]#1 : !quantum.bit, !quantum.bit
    %dcrxc, %dcrx = qco.ctrl(%cpc) targets(%arg = %cp) {
      %out = qco.rx(%phi) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %dcryc, %dcry = qco.ctrl(%dcrxc) targets(%arg = %dcrx) {
      %out = qco.ry(%phi) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %dcrzc, %dcrz = qco.ctrl(%dcryc) targets(%arg = %dcry) {
      %out = qco.rz(%phi) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    %dcpc, %dcp = qco.ctrl(%dcrzc) targets(%arg = %dcrz) {
      %out = qco.p(%phi) %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    // --- Reinsertion ---------------------------------------------------------------------------
    qco.dealloc %dcp : !qco.qubit
    qco.dealloc %dcpc : !qco.qubit
    qco.dealloc %controls#1 : !qco.qubit
    // CHECK: %[[REG0:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[DCP]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG0]][{{ *}}1], %[[DCP]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG2:.*]] = quantum.insert %[[REG1]][{{ *}}2], %[[CRX2_C]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG2]] : !quantum.reg
    // Release qubits
    return
  }
}
