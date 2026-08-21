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
// RUN:   %s | FileCheck %s

module {
  // Core QCO gates without direct Catalyst equivalents are decomposed into
  // Catalyst operations supported by the forward bridge.
  // CHECK-LABEL: func.func @lower_dcx_and_rzx(
  // CHECK-SAME: %[[THETA:.*]]: f64)
  func.func @lower_dcx_and_rzx(%theta: f64) {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit

    // CHECK: quantum.custom "CNOT"
    // CHECK: quantum.custom "CNOT"
    %dcx0, %dcx1 = qco.dcx %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // CHECK: quantum.custom "Hadamard"
    // CHECK: quantum.custom "IsingZZ"(%[[THETA]])
    // CHECK: quantum.custom "Hadamard"
    %rzx0, %rzx1 = qco.rzx(%theta) %dcx0, %dcx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    qco.dealloc %rzx0 : !qco.qubit
    qco.dealloc %rzx1 : !qco.qubit
    return
  }

  // CHECK-LABEL: func.func @lower_u_u2_and_r(
  // CHECK-SAME: %[[THETA:.*]]: f64, %[[PHI:.*]]: f64, %[[LAMBDA:.*]]: f64)
  func.func @lower_u_u2_and_r(%theta: f64, %phi: f64, %lambda: f64) {
    %q = qco.alloc : !qco.qubit

    // CHECK: quantum.custom "RZ"(%[[LAMBDA]])
    // CHECK: quantum.custom "RY"(%[[THETA]])
    // CHECK: quantum.custom "RZ"(%[[PHI]])
    // CHECK: quantum.gphase
    %u = qco.u(%theta, %phi, %lambda) %q : !qco.qubit -> !qco.qubit

    // CHECK: quantum.custom "RZ"(%[[LAMBDA]])
    // CHECK: quantum.custom "RY"
    // CHECK: quantum.custom "RZ"(%[[PHI]])
    // CHECK: quantum.gphase
    %u2 = qco.u2(%phi, %lambda) %u : !qco.qubit -> !qco.qubit

    // CHECK: quantum.custom "RZ"
    // CHECK: quantum.custom "RY"(%[[THETA]])
    // CHECK: quantum.custom "RZ"
    %r = qco.r(%theta, %phi) %u2 : !qco.qubit -> !qco.qubit

    qco.dealloc %r : !qco.qubit
    return
  }

  // CHECK-LABEL: func.func @lower_xx_plus_minus_yy(
  // CHECK-SAME: %[[THETA:.*]]: f64, %[[BETA:.*]]: f64)
  func.func @lower_xx_plus_minus_yy(%theta: f64, %beta: f64) {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit

    // CHECK: quantum.custom "RZ"
    // CHECK: quantum.custom "IsingXY"(%[[THETA]])
    // CHECK: quantum.custom "RZ"
    %plus0, %plus1 = qco.xx_plus_yy(%theta, %beta) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // CHECK: quantum.custom "PauliX"
    // CHECK: quantum.custom "RZ"
    // CHECK: quantum.custom "IsingXY"(%[[THETA]])
    // CHECK: quantum.custom "RZ"
    // CHECK: quantum.custom "PauliX"
    %minus0, %minus1 = qco.xx_minus_yy(%theta, %beta) %plus0, %plus1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    qco.dealloc %minus0 : !qco.qubit
    qco.dealloc %minus1 : !qco.qubit
    return
  }

  // Inversion reverses each decomposition while the enclosing control is
  // threaded through every emitted gate.
  // CHECK-LABEL: func.func @lower_inverted_controlled_u_and_r(
  // CHECK-SAME: %[[THETA:.*]]: f64, %[[PHI:.*]]: f64, %[[LAMBDA:.*]]: f64)
  func.func @lower_inverted_controlled_u_and_r(%theta: f64, %phi: f64, %lambda: f64) {
    // CHECK: %[[U_CONTROL:.*]] = quantum.alloc_qb
    // CHECK: %[[U_TARGET:.*]] = quantum.alloc_qb
    // CHECK: %[[R_CONTROL:.*]] = quantum.alloc_qb
    // CHECK: %[[R_TARGET:.*]] = quantum.alloc_qb
    %u_control = qco.alloc : !qco.qubit
    %u_target = qco.alloc : !qco.qubit
    %r_control = qco.alloc : !qco.qubit
    %r_target = qco.alloc : !qco.qubit

    // CHECK: %[[U_PHASE_CONTROL:.*]] = quantum.gphase({{.*}}) ctrls(%[[U_CONTROL]])
    // CHECK: %[[U_RZ_PHI:.*]], %[[U_RZ_PHI_CTRL:.*]] = quantum.custom "RZ"(%[[PHI]]) %[[U_TARGET]] adj {{.*}}ctrls(%[[U_PHASE_CONTROL]])
    // CHECK: %[[U_RY:.*]], %[[U_RY_CTRL:.*]] = quantum.custom "RY"(%[[THETA]]) %[[U_RZ_PHI]] adj {{.*}}ctrls(%[[U_RZ_PHI_CTRL]])
    // CHECK: %[[U_RZ_LAMBDA:.*]], %[[U_RZ_LAMBDA_CTRL:.*]] = quantum.custom "RZ"(%[[LAMBDA]]) %[[U_RY]] adj {{.*}}ctrls(%[[U_RY_CTRL]])
    %u_control_out, %u_target_out = qco.ctrl(%u_control) targets(%ctrl_arg = %u_target) {
      %inv_out = qco.inv (%inv_arg = %ctrl_arg) {
        %out = qco.u(%theta, %phi, %lambda) %inv_arg : !qco.qubit -> !qco.qubit
        qco.yield %out
      } : {!qco.qubit} -> {!qco.qubit}
      qco.yield %inv_out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    // CHECK: %[[PI_HALF:.*]] = arith.constant 1.5707963267948966
    // CHECK: %[[FIRST_ANGLE:.*]] = arith.subf %[[PI_HALF]], %[[PHI]]
    // CHECK: %[[LAST_ANGLE:.*]] = arith.subf %[[PHI]], %[[PI_HALF]]
    // CHECK: %[[R_RZ_LAST:.*]], %[[R_RZ_LAST_CTRL:.*]] = quantum.custom "RZ"(%[[LAST_ANGLE]]) %[[R_TARGET]] adj {{.*}}ctrls(%[[R_CONTROL]])
    // CHECK: %[[R_RY:.*]], %[[R_RY_CTRL:.*]] = quantum.custom "RY"(%[[THETA]]) %[[R_RZ_LAST]] adj {{.*}}ctrls(%[[R_RZ_LAST_CTRL]])
    // CHECK: %[[R_RZ_FIRST:.*]], %[[R_RZ_FIRST_CTRL:.*]] = quantum.custom "RZ"(%[[FIRST_ANGLE]]) %[[R_RY]] adj {{.*}}ctrls(%[[R_RY_CTRL]])
    %r_control_out, %r_target_out = qco.ctrl(%r_control) targets(%ctrl_arg = %r_target) {
      %inv_out = qco.inv (%inv_arg = %ctrl_arg) {
        %out = qco.r(%theta, %phi) %inv_arg : !qco.qubit -> !qco.qubit
        qco.yield %out
      } : {!qco.qubit} -> {!qco.qubit}
      qco.yield %inv_out
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    qco.dealloc %u_control_out : !qco.qubit
    qco.dealloc %u_target_out : !qco.qubit
    qco.dealloc %r_control_out : !qco.qubit
    qco.dealloc %r_target_out : !qco.qubit
    return
  }

  // CHECK-LABEL: func.func @lower_inverted_controlled_dcx_and_rzx(
  // CHECK-SAME: %[[THETA:.*]]: f64)
  func.func @lower_inverted_controlled_dcx_and_rzx(%theta: f64) {
    // CHECK: %[[DCX_CONTROL:.*]] = quantum.alloc_qb
    // CHECK: %[[DCX_Q0:.*]] = quantum.alloc_qb
    // CHECK: %[[DCX_Q1:.*]] = quantum.alloc_qb
    // CHECK: %[[RZX_CONTROL:.*]] = quantum.alloc_qb
    // CHECK: %[[RZX_Q0:.*]] = quantum.alloc_qb
    // CHECK: %[[RZX_Q1:.*]] = quantum.alloc_qb
    %dcx_control = qco.alloc : !qco.qubit
    %dcx_q0 = qco.alloc : !qco.qubit
    %dcx_q1 = qco.alloc : !qco.qubit
    %rzx_control = qco.alloc : !qco.qubit
    %rzx_q0 = qco.alloc : !qco.qubit
    %rzx_q1 = qco.alloc : !qco.qubit

    // The inverted DCX starts with the opposite CNOT direction. The updated
    // control and both updated qubits feed the second CNOT.
    // CHECK: %[[DCX_FIRST:.*]]:2, %[[DCX_FIRST_CTRL:.*]] = quantum.custom "CNOT"() %[[DCX_Q1]], %[[DCX_Q0]] {{.*}}ctrls(%[[DCX_CONTROL]])
    // CHECK: %[[DCX_SECOND:.*]]:2, %[[DCX_SECOND_CTRL:.*]] = quantum.custom "CNOT"() %[[DCX_FIRST]]#1, %[[DCX_FIRST]]#0 {{.*}}ctrls(%[[DCX_FIRST_CTRL]])
    %dcx_control_out, %dcx_out0, %dcx_out1 = qco.ctrl(%dcx_control) targets(%arg0 = %dcx_q0, %arg1 = %dcx_q1) {
      %inv0, %inv1 = qco.inv (%inv_arg0 = %arg0, %inv_arg1 = %arg1) {
        %out0, %out1 = qco.dcx %inv_arg0, %inv_arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        qco.yield %out0, %out1
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      qco.yield %inv0, %inv1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    // CHECK: %[[RZX_H0:.*]] = quantum.custom "Hadamard"() %[[RZX_Q1]]
    // CHECK: %[[RZX_RZZ:.*]]:2, %[[RZX_RZZ_CTRL:.*]] = quantum.custom "IsingZZ"(%[[THETA]]) %[[RZX_Q0]], %[[RZX_H0]] adj {{.*}}ctrls(%[[RZX_CONTROL]])
    // CHECK: %[[RZX_H1:.*]] = quantum.custom "Hadamard"() %[[RZX_RZZ]]#1
    %rzx_control_out, %rzx_out0, %rzx_out1 = qco.ctrl(%rzx_control) targets(%arg0 = %rzx_q0, %arg1 = %rzx_q1) {
      %inv0, %inv1 = qco.inv (%inv_arg0 = %arg0, %inv_arg1 = %arg1) {
        %out0, %out1 = qco.rzx(%theta) %inv_arg0, %inv_arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        qco.yield %out0, %out1
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      qco.yield %inv0, %inv1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    qco.dealloc %dcx_control_out : !qco.qubit
    qco.dealloc %dcx_out0 : !qco.qubit
    qco.dealloc %dcx_out1 : !qco.qubit
    qco.dealloc %rzx_control_out : !qco.qubit
    qco.dealloc %rzx_out0 : !qco.qubit
    qco.dealloc %rzx_out1 : !qco.qubit
    return
  }

  // CHECK-LABEL: func.func @lower_inverted_controlled_xx_plus_yy(
  // CHECK-SAME: %[[THETA:.*]]: f64, %[[BETA:.*]]: f64)
  func.func @lower_inverted_controlled_xx_plus_yy(%theta: f64, %beta: f64) {
    // CHECK: %[[CONTROL:.*]] = quantum.alloc_qb
    // CHECK: %[[Q0:.*]] = quantum.alloc_qb
    // CHECK: %[[Q1:.*]] = quantum.alloc_qb
    %control = qco.alloc : !qco.qubit
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit

    // CHECK: %[[PI:.*]] = arith.constant 3.141592653589793
    // CHECK: %[[PI_MINUS_BETA:.*]] = arith.subf %[[PI]], %[[BETA]]
    // CHECK: %[[BETA_MINUS_PI:.*]] = arith.subf %[[BETA]], %[[PI]]
    // CHECK: %[[FIRST_RZ:.*]], %[[FIRST_RZ_CTRL:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA]]) %[[Q1]] {{.*}}ctrls(%[[CONTROL]])
    // CHECK: %[[ISING:.*]]:2, %[[ISING_CTRL:.*]] = quantum.custom "IsingXY"(%[[THETA]]) %[[Q0]], %[[FIRST_RZ]] adj {{.*}}ctrls(%[[FIRST_RZ_CTRL]])
    // CHECK: quantum.custom "RZ"(%[[BETA_MINUS_PI]]) %[[ISING]]#1 {{.*}}ctrls(%[[ISING_CTRL]])
    %control_out, %out0, %out1 = qco.ctrl(%control) targets(%arg0 = %q0, %arg1 = %q1) {
      %inv0, %inv1 = qco.inv (%inv_arg0 = %arg0, %inv_arg1 = %arg1) {
        %gate0, %gate1 = qco.xx_plus_yy(%theta, %beta) %inv_arg0, %inv_arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        qco.yield %gate0, %gate1
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      qco.yield %inv0, %inv1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    qco.dealloc %control_out : !qco.qubit
    qco.dealloc %out0 : !qco.qubit
    qco.dealloc %out1 : !qco.qubit
    return
  }

  // A constant beta=pi uses Catalyst's direct IsingXY gate instead of the
  // general XX+YY decomposition.
  // CHECK-LABEL: func.func @lower_direct_ising_xy(
  // CHECK-SAME: %[[THETA:.*]]: f64)
  func.func @lower_direct_ising_xy(%theta: f64) {
    // CHECK: %[[Q0:.*]] = quantum.alloc_qb
    // CHECK: %[[Q1:.*]] = quantum.alloc_qb
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %pi = arith.constant 3.141592653589793 : f64

    // CHECK: %[[ISING:.*]]:2 = quantum.custom "IsingXY"(%[[THETA]]) %[[Q0]], %[[Q1]]
    %out0, %out1 = qco.xx_plus_yy(%theta, %pi) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    qco.dealloc %out0 : !qco.qubit
    qco.dealloc %out1 : !qco.qubit
    return
  }
}
