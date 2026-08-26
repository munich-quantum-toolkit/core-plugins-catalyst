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
// Ising-type gates and controlled variants
// Tests both static constants and dynamic parameters
// Groups: Allocation & extraction / Static params / Dynamic params / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testQCOToCatalystQuantumIsingGates(
  // CHECK-SAME: %[[THETA:.*]]: f64, %[[BETA:.*]]: f64)
  func.func @testQCOToCatalystQuantumIsingGates(%theta: f64, %beta: f64) {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[CST:.*]] = arith.constant 3.000000e-01 : f64
    // CHECK: %[[REG:.*]] = quantum.alloc({{ *}}3)
    // CHECK: %[[Q0:.*]] = quantum.extract %[[REG]][{{ *}}0]
    // CHECK: %[[Q1:.*]] = quantum.extract %[[REG]][{{ *}}1]
    // CHECK: %[[Q2:.*]] = quantum.extract %[[REG]][{{ *}}2]
    // Prepare qubits
    %cst = arith.constant 3.000000e-01 : f64
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q1 = qco.alloc("input", 3, 1) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit

    // --- Static parameters --------------------------------------------------------------------
    // CHECK: %[[PI0:.*]] = arith.constant 3.1415926535897931 : f64
    // CHECK: %[[PI_MINUS_BETA0:.*]] = arith.subf %[[PI0]], %[[CST]] : f64
    // CHECK: %[[BETA_MINUS_PI0:.*]] = arith.subf %[[CST]], %[[PI0]] : f64
    // CHECK: %[[RZ0:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA0]]) %[[Q1]] : !quantum.bit
    // CHECK: %[[XY_P:.*]]:2 = quantum.custom "IsingXY"(%[[CST]]) %[[Q0]], %[[RZ0]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI0]]) %[[XY_P]]#1 : !quantum.bit

    // CHECK: %[[PI1:.*]] = arith.constant 3.1415926535897931 : f64
    // CHECK: %[[PI_MINUS_BETA1:.*]] = arith.subf %[[PI1]], %[[CST]] : f64
    // CHECK: %[[BETA_MINUS_PI1:.*]] = arith.subf %[[CST]], %[[PI1]] : f64
    // CHECK: %[[X1:.*]] = quantum.custom "PauliX"() %[[XY_P]]#0 : !quantum.bit
    // CHECK: %[[RZ2:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA1]]) %[[RZ1]] : !quantum.bit
    // CHECK: %[[XY_M:.*]]:2 = quantum.custom "IsingXY"(%[[CST]]) %[[X1]], %[[RZ2]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ3:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI1]]) %[[XY_M]]#1 : !quantum.bit
    // CHECK: %[[X2:.*]] = quantum.custom "PauliX"() %[[XY_M]]#0 : !quantum.bit

    // CHECK: %[[XX_P:.*]]:2 = quantum.custom "IsingXX"(%[[CST]]) %[[X2]], %[[RZ3]] : !quantum.bit, !quantum.bit
    // CHECK: %[[YY_P:.*]]:2 = quantum.custom "IsingYY"(%[[CST]]) %[[XX_P]]#0, %[[XX_P]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ZZ_P1:.*]]:2 = quantum.custom "IsingZZ"(%[[CST]]) %[[YY_P]]#0, %[[YY_P]]#1 : !quantum.bit, !quantum.bit

    // CHECK: %[[H1U:.*]] = quantum.custom "Hadamard"() %[[ZZ_P1]]#1 : !quantum.bit
    // CHECK: %[[ZZ_P2:.*]]:2 = quantum.custom "IsingZZ"(%[[CST]]) %[[ZZ_P1]]#0, %[[H1U]] : !quantum.bit, !quantum.bit
    // CHECK: %[[H2U:.*]] = quantum.custom "Hadamard"() %[[ZZ_P2]]#1 : !quantum.bit
    %plus0, %plus1 = qco.xx_plus_yy(%cst, %cst) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %minus0, %minus1 = qco.xx_minus_yy(%cst, %cst) %plus0, %plus1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %xx0, %xx1 = qco.rxx(%cst) %minus0, %minus1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %yy0, %yy1 = qco.ryy(%cst) %xx0, %xx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %zz0, %zz1 = qco.rzz(%cst) %yy0, %yy1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %rzx0, %rzx1 = qco.rzx(%cst) %zz0, %zz1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // --- Controlled with static parameters ----------------------------------------------------
    // CHECK: %[[PI2:.*]] = arith.constant 3.1415926535897931 : f64
    // CHECK: %[[PI_MINUS_BETA2:.*]] = arith.subf %[[PI2]], %[[CST]] : f64
    // CHECK: %[[BETA_MINUS_PI2:.*]] = arith.subf %[[CST]], %[[PI2]] : f64
    // CHECK: %[[TRUE0:.*]] = arith.constant true
    // CHECK: %[[RZ_C0:.*]], %[[CTRL1A:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA2]]) %[[H2U]] ctrls(%[[Q2]]) ctrlvals(%[[TRUE0]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE1:.*]] = arith.constant true
    // CHECK: %[[XY_C:.*]]:2, %[[CTRL1B:.*]] = quantum.custom "IsingXY"(%[[CST]]) %[[ZZ_P2]]#0, %[[RZ_C0]] ctrls(%[[CTRL1A]]) ctrlvals(%[[TRUE1]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE2:.*]] = arith.constant true
    // CHECK: %[[RZ_C1:.*]], %[[CTRL1:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI2]]) %[[XY_C]]#1 ctrls(%[[CTRL1B]]) ctrlvals(%[[TRUE2]]) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[PI3:.*]] = arith.constant 3.1415926535897931 : f64
    // CHECK: %[[PI_MINUS_BETA3:.*]] = arith.subf %[[PI3]], %[[CST]] : f64
    // CHECK: %[[BETA_MINUS_PI3:.*]] = arith.subf %[[CST]], %[[PI3]] : f64
    // CHECK: %[[TRUE3:.*]] = arith.constant true
    // CHECK: %[[X1C:.*]], %[[CTRL2A:.*]] = quantum.custom "PauliX"() %[[XY_C]]#0 ctrls(%[[CTRL1]]) ctrlvals(%[[TRUE3]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE4:.*]] = arith.constant true
    // CHECK: %[[RZ_C2:.*]], %[[CTRL2B:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA3]]) %[[RZ_C1]] ctrls(%[[CTRL2A]]) ctrlvals(%[[TRUE4]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE5:.*]] = arith.constant true
    // CHECK: %[[XY_CM:.*]]:2, %[[CTRL2C:.*]] = quantum.custom "IsingXY"(%[[CST]]) %[[X1C]], %[[RZ_C2]] ctrls(%[[CTRL2B]]) ctrlvals(%[[TRUE5]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE6:.*]] = arith.constant true
    // CHECK: %[[RZ_C3:.*]], %[[CTRL2D:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI3]]) %[[XY_CM]]#1 ctrls(%[[CTRL2C]]) ctrlvals(%[[TRUE6]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE7:.*]] = arith.constant true
    // CHECK: %[[X2C:.*]], %[[CTRL2:.*]] = quantum.custom "PauliX"() %[[XY_CM]]#0 ctrls(%[[CTRL2D]]) ctrlvals(%[[TRUE7]]) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[TRUE8:.*]] = arith.constant true
    // CHECK: %[[XX_C:.*]]:2, %[[CTRL3:.*]] = quantum.custom "IsingXX"(%[[CST]]) %[[X2C]], %[[RZ_C3]] ctrls(%[[CTRL2]]) ctrlvals(%[[TRUE8]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE9:.*]] = arith.constant true
    // CHECK: %[[YY_C:.*]]:2, %[[CTRL4:.*]] = quantum.custom "IsingYY"(%[[CST]]) %[[XX_C]]#0, %[[XX_C]]#1 ctrls(%[[CTRL3]]) ctrlvals(%[[TRUE9]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE10:.*]] = arith.constant true
    // CHECK: %[[ZZ_C1:.*]]:2, %[[CTRL5:.*]] = quantum.custom "IsingZZ"(%[[CST]]) %[[YY_C]]#0, %[[YY_C]]#1 ctrls(%[[CTRL4]]) ctrlvals(%[[TRUE10]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // CHECK: %[[H1C:.*]] = quantum.custom "Hadamard"() %[[ZZ_C1]]#1 : !quantum.bit
    // CHECK: %[[TRUE11:.*]] = arith.constant true
    // CHECK: %[[ZZ_C2:.*]]:2, %[[CTRL6:.*]] = quantum.custom "IsingZZ"(%[[CST]]) %[[ZZ_C1]]#0, %[[H1C]] ctrls(%[[CTRL5]]) ctrlvals(%[[TRUE11]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[H2C:.*]] = quantum.custom "Hadamard"() %[[ZZ_C2]]#1 : !quantum.bit
    %cplusc, %cplus0, %cplus1 = qco.ctrl(%q2) targets(%arg0 = %rzx0, %arg1 = %rzx1) {
      %out0, %out1 = qco.xx_plus_yy(%cst, %cst) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %cminusc, %cminus0, %cminus1 = qco.ctrl(%cplusc) targets(%arg0 = %cplus0, %arg1 = %cplus1) {
      %out0, %out1 = qco.xx_minus_yy(%cst, %cst) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %cxxc, %cxx0, %cxx1 = qco.ctrl(%cminusc) targets(%arg0 = %cminus0, %arg1 = %cminus1) {
      %out0, %out1 = qco.rxx(%cst) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %cyyc, %cyy0, %cyy1 = qco.ctrl(%cxxc) targets(%arg0 = %cxx0, %arg1 = %cxx1) {
      %out0, %out1 = qco.ryy(%cst) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %czzc, %czz0, %czz1 = qco.ctrl(%cyyc) targets(%arg0 = %cyy0, %arg1 = %cyy1) {
      %out0, %out1 = qco.rzz(%cst) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %crzxc, %crzx0, %crzx1 = qco.ctrl(%czzc) targets(%arg0 = %czz0, %arg1 = %czz1) {
      %out0, %out1 = qco.rzx(%cst) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    // --- Dynamic parameters (runtime values) --------------------------------------------------
    // CHECK: %[[DXX:.*]]:2 = quantum.custom "IsingXX"(%[[THETA]]) %[[ZZ_C2]]#0, %[[H2C]] : !quantum.bit, !quantum.bit
    // CHECK: %[[DYY:.*]]:2 = quantum.custom "IsingYY"(%[[THETA]]) %[[DXX]]#0, %[[DXX]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[DZZ:.*]]:2 = quantum.custom "IsingZZ"(%[[THETA]]) %[[DYY]]#0, %[[DYY]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[PI4:.*]] = arith.constant 3.1415926535897931 : f64
    // CHECK: %[[PI_MINUS_BETA4:.*]] = arith.subf %[[PI4]], %[[BETA]] : f64
    // CHECK: %[[BETA_MINUS_PI4:.*]] = arith.subf %[[BETA]], %[[PI4]] : f64
    // CHECK: %[[RZ_D1:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA4]]) %[[DZZ]]#1 : !quantum.bit
    // CHECK: %[[DXY:.*]]:2 = quantum.custom "IsingXY"(%[[THETA]]) %[[DZZ]]#0, %[[RZ_D1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ_D2:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI4]]) %[[DXY]]#1 : !quantum.bit
    %dxx0, %dxx1 = qco.rxx(%theta) %crzx0, %crzx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %dyy0, %dyy1 = qco.ryy(%theta) %dxx0, %dxx1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %dzz0, %dzz1 = qco.rzz(%theta) %dyy0, %dyy1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
    %dplus0, %dplus1 = qco.xx_plus_yy(%theta, %beta) %dzz0, %dzz1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit

    // --- Controlled with dynamic parameters ---------------------------------------------------
    // CHECK: %[[TRUE12:.*]] = arith.constant true
    // CHECK: %[[DCXX:.*]]:2, %[[CTRL7:.*]] = quantum.custom "IsingXX"(%[[THETA]]) %[[DXY]]#0, %[[RZ_D2]] ctrls(%[[CTRL6]]) ctrlvals(%[[TRUE12]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE13:.*]] = arith.constant true
    // CHECK: %[[DCYY:.*]]:2, %[[CTRL8:.*]] = quantum.custom "IsingYY"(%[[THETA]]) %[[DCXX]]#0, %[[DCXX]]#1 ctrls(%[[CTRL7]]) ctrlvals(%[[TRUE13]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE14:.*]] = arith.constant true
    // CHECK: %[[DCZZ:.*]]:2, %[[CTRL9:.*]] = quantum.custom "IsingZZ"(%[[THETA]]) %[[DCYY]]#0, %[[DCYY]]#1 ctrls(%[[CTRL8]]) ctrlvals(%[[TRUE14]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[PI5:.*]] = arith.constant 3.1415926535897931 : f64
    // CHECK: %[[PI_MINUS_BETA5:.*]] = arith.subf %[[PI5]], %[[BETA]] : f64
    // CHECK: %[[BETA_MINUS_PI5:.*]] = arith.subf %[[BETA]], %[[PI5]] : f64
    // CHECK: %[[TRUE15:.*]] = arith.constant true
    // CHECK: %[[RZ_DC1:.*]], %[[CTRL10:.*]] = quantum.custom "RZ"(%[[PI_MINUS_BETA5]]) %[[DCZZ]]#1 ctrls(%[[CTRL9]]) ctrlvals(%[[TRUE15]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE16:.*]] = arith.constant true
    // CHECK: %[[DCXY:.*]]:2, %[[CTRL11:.*]] = quantum.custom "IsingXY"(%[[THETA]]) %[[DCZZ]]#0, %[[RZ_DC1]] ctrls(%[[CTRL10]]) ctrlvals(%[[TRUE16]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE17:.*]] = arith.constant true
    // CHECK: %[[RZ_DC2:.*]], %[[CTRL_FINAL:.*]] = quantum.custom "RZ"(%[[BETA_MINUS_PI5]]) %[[DCXY]]#1 ctrls(%[[CTRL11]]) ctrlvals(%[[TRUE17]]) : !quantum.bit ctrls !quantum.bit
    %dcxxc, %dcxx0, %dcxx1 = qco.ctrl(%crzxc) targets(%arg0 = %dplus0, %arg1 = %dplus1) {
      %out0, %out1 = qco.rxx(%theta) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %dcyyc, %dcyy0, %dcyy1 = qco.ctrl(%dcxxc) targets(%arg0 = %dcxx0, %arg1 = %dcxx1) {
      %out0, %out1 = qco.ryy(%theta) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %dczzc, %dczz0, %dczz1 = qco.ctrl(%dcyyc) targets(%arg0 = %dcyy0, %arg1 = %dcyy1) {
      %out0, %out1 = qco.rzz(%theta) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
    %dcplusc, %dcplus0, %dcplus1 = qco.ctrl(%dczzc) targets(%arg0 = %dczz0, %arg1 = %dczz1) {
      %out0, %out1 = qco.xx_plus_yy(%theta, %beta) %arg0, %arg1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      qco.yield %out0, %out1
    } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})

    // --- Reinsertion ---------------------------------------------------------------------------
    qco.dealloc %dcplus0 : !qco.qubit
    qco.dealloc %dcplus1 : !qco.qubit
    qco.dealloc %dcplusc : !qco.qubit
    // CHECK: %[[REG1:.*]] = quantum.insert %[[REG]][{{ *}}0], %[[DCXY]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[REG2:.*]] = quantum.insert %[[REG1]][{{ *}}1], %[[RZ_DC2]] : !quantum.reg, !quantum.bit
    // CHECK: %[[REG3:.*]] = quantum.insert %[[REG2]][{{ *}}2], %[[CTRL_FINAL]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[REG3]] : !quantum.reg
    // Release qubits
    return
  }
}
