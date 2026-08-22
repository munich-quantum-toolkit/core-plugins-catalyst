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
// RUN:   %s | FileCheck %s --check-prefix=QCO \
// RUN:     --implicit-check-not='quantum.custom "Rot"' \
// RUN:     --implicit-check-not='quantum.paulirot' \
// RUN:     --implicit-check-not='quantum.multirz'
// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco,qco-to-catalystquantum)" \
// RUN:   %s | FileCheck %s --check-prefix=ROUNDTRIP \
// RUN:     --implicit-check-not='quantum.custom "SX"' \
// RUN:     --implicit-check-not='quantum.custom "ECR"' \
// RUN:     --implicit-check-not='quantum.custom "Rot"' \
// RUN:     --implicit-check-not='quantum.paulirot' \
// RUN:     --implicit-check-not='quantum.multirz'
// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco,qco-to-qc,qc-to-qco,qco-to-catalystquantum)" \
// RUN:   %s | FileCheck %s --check-prefix=ROUNDTRIP \
// RUN:     --implicit-check-not='quantum.custom "SX"' \
// RUN:     --implicit-check-not='quantum.custom "ECR"' \
// RUN:     --implicit-check-not='quantum.custom "Rot"' \
// RUN:     --implicit-check-not='quantum.paulirot' \
// RUN:     --implicit-check-not='quantum.multirz'

module {
  // QCO-LABEL: func.func @supported_gates(
  // QCO-SAME: %[[QCO_THETA:.*]]: f64)
  // ROUNDTRIP-LABEL: func.func @supported_gates(
  // ROUNDTRIP-SAME: %[[ROUNDTRIP_THETA:.*]]: f64)
  func.func @supported_gates(%theta: f64) {
    %reg = quantum.alloc(3) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit

    // QCO: qco.id
    // QCO: qco.h
    // QCO: qco.x
    // QCO: qco.y
    // QCO: qco.z
    // QCO: qco.s
    // QCO: qco.t
    // QCO: qco.sx
    // ROUNDTRIP: quantum.custom "Identity"
    // ROUNDTRIP: quantum.custom "Hadamard"
    // ROUNDTRIP: quantum.custom "PauliX"
    // ROUNDTRIP: quantum.custom "PauliY"
    // ROUNDTRIP: quantum.custom "PauliZ"
    // ROUNDTRIP: quantum.custom "S"
    // ROUNDTRIP: quantum.custom "T"
    // ROUNDTRIP: quantum.custom "RX"
    // ROUNDTRIP: quantum.gphase
    %id = quantum.custom "Identity"() %q0 : !quantum.bit
    %h = quantum.custom "Hadamard"() %id : !quantum.bit
    %x = quantum.custom "PauliX"() %h : !quantum.bit
    %y = quantum.custom "PauliY"() %x : !quantum.bit
    %z = quantum.custom "PauliZ"() %y : !quantum.bit
    %s = quantum.custom "S"() %z : !quantum.bit
    %t = quantum.custom "T"() %s : !quantum.bit
    %sx = quantum.custom "SX"() %t : !quantum.bit

    // QCO: qco.rx(%[[QCO_THETA]])
    // QCO: qco.ry(%[[QCO_THETA]])
    // QCO: qco.rz(%[[QCO_THETA]])
    // QCO: qco.p(%[[QCO_THETA]])
    // ROUNDTRIP: quantum.custom "RX"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "RY"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "RZ"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "PhaseShift"(%[[ROUNDTRIP_THETA]])
    %rx = quantum.custom "RX"(%theta) %sx : !quantum.bit
    %ry = quantum.custom "RY"(%theta) %rx : !quantum.bit
    %rz = quantum.custom "RZ"(%theta) %ry : !quantum.bit
    %phase = quantum.custom "PhaseShift"(%theta) %rz : !quantum.bit

    // QCO: qco.swap
    // QCO: qco.iswap
    // QCO: qco.ecr
    // QCO: qco.rxx(%[[QCO_THETA]])
    // QCO: qco.ryy(%[[QCO_THETA]])
    // QCO: qco.rzz(%[[QCO_THETA]])
    // QCO: qco.xx_plus_yy(%[[QCO_THETA]],
    // ROUNDTRIP: quantum.custom "SWAP"
    // ROUNDTRIP: quantum.custom "ISWAP"
    // ROUNDTRIP: quantum.custom "PauliZ"
    // ROUNDTRIP: quantum.custom "CNOT"
    // ROUNDTRIP: quantum.custom "RX"
    // ROUNDTRIP: quantum.gphase
    // ROUNDTRIP: quantum.custom "RX"
    // ROUNDTRIP: quantum.custom "RY"
    // ROUNDTRIP: quantum.custom "RX"
    // ROUNDTRIP: quantum.custom "IsingXX"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "IsingYY"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "IsingZZ"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "IsingXY"(%[[ROUNDTRIP_THETA]])
    %swap:2 = quantum.custom "SWAP"() %phase, %q1 : !quantum.bit, !quantum.bit
    %iswap:2 = quantum.custom "ISWAP"() %swap#0, %swap#1 : !quantum.bit, !quantum.bit
    %ecr:2 = quantum.custom "ECR"() %iswap#0, %iswap#1 : !quantum.bit, !quantum.bit
    %xx:2 = quantum.custom "IsingXX"(%theta) %ecr#0, %ecr#1 : !quantum.bit, !quantum.bit
    %yy:2 = quantum.custom "IsingYY"(%theta) %xx#0, %xx#1 : !quantum.bit, !quantum.bit
    %zz:2 = quantum.custom "IsingZZ"(%theta) %yy#0, %yy#1 : !quantum.bit, !quantum.bit
    %xy:2 = quantum.custom "IsingXY"(%theta) %zz#0, %zz#1 : !quantum.bit, !quantum.bit

    // QCO: catalyst.gate_name = "CNOT"
    // QCO: catalyst.gate_name = "CY"
    // QCO: catalyst.gate_name = "CZ"
    // QCO: catalyst.gate_name = "CRX"
    // QCO: catalyst.gate_name = "CRY"
    // QCO: catalyst.gate_name = "CRZ"
    // QCO: catalyst.gate_name = "ControlledPhaseShift"
    // QCO: catalyst.gate_name = "Toffoli"
    // QCO: catalyst.gate_name = "CSWAP"
    // ROUNDTRIP: quantum.custom "CNOT"
    // ROUNDTRIP: quantum.custom "CY"
    // ROUNDTRIP: quantum.custom "CZ"
    // ROUNDTRIP: quantum.custom "CRX"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "CRY"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "CRZ"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "ControlledPhaseShift"(%[[ROUNDTRIP_THETA]])
    // ROUNDTRIP: quantum.custom "Toffoli"
    // ROUNDTRIP: quantum.custom "CSWAP"
    %cnot:2 = quantum.custom "CNOT"() %q2, %xy#0 : !quantum.bit, !quantum.bit
    %cy:2 = quantum.custom "CY"() %cnot#0, %cnot#1 : !quantum.bit, !quantum.bit
    %cz:2 = quantum.custom "CZ"() %cy#0, %cy#1 : !quantum.bit, !quantum.bit
    %crx:2 = quantum.custom "CRX"(%theta) %cz#0, %cz#1 : !quantum.bit, !quantum.bit
    %cry:2 = quantum.custom "CRY"(%theta) %crx#0, %crx#1 : !quantum.bit, !quantum.bit
    %crz:2 = quantum.custom "CRZ"(%theta) %cry#0, %cry#1 : !quantum.bit, !quantum.bit
    %cphase:2 = quantum.custom "ControlledPhaseShift"(%theta) %crz#0, %crz#1 : !quantum.bit, !quantum.bit
    %toffoli:3 = quantum.custom "Toffoli"() %cphase#0, %xy#1, %cphase#1 : !quantum.bit, !quantum.bit, !quantum.bit
    %cswap:3 = quantum.custom "CSWAP"() %toffoli#0, %toffoli#1, %toffoli#2 : !quantum.bit, !quantum.bit, !quantum.bit

    %reg0 = quantum.insert %reg[0], %cswap#2 : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %cswap#1 : !quantum.reg, !quantum.bit
    %reg2 = quantum.insert %reg1[2], %cswap#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %reg2 : !quantum.reg
    return
  }

  // QCO-LABEL: func.func @rot(
  // QCO-SAME: %[[QCO_PHI:.*]]: f64, %[[QCO_THETA:.*]]: f64, %[[QCO_OMEGA:.*]]: f64)
  // QCO: qco.rz(%[[QCO_PHI]])
  // QCO: qco.ry(%[[QCO_THETA]])
  // QCO: qco.rz(%[[QCO_OMEGA]])
  // ROUNDTRIP-LABEL: func.func @rot(
  // ROUNDTRIP-SAME: %[[RT_PHI:.*]]: f64, %[[RT_THETA:.*]]: f64, %[[RT_OMEGA:.*]]: f64)
  // ROUNDTRIP: quantum.custom "RZ"(%[[RT_PHI]])
  // ROUNDTRIP: quantum.custom "RY"(%[[RT_THETA]])
  // ROUNDTRIP: quantum.custom "RZ"(%[[RT_OMEGA]])
  func.func @rot(%phi: f64, %theta: f64, %omega: f64) {
    %q = quantum.alloc_qb : !quantum.bit
    %out = quantum.custom "Rot"(%phi, %theta, %omega) %q : !quantum.bit
    quantum.dealloc_qb %out : !quantum.bit
    return
  }

  // QCO-LABEL: func.func @pauli_rot_and_multirz(
  // QCO-SAME: %[[QCO_ANGLE:.*]]: f64)
  // QCO: %[[PI_HALF:.*]] = arith.constant 1.5707963267948966
  // QCO: qco.h
  // QCO: qco.rx(%[[PI_HALF]])
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: qco.rz(%[[QCO_ANGLE]])
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: qco.inv
  // QCO: qco.rx(%[[PI_HALF]])
  // QCO: qco.h
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: qco.rz(%[[QCO_ANGLE]])
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: catalyst.gate_name = "CNOT"
  // QCO: catalyst.gate_name = "CNOT"
  // ROUNDTRIP-LABEL: func.func @pauli_rot_and_multirz(
  // ROUNDTRIP-SAME: %[[RT_ANGLE:.*]]: f64)
  // ROUNDTRIP: quantum.custom "Hadamard"
  // ROUNDTRIP: quantum.custom "RX"
  // ROUNDTRIP: quantum.custom "CNOT"
  // ROUNDTRIP: quantum.custom "CNOT"
  // ROUNDTRIP: quantum.custom "RZ"(%[[RT_ANGLE]])
  // ROUNDTRIP: quantum.custom "CNOT"
  // ROUNDTRIP: quantum.custom "CNOT"
  // ROUNDTRIP: quantum.custom "RX"({{.*}}) {{.*}} adj
  // ROUNDTRIP: quantum.custom "Hadamard"
  // ROUNDTRIP: quantum.custom "RZ"(%[[RT_ANGLE]])
  func.func @pauli_rot_and_multirz(%angle: f64) {
    %reg = quantum.alloc(4) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit
    %q3 = quantum.extract %reg[3] : !quantum.reg -> !quantum.bit
    %rot:4 = quantum.paulirot ["I", "X", "Y", "Z"](%angle) %q0, %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %multi:4 = quantum.multirz(%angle) %rot#0, %rot#1, %rot#2, %rot#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %reg0 = quantum.insert %reg[0], %multi#0 : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %multi#1 : !quantum.reg, !quantum.bit
    %reg2 = quantum.insert %reg1[2], %multi#2 : !quantum.reg, !quantum.bit
    %reg3 = quantum.insert %reg2[3], %multi#3 : !quantum.reg, !quantum.bit
    quantum.dealloc %reg3 : !quantum.reg
    return
  }

  // QCO-LABEL: func.func @modified_rotations(
  // QCO-SAME: %[[MOD_PHI:.*]]: f64, %[[MOD_THETA:.*]]: f64, %[[MOD_OMEGA:.*]]: f64, %[[MOD_ANGLE:.*]]: f64)
  // Rot adjoint reverses the Euler order and inverts every primitive.
  // QCO: qco.rz(%[[MOD_OMEGA]])
  // QCO: qco.ry(%[[MOD_THETA]])
  // QCO: qco.rz(%[[MOD_PHI]])
  // QCO: catalyst.control_values = array<i1: false>
  // PauliRot and MultiRZ share the modifier-aware parity lowering.
  // QCO: qco.ctrl
  // QCO: qco.h
  // QCO: catalyst.control_values = array<i1: false>
  // QCO: qco.ctrl
  // QCO: qco.rx
  // QCO: catalyst.control_values = array<i1: false>
  // QCO: qco.inv
  // QCO: qco.rz(%[[MOD_ANGLE]])
  // QCO: catalyst.control_values = array<i1: false>
  // QCO: qco.inv
  // QCO: qco.rz(%[[MOD_ANGLE]])
  // QCO: catalyst.control_values = array<i1: false>
  // ROUNDTRIP-LABEL: func.func @modified_rotations(
  // ROUNDTRIP-SAME: {{.*}}, %[[RT_MOD_ANGLE:[^, )]+]]: f64)
  // ROUNDTRIP: quantum.custom "RZ"({{.*}}) {{.*}} adj {{.*}}ctrlvals({{.*}})
  // ROUNDTRIP: quantum.custom "RY"({{.*}}) {{.*}} adj
  // ROUNDTRIP: quantum.custom "RZ"({{.*}}) {{.*}} adj
  // ROUNDTRIP: quantum.custom "Hadamard"() {{.*}}ctrls({{.*}}) ctrlvals({{.*}})
  // ROUNDTRIP: quantum.custom "RX"({{.*}}) {{.*}}ctrls({{.*}}) ctrlvals({{.*}})
  // ROUNDTRIP: quantum.custom "RZ"(%[[RT_MOD_ANGLE]]) {{.*}} adj {{.*}}ctrls({{.*}}) ctrlvals({{.*}})
  // ROUNDTRIP: quantum.custom "RZ"(%[[RT_MOD_ANGLE]]) {{.*}} adj {{.*}}ctrls({{.*}}) ctrlvals({{.*}})
  func.func @modified_rotations(%phi: f64, %theta: f64, %omega: f64, %angle: f64) {
    %false = arith.constant false
    %control = quantum.alloc_qb : !quantum.bit
    %q0 = quantum.alloc_qb : !quantum.bit
    %q1 = quantum.alloc_qb : !quantum.bit
    %q2 = quantum.alloc_qb : !quantum.bit
    %rot, %control0 = quantum.custom "Rot"(%phi, %theta, %omega) %q0 adj ctrls(%control) ctrlvals(%false) : !quantum.bit ctrls !quantum.bit
    %pauli:3, %control1 = quantum.paulirot ["X", "Y", "Z"](%angle) %rot, %q1, %q2 adj ctrls(%control0) ctrlvals(%false) : !quantum.bit, !quantum.bit, !quantum.bit ctrls !quantum.bit
    %multi:3, %control2 = quantum.multirz(%angle) %pauli#0, %pauli#1, %pauli#2 adj ctrls(%control1) ctrlvals(%false) : !quantum.bit, !quantum.bit, !quantum.bit ctrls !quantum.bit
    quantum.dealloc_qb %multi#0 : !quantum.bit
    quantum.dealloc_qb %multi#1 : !quantum.bit
    quantum.dealloc_qb %multi#2 : !quantum.bit
    quantum.dealloc_qb %control2 : !quantum.bit
    return
  }

  // An all-identity PauliRot is a relative phase when controlled.
  // QCO-LABEL: func.func @controlled_identity_pauli(
  // QCO-SAME: %[[IDENTITY_ANGLE:.*]]: f64)
  // QCO: %[[MINUS_HALF:.*]] = arith.constant {{-5.000000e-01|-0.5}}
  // QCO: %[[PHASE_ANGLE:.*]] = arith.mulf %[[IDENTITY_ANGLE]], %[[MINUS_HALF]]
  // QCO: qco.gphase(%[[PHASE_ANGLE]])
  // ROUNDTRIP-LABEL: func.func @controlled_identity_pauli(
  // ROUNDTRIP: quantum.gphase({{.*}}) adj {{.*}}ctrls(
  func.func @controlled_identity_pauli(%angle: f64) {
    %true = arith.constant true
    %control = quantum.alloc_qb : !quantum.bit
    %target = quantum.alloc_qb : !quantum.bit
    %out, %control_out = quantum.paulirot ["I"](%angle) %target ctrls(%control) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    quantum.dealloc_qb %out : !quantum.bit
    quantum.dealloc_qb %control_out : !quantum.bit
    return
  }
}
