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
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco,cse,qco-to-catalystquantum)" \
// RUN:   %s | FileCheck %s --check-prefix=DIRECT \
// RUN:     --implicit-check-not="qco." --implicit-check-not="qc."
// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco,cse,qco-to-qc,cse)" \
// RUN:   %s | FileCheck %s --check-prefix=QC
// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco,cse,qco-to-qc,cse,qc-to-qco,cse,qco-to-catalystquantum)" \
// RUN:   %s | FileCheck %s --check-prefix=CHAINED \
// RUN:     --implicit-check-not="qco." --implicit-check-not="qc."

module {
  // DIRECT-LABEL: func.func @round_trip(
  // DIRECT-SAME: %[[DIRECT_THETA:.*]]: f64)
  // QC-LABEL: func.func @round_trip(
  // QC-SAME: %[[QC_THETA:.*]]: f64)
  // CHAINED-LABEL: func.func @round_trip(
  // CHAINED-SAME: %[[CHAINED_THETA:.*]]: f64)
  func.func @round_trip(%theta: f64) -> i1 {
    // QC: %[[Q0:.*]] = qc.alloc("qreg0", 2, 0) : !qc.qubit
    // QC: %[[Q1:.*]] = qc.alloc("qreg0", 2, 1) : !qc.qubit
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit

    // DIRECT: quantum.custom "Hadamard"
    // QC: qc.h %[[Q0]]
    // CHAINED: quantum.custom "Hadamard"
    %h = quantum.custom "Hadamard"() %q0 : !quantum.bit

    // DIRECT: quantum.custom "RX"(%[[DIRECT_THETA]])
    // QC: qc.rx(%[[QC_THETA]]) %[[Q0]]
    // CHAINED: quantum.custom "RX"(%[[CHAINED_THETA]])
    %rx = quantum.custom "RX"(%theta) %h : !quantum.bit

    // DIRECT: quantum.custom "PauliX"() {{.*}} ctrls(
    // QC: qc.ctrl(%[[Q1]]) {
    // QC: qc.x
    // CHAINED: quantum.custom "PauliX"() {{.*}} ctrls(
    %true = arith.constant true
    %target, %control = quantum.custom "PauliX"() %rx ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // The QCO/QC bridge drops provenance attributes, but the exact X-control-X
    // sandwich still reconstructs the original negative Catalyst control.
    // DIRECT-NOT: quantum.custom "PauliX"()
    // DIRECT: %[[DIRECT_FALSE:.*]] = arith.constant false
    // DIRECT-NEXT: quantum.custom "PauliX"() {{.*}} ctrls({{.*}}) ctrlvals(%[[DIRECT_FALSE]])
    // DIRECT-NOT: quantum.custom "PauliX"()
    // QC: qc.x %[[Q1]]
    // QC: qc.ctrl(%[[Q1]]) {
    // QC: qc.x %[[Q0]]
    // QC: }
    // QC: qc.x %[[Q1]]
    // CHAINED-NOT: quantum.custom "PauliX"()
    // CHAINED: %[[CHAINED_FALSE:.*]] = arith.constant false
    // CHAINED-NEXT: quantum.custom "PauliX"() {{.*}} ctrls({{.*}}) ctrlvals(%[[CHAINED_FALSE]])
    // CHAINED-NOT: quantum.custom "PauliX"()
    %false = arith.constant false
    %negative_target, %negative_control = quantum.custom "PauliX"() %target ctrls(%control) ctrlvals(%false) : !quantum.bit ctrls !quantum.bit

    // The shared X remains owned by the preceding negative control.
    // DIRECT: %[[DIRECT_TRUE:.*]] = arith.constant true
    // DIRECT-NEXT: %[[DIRECT_POSITIVE_TARGET:.*]], %[[DIRECT_POSITIVE_CONTROL:.*]] = quantum.custom "PauliZ"() {{.*}} ctrls({{.*}}) ctrlvals(%[[DIRECT_TRUE]])
    // DIRECT-NEXT: %[[DIRECT_FINAL_CONTROL:.*]] = quantum.custom "PauliX"() %[[DIRECT_POSITIVE_CONTROL]]
    // CHAINED: %[[CHAINED_TRUE:.*]] = arith.constant true
    // CHAINED-NEXT: %[[CHAINED_POSITIVE_TARGET:.*]], %[[CHAINED_POSITIVE_CONTROL:.*]] = quantum.custom "PauliZ"() {{.*}} ctrls({{.*}}) ctrlvals(%[[CHAINED_TRUE]])
    // CHAINED-NEXT: %[[CHAINED_FINAL_CONTROL:.*]] = quantum.custom "PauliX"() %[[CHAINED_POSITIVE_CONTROL]]
    %positive_target, %positive_control = quantum.custom "PauliZ"() %negative_target ctrls(%negative_control) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %final_control = quantum.custom "PauliX"() %positive_control : !quantum.bit

    // DIRECT: %[[RESULT:.*]], %[[MEASURED:.*]] = quantum.measure %[[DIRECT_POSITIVE_TARGET]]
    // DIRECT-SAME: mqt.qco_measure_register_name = "c"
    // QC: %[[QC_RESULT:.*]] = qc.measure("c", 1, 0)
    // CHAINED: %[[CHAIN_RESULT:.*]], %[[CHAIN_MEASURED:.*]] = quantum.measure %[[CHAINED_POSITIVE_TARGET]]
    // CHAINED-SAME: mqt.qco_measure_register_name = "c"
    %result, %measured = quantum.measure %positive_target {mqt.qco_measure_register_index = 0 : i64, mqt.qco_measure_register_name = "c", mqt.qco_measure_register_size = 1 : i64} : i1, !quantum.bit
    %reg0 = quantum.insert %reg[0], %measured : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %final_control : !quantum.reg, !quantum.bit
    quantum.dealloc %reg1 : !quantum.reg
    return %result : i1
  }

  // A preserved observable reads the latest qubit value even after Core's
  // value-to-reference-to-value conversion collapses the intermediate SSA
  // chain to one QC reference.
  // DIRECT-LABEL: func.func @round_trip_observable
  // DIRECT: %[[DIRECT_OBS_REG:.*]] = quantum.alloc
  // DIRECT: %[[DIRECT_OBS_Q:.*]] = quantum.extract %[[DIRECT_OBS_REG]][{{ *}}0]
  // DIRECT: %[[DIRECT_OBS_H:.*]] = quantum.custom "Hadamard"() %[[DIRECT_OBS_Q]]
  // DIRECT: %[[DIRECT_OBS:.*]] = quantum.namedobs %[[DIRECT_OBS_H]][{{ *}}PauliZ]
  // DIRECT: %[[DIRECT_EXPVAL:.*]] = quantum.expval %[[DIRECT_OBS]]
  // DIRECT: return %[[DIRECT_EXPVAL]] : f64
  // QC-LABEL: func.func @round_trip_observable
  // QC: qc.h
  // The bridge declaration's argument follows Core's QCO-to-QC type
  // conversion, while the call remains an observable read barrier.
  // QC: %[[QC_OBS_BRIDGE:.*]] = call @__mqt_catalyst_qco_qubit_bridge(%{{.*}}) : (!qc.qubit) -> !quantum.bit
  // QC: quantum.namedobs %[[QC_OBS_BRIDGE]][{{ *}}PauliZ]
  // CHAINED-LABEL: func.func @round_trip_observable
  // CHAINED: %[[CHAINED_OBS_REG:.*]] = quantum.alloc
  // CHAINED: %[[CHAINED_OBS_Q:.*]] = quantum.extract %[[CHAINED_OBS_REG]][{{ *}}0]
  // CHAINED: %[[CHAINED_OBS_H:.*]] = quantum.custom "Hadamard"() %[[CHAINED_OBS_Q]]
  // CHAINED: %[[CHAINED_OBS:.*]] = quantum.namedobs %[[CHAINED_OBS_H]][{{ *}}PauliZ]
  // CHAINED: %[[CHAINED_EXPVAL:.*]] = quantum.expval %[[CHAINED_OBS]]
  // CHAINED: return %[[CHAINED_EXPVAL]] : f64
  func.func @round_trip_observable() -> f64 {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %h = quantum.custom "Hadamard"() %q : !quantum.bit
    %obs = quantum.namedobs %h[PauliZ] : !quantum.obs
    %expval = quantum.expval %obs : f64
    %updated = quantum.insert %reg[0], %h : !quantum.reg, !quantum.bit
    quantum.dealloc %updated : !quantum.reg
    return %expval : f64
  }

  // Scalar alloc_qb origins use the same direct and chained bridge path.
  // DIRECT-LABEL: func.func @round_trip_scalar_observable
  // DIRECT: %[[DIRECT_SCALAR_Q:.*]] = quantum.alloc_qb : !quantum.bit
  // DIRECT: %[[DIRECT_SCALAR_H:.*]] = quantum.custom "Hadamard"() %[[DIRECT_SCALAR_Q]]
  // DIRECT: quantum.namedobs %[[DIRECT_SCALAR_H]][{{ *}}PauliZ]
  // DIRECT: quantum.dealloc_qb %[[DIRECT_SCALAR_H]]
  // QC-LABEL: func.func @round_trip_scalar_observable
  // QC: %[[QC_SCALAR_Q:.*]] = qc.alloc : !qc.qubit
  // QC: qc.h %[[QC_SCALAR_Q]]
  // QC: call @__mqt_catalyst_qco_qubit_bridge(%[[QC_SCALAR_Q]]) : (!qc.qubit) -> !quantum.bit
  // CHAINED-LABEL: func.func @round_trip_scalar_observable
  // CHAINED: %[[CHAINED_SCALAR_Q:.*]] = quantum.alloc_qb : !quantum.bit
  // CHAINED: %[[CHAINED_SCALAR_H:.*]] = quantum.custom "Hadamard"() %[[CHAINED_SCALAR_Q]]
  // CHAINED: quantum.namedobs %[[CHAINED_SCALAR_H]][{{ *}}PauliZ]
  // CHAINED: quantum.dealloc_qb %[[CHAINED_SCALAR_H]]
  func.func @round_trip_scalar_observable() -> f64 {
    %q = quantum.alloc_qb : !quantum.bit
    %h = quantum.custom "Hadamard"() %q : !quantum.bit
    %obs = quantum.namedobs %h[PauliZ] : !quantum.obs
    %expval = quantum.expval %obs : f64
    quantum.dealloc_qb %h : !quantum.bit
    return %expval : f64
  }

  // Calls keep the pre-gate and post-gate observable reads distinct even when
  // CSE runs before, during, and after the Core QCO/QC bridge.
  // DIRECT-LABEL: func.func @cse_safe_observables
  // DIRECT: %[[DIRECT_CSE_Q:.*]] = quantum.alloc_qb
  // DIRECT: quantum.namedobs %[[DIRECT_CSE_Q]][{{ *}}PauliZ]
  // DIRECT: %[[DIRECT_CSE_H:.*]] = quantum.custom "Hadamard"() %[[DIRECT_CSE_Q]]
  // DIRECT: quantum.namedobs %[[DIRECT_CSE_H]][{{ *}}PauliZ]
  // QC-LABEL: func.func @cse_safe_observables
  // QC: %[[QC_CSE_Q:.*]] = qc.alloc
  // QC: %[[QC_CSE_BEFORE:.*]] = call @__mqt_catalyst_qco_qubit_bridge(%[[QC_CSE_Q]]) : (!qc.qubit) -> !quantum.bit
  // QC: quantum.namedobs %[[QC_CSE_BEFORE]][{{ *}}PauliZ]
  // QC: qc.h %[[QC_CSE_Q]]
  // QC: %[[QC_CSE_AFTER:.*]] = call @__mqt_catalyst_qco_qubit_bridge(%[[QC_CSE_Q]]) : (!qc.qubit) -> !quantum.bit
  // QC: quantum.namedobs %[[QC_CSE_AFTER]][{{ *}}PauliZ]
  // CHAINED-LABEL: func.func @cse_safe_observables
  // CHAINED: %[[CHAIN_CSE_Q:.*]] = quantum.alloc_qb
  // CHAINED: quantum.namedobs %[[CHAIN_CSE_Q]][{{ *}}PauliZ]
  // CHAINED: %[[CHAIN_CSE_H:.*]] = quantum.custom "Hadamard"() %[[CHAIN_CSE_Q]]
  // CHAINED: quantum.namedobs %[[CHAIN_CSE_H]][{{ *}}PauliZ]
  func.func @cse_safe_observables() -> f64 {
    %q = quantum.alloc_qb : !quantum.bit
    %before = quantum.namedobs %q[PauliZ] : !quantum.obs
    %h = quantum.custom "Hadamard"() %q : !quantum.bit
    %after = quantum.namedobs %h[PauliZ] : !quantum.obs
    %before_value = quantum.expval %before : f64
    %after_value = quantum.expval %after : f64
    %sum = arith.addf %before_value, %after_value : f64
    quantum.dealloc_qb %h : !quantum.bit
    return %sum : f64
  }
}
