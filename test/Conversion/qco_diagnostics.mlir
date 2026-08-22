// Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: split-file %s %t
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/incomplete_register_metadata.mlir 2>&1 | FileCheck %s --check-prefix=INCOMPLETE
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/duplicate_register_index.mlir 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/partial_register_metadata.mlir 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/register_size_exceeds_allocation_count.mlir 2>&1 | FileCheck %s --check-prefix=REGISTER-SIZE
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/static_hardware_qubit.mlir 2>&1 | FileCheck %s --check-prefix=STATIC
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/control_flow_boundary.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/unmatched_negative_control_wrapper.mlir 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/malformed_qubit_bridge.mlir 2>&1 | FileCheck %s --check-prefix=BRIDGE-METADATA
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/negative_control_without_sandwich.mlir 2>&1 | FileCheck %s --check-prefix=NEGATIVE-METADATA
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/inconsistent_gate_hint.mlir 2>&1 | FileCheck %s --check-prefix=GATE-HINT
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/unmatched_gate_hint_bridge.mlir 2>&1 | FileCheck %s --check-prefix=GATE-BRIDGE
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(qco-to-catalystquantum)" \
// RUN:   %t/external_qco_boundary.mlir 2>&1 | FileCheck %s --check-prefix=EXTERNAL-BOUNDARY

//--- incomplete_register_metadata.mlir
module {
  func.func @incomplete_register_metadata() {
    // INCOMPLETE: qco.alloc register is incomplete: input
    %scalar = qco.alloc : !qco.qubit
    %q0 = qco.alloc("input", 3, 0) : !qco.qubit
    %q2 = qco.alloc("input", 3, 2) : !qco.qubit
    qco.dealloc %scalar : !qco.qubit
    qco.dealloc %q0 : !qco.qubit
    qco.dealloc %q2 : !qco.qubit
    return
  }
}

//--- duplicate_register_index.mlir

module {
  func.func @duplicate_register_index() {
    // DUPLICATE: duplicate qco.alloc register_index
    %q0 = qco.alloc("input", 2, 0) : !qco.qubit
    %duplicate = qco.alloc("input", 2, 0) : !qco.qubit
    qco.dealloc %q0 : !qco.qubit
    qco.dealloc %duplicate : !qco.qubit
    return
  }
}

//--- partial_register_metadata.mlir

module {
  func.func @partial_register_metadata() {
    // PARTIAL: 'qco.alloc' op register attributes must all be present or all absent
    %q = "qco.alloc"() <{register_name = "input"}> : () -> !qco.qubit
    qco.dealloc %q : !qco.qubit
    return
  }
}

//--- register_size_exceeds_allocation_count.mlir

module {
  func.func @register_size_exceeds_allocation_count() {
    // REGISTER-SIZE: error: malformed qco.alloc register metadata
    %q = qco.alloc("input", 2, 0) : !qco.qubit
    qco.dealloc %q : !qco.qubit
    return
  }
}

//--- static_hardware_qubit.mlir

module {
  func.func @static_hardware_qubit() {
    // STATIC: qco.static hardware qubits are not supported
    %q = qco.static 0 : !qco.qubit
    qco.dealloc %q : !qco.qubit
    return
  }
}

//--- control_flow_boundary.mlir

module {
  func.func @control_flow_boundary(%condition: i1) {
    // BOUNDARY: QCO qubits across function or control-flow boundaries are not supported
    %q = qco.alloc : !qco.qubit
    %out = scf.if %condition -> (!qco.qubit) {
      scf.yield %q : !qco.qubit
    } else {
      scf.yield %q : !qco.qubit
    }
    qco.dealloc %out : !qco.qubit
    return
  }
}

//--- unmatched_negative_control_wrapper.mlir

module {
  func.func @unmatched_negative_control_wrapper() {
    // NEGATIVE: malformed negative-control wrapper
    %q = qco.alloc : !qco.qubit
    %out = qco.x %q {catalyst.negative_control_wrapper} : !qco.qubit -> !qco.qubit
    qco.dealloc %out : !qco.qubit
    return
  }
}

//--- malformed_qubit_bridge.mlir

module {
  // BRIDGE-METADATA: malformed catalyst.qco_qubit_bridge metadata
  func.func private @malformed_qubit_bridge(!qco.qubit) -> !quantum.bit attributes {catalyst.qco_qubit_bridge = 0 : i64}
}

//--- negative_control_without_sandwich.mlir

module {
  func.func @negative_control_without_sandwich() {
    // NEGATIVE-METADATA: negative catalyst.control_values metadata requires an X-control-X sandwich
    %control = qco.alloc : !qco.qubit
    %target = qco.alloc : !qco.qubit
    %control_out, %target_out = qco.ctrl(%control) targets (%arg = %target) {
      %out = qco.x %arg : !qco.qubit -> !qco.qubit
      qco.yield %out
    } {catalyst.control_values = array<i1: false>} : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
    qco.dealloc %control_out : !qco.qubit
    qco.dealloc %target_out : !qco.qubit
    return
  }
}

//--- inconsistent_gate_hint.mlir

module {
  func.func @inconsistent_gate_hint() {
    // GATE-HINT: catalyst gate metadata is inconsistent with qco.x
    %q = qco.alloc : !qco.qubit
    %out = qco.x %q {catalyst.gate_name = "Hadamard"} : !qco.qubit -> !qco.qubit
    qco.dealloc %out : !qco.qubit
    return
  }
}

//--- unmatched_gate_hint_bridge.mlir

module {
  func.func private @gate_hint() attributes {catalyst.gate_name = "PauliX", catalyst.native_control_count = 0 : i64, catalyst.qco_gate_hint_bridge}

  func.func @unmatched_gate_hint_bridge() {
    // GATE-BRIDGE: catalyst.qco_gate_hint_bridge call must immediately precede qco.ctrl
    call @gate_hint() : () -> ()
    return
  }
}

//--- external_qco_boundary.mlir

module {
  // EXTERNAL-BOUNDARY: QCO qubits across function or control-flow boundaries are not supported
  func.func private @external_qco_boundary(!qco.qubit)
}
