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
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/dynamic_register_size.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC-SIZE
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/dynamic_register_index.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC-INDEX
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/dynamic_control_value.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC-CONTROL
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/unresolved_operator.mlir 2>&1 | FileCheck %s --check-prefix=OPERATOR
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/control_flow_boundary.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/empty_register_name.mlir 2>&1 | FileCheck %s --check-prefix=REGISTER-NAME
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/partial_measurement_metadata.mlir 2>&1 | FileCheck %s --check-prefix=MEASURE-METADATA
// RUN: not catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --pass-pipeline="builtin.module(catalystquantum-to-qco)" \
// RUN:   %t/multi_block_function.mlir 2>&1 | FileCheck %s --check-prefix=MULTI-BLOCK

//--- dynamic_register_size.mlir
module {
  func.func @dynamic_register_size(%size: i64) {
    // DYNAMIC-SIZE: dynamic register sizes are not supported
    %reg = quantum.alloc(%size) : !quantum.reg
    quantum.dealloc %reg : !quantum.reg
    return
  }
}

//--- empty_register_name.mlir

module {
  func.func @empty_register_name() {
    // REGISTER-NAME: mqt.qco_register_name must be a nonempty string
    %reg = quantum.alloc(1) {mqt.qco_register_name = ""} : !quantum.reg
    quantum.dealloc %reg : !quantum.reg
    return
  }
}

//--- partial_measurement_metadata.mlir

module {
  func.func @partial_measurement_metadata() -> i1 {
    // MEASURE-METADATA: QCO measurement register metadata must be all present or all absent
    %q = quantum.alloc_qb : !quantum.bit
    %result, %out = quantum.measure %q {mqt.qco_measure_register_name = "c"} : i1, !quantum.bit
    quantum.dealloc_qb %out : !quantum.bit
    return %result : i1
  }
}

//--- dynamic_register_index.mlir

module {
  func.func @dynamic_register_index(%index: i64) {
    // DYNAMIC-INDEX: dynamic register indices are not supported
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %updated = quantum.insert %reg[%index], %q : !quantum.reg, !quantum.bit
    quantum.dealloc %updated : !quantum.reg
    return
  }
}

//--- dynamic_control_value.mlir

module {
  func.func @dynamic_control_value(%value: i1) {
    // DYNAMIC-CONTROL: dynamic control values are not supported
    %reg = quantum.alloc(2) : !quantum.reg
    %target = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %control = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %out, %out_control = quantum.custom "PauliX"() %target ctrls(%control) ctrlvals(%value) : !quantum.bit ctrls !quantum.bit
    %reg0 = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
    %reg1 = quantum.insert %reg0[1], %out_control : !quantum.reg, !quantum.bit
    quantum.dealloc %reg1 : !quantum.reg
    return
  }
}

//--- unresolved_operator.mlir

module {
  func.func @unresolved_operator() {
    // OPERATOR: quantum.operator must be decomposed
    %q = quantum.alloc_qb : !quantum.bit
    %out = quantum.operator "Unsupported"() qubits(%q)
    quantum.dealloc_qb %out : !quantum.bit
    return
  }
}

//--- control_flow_boundary.mlir

module {
  func.func @control_flow_boundary(%condition: i1) {
    // BOUNDARY: quantum values crossing control-flow boundaries are not supported
    %reg = quantum.alloc(1) : !quantum.reg
    %out = scf.if %condition -> (!quantum.reg) {
      scf.yield %reg : !quantum.reg
    } else {
      scf.yield %reg : !quantum.reg
    }
    quantum.dealloc %out : !quantum.reg
    return
  }
}

//--- multi_block_function.mlir

module {
  // MULTI-BLOCK: quantum operations in multi-block functions are not supported
  func.func @multi_block_function() {
    %q = quantum.alloc_qb : !quantum.bit
    quantum.dealloc_qb %q : !quantum.bit
    cf.br ^exit
  ^exit:
    return
  }
}
