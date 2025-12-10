# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for MQT plugin execution with PennyLane and Catalyst.

These tests check that the MQT plugin conversion passes execute successfully
for various gate categories, mirroring the MLIR conversion tests. They verify
that the full lossless roundtrip (CatalystQuantum → MQTOpt → CatalystQuantum)
works correctly. The tests use FileCheck (from LLVM) to verify the generated MLIR output.

Environment Variables:
    FILECHECK_PATH: Optional path to FileCheck binary if not in PATH
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pennylane as qml
from catalyst.passes import apply_pass

from mqt.core.plugins.catalyst import get_device

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest


@pytest.fixture(autouse=True)
def _cleanup_mlir_files():
    """Automatically clean up MLIR artifacts before and after each test."""
    # Clean up before test
    _cleanup_mlir_artifacts()
    yield
    # Clean up after test
    _cleanup_mlir_artifacts()


def _cleanup_mlir_artifacts() -> None:
    """Clean up MLIR intermediate files and directories created by Catalyst.
    
    Catalyst creates module_N directories when keep_intermediate is used.
    This function removes all such directories to prevent accumulation.
    """
    mlir_dir = Path.cwd()
    # Remove all module_N directories
    for module_dir in mlir_dir.glob("module_*"):
        if module_dir.is_dir():
            import shutil
            shutil.rmtree(module_dir)
    # Remove any loose .mlir files
    for mlir_file in mlir_dir.glob("*.mlir"):
        mlir_file.unlink()


def _run_filecheck(mlir_content: str, check_patterns: str, test_name: str = "test") -> None:
    """Run FileCheck on MLIR content using CHECK patterns from a string.

    Args:
        mlir_content: The MLIR output to verify
        check_patterns: String containing FileCheck directives (lines starting with // CHECK)
        test_name: Name of the test (for error messages)

    Raises:
        RuntimeError: If FileCheck is not found
        AssertionError: If FileCheck validation fails
    """
    # Find FileCheck (usually in LLVM bin directory)
    filecheck = None
    possible_paths = [
        "FileCheck",  # If in PATH
        os.environ.get("FILECHECK_PATH"),  # Custom env variable
        "/opt/homebrew/opt/llvm/bin/FileCheck",  # Common macOS location
    ]

    for path in possible_paths:
        if path:
            try:
                result = subprocess.run([path, "--version"], check=False, capture_output=True, timeout=5)  # noqa: S603
                if result.returncode == 0:
                    filecheck = path
                    break
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

    if not filecheck:
        msg = (
            "FileCheck not found. Please ensure LLVM's FileCheck is in your PATH, "
            "or set FILECHECK_PATH environment variable."
        )
        raise RuntimeError(msg)

    # Write CHECK patterns to a temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".mlir", delete=False) as check_file:
        check_file.write(check_patterns)
        check_file_path = check_file.name

    try:
        # Run FileCheck: pipe MLIR content as stdin, use check_file for CHECK directives
        result = subprocess.run(  # noqa: S603
            [filecheck, check_file_path, "--allow-unused-prefixes"],
            check=False,
            input=mlir_content.encode(),
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            msg = (
                f"FileCheck failed for {test_name}:\n{error_msg}\n\n"
                f"MLIR Output (first 2000 chars):\n{mlir_content[:2000]}..."
            )
            raise AssertionError(msg)
    finally:
        # Clean up temporary file
        Path(check_file_path).unlink()


def test_clifford_gates_roundtrip() -> None:
    """Test roundtrip conversion of Clifford+T gates.

    Mirrors: quantum_clifford.mlir
    Gates: H, SX, SX†, S, S†, T, T†, and their controlled variants

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        # Non-controlled Clifford+T gates
        qml.Hadamard(wires=0)
        qml.SX(wires=0)
        qml.adjoint(qml.SX(wires=0))
        qml.S(wires=0)
        qml.adjoint(qml.S(wires=0))
        qml.T(wires=0)
        qml.adjoint(qml.T(wires=0))

        # Controlled Clifford+T gates
        qml.CH(wires=[1, 0])
        qml.ctrl(qml.SX(wires=0), control=1)
        # Why is `qml.ctrl(qml.adjoint(qml.SX(wires=0)), control=1)` not supported by Catalyst?
        qml.ctrl(qml.S(wires=0), control=1)
        qml.ctrl(qml.adjoint(qml.S(wires=0)), control=1)
        qml.ctrl(qml.T(wires=0), control=1)
        qml.ctrl(qml.adjoint(qml.T(wires=0)), control=1)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // Allocate and load first qubit
    // CHECK: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x!mqtopt.Qubit>

    // Load target qubit 0
    // CHECK: %[[Q0_I64:.*]] = tensor.extract %[[C0]][] : tensor<i64>
    // CHECK: %[[Q0_IDX:.*]] = arith.index_cast %[[Q0_I64]] : i64 to index
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][%[[Q0_IDX]]] : memref<2x!mqtopt.Qubit>

    // --- Uncontrolled Clifford+T gates ---------------------------------------------------------
    // Hadamard
    // CHECK: %[[OUT_QUBITS:.*]] = mqtopt.h(static [] mask []) %[[Q0]] : !mqtopt.Qubit

    // Constants for SX decomposition
    // CHECK: %[[PI2:.*]] = stablehlo.constant dense<1.5707963267948966> : tensor<f64>
    
    // SX decomposition from Catalyst (RZ(π/2) → RY(π/2) → RZ(-π/2)
    // CHECK: %[[PI2_EX1:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_3:.*]] = mqtopt.rz(%[[PI2_EX1]] static [] mask [false]) %[[OUT_QUBITS]] : !mqtopt.Qubit

    // CHECK: %[[PI2_EX2:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_5:.*]] = mqtopt.ry(%[[PI2_EX2]] static [] mask [false]) %[[OUT_QUBITS_3]] : !mqtopt.Qubit

    // CHECK: %[[NEG_PI2:.*]] = stablehlo.constant dense<-1.5707963267948966> : tensor<f64>
    // CHECK: %[[NEG_PI2_EX:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_8:.*]] = mqtopt.rz(%[[NEG_PI2_EX]] static [] mask [false]) %[[OUT_QUBITS_5]] : !mqtopt.Qubit

    // Capture -π/4 constant and uncontrolled gphase operations (from SX decomposition)
    // CHECK: %[[NEG_PI4:.*]] = stablehlo.constant dense<-0.78539816339744828> : tensor<f64>
    // CHECK: %[[NEG_PI4_EX1:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: mqtopt.gphase(%[[NEG_PI4_EX1]] static [] mask [false])
    // CHECK: %[[NEG_PI4_EX2:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: mqtopt.gphase(%[[NEG_PI4_EX2]] static [] mask [false])

    // SX† decomposition (RZ(-π/2) → RY(π/2) → RZ(π/2))
    // CHECK: %[[NEG_PI2_EX2:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_13:.*]] = mqtopt.rz(%[[NEG_PI2_EX2]] static [] mask [false]) %[[OUT_QUBITS_8]] : !mqtopt.Qubit
    // CHECK: %[[PI2_EX3:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_15:.*]] = mqtopt.ry(%[[PI2_EX3]] static [] mask [false]) %[[OUT_QUBITS_13]] : !mqtopt.Qubit
    // CHECK: %[[PI2_EX4:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_17:.*]] = mqtopt.rz(%[[PI2_EX4]] static [] mask [false]) %[[OUT_QUBITS_15]] : !mqtopt.Qubit

    // S, S†, T, T†
    // CHECK: %[[OUT_QUBITS_18:.*]] = mqtopt.s(static [] mask []) %[[OUT_QUBITS_17]] : !mqtopt.Qubit
    // CHECK: %[[OUT_QUBITS_19:.*]] = mqtopt.s(static [] mask []) %[[OUT_QUBITS_18]] : !mqtopt.Qubit
    // CHECK: %[[OUT_QUBITS_20:.*]] = mqtopt.t(static [] mask []) %[[OUT_QUBITS_19]] : !mqtopt.Qubit
    // CHECK: %[[OUT_QUBITS_21:.*]] = mqtopt.t(static [] mask []) %[[OUT_QUBITS_20]] : !mqtopt.Qubit

    // --- Controlled Section --------------------------------------------------------------------
    // Load control qubit
    // CHECK: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i64>
    // CHECK: %[[C1_EX:.*]] = tensor.extract %[[C1]][] : tensor<i64>
    // CHECK: %[[C1_IDX:.*]] = arith.index_cast %[[C1_EX]] : i64 to index
    // CHECK: %[[CTRL:.*]] = memref.load %[[ALLOC]][%[[C1_IDX]]] : memref<2x!mqtopt.Qubit>

    // Controlled H
    // CHECK: %[[OUT_QUBITS_26:.*]], %[[POS_CTRL_OUT_QUBITS:.*]] = mqtopt.h(static [] mask []) %[[OUT_QUBITS_21]] ctrl %[[CTRL]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled SX decomposition
    // CHECK: %[[PI2_EX5:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_28:.*]], %[[POS_CTRL_OUT_QUBITS_29:.*]] = mqtopt.rz(%[[PI2_EX5]] static [] mask [false]) %[[OUT_QUBITS_26]] ctrl %[[POS_CTRL_OUT_QUBITS]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: %[[PI2_EX6:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_31:.*]], %[[POS_CTRL_OUT_QUBITS_32:.*]] = mqtopt.ry(%[[PI2_EX6]] static [] mask [false]) %[[OUT_QUBITS_28]] ctrl %[[POS_CTRL_OUT_QUBITS_29]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: %[[NEG_PI2_EX3:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_34:.*]], %[[POS_CTRL_OUT_QUBITS_35:.*]] = mqtopt.rz(%[[NEG_PI2_EX3]] static [] mask [false]) %[[OUT_QUBITS_31]] ctrl %[[POS_CTRL_OUT_QUBITS_32]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled gphase
    // CHECK: %[[NEG_PI4_EX3:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: %[[POS_CTRL_OUT_QUBITS_38:.*]] = mqtopt.gphase(%[[NEG_PI4_EX3]] static [] mask [false]) ctrl %[[POS_CTRL_OUT_QUBITS_35]] : ctrl !mqtopt.Qubit

    // Controlled S
    // CHECK: %[[OUT_QUBITS_40:.*]], %[[POS_CTRL_OUT_QUBITS_41:.*]] = mqtopt.s(static [] mask []) %[[OUT_QUBITS_34]] ctrl %[[POS_CTRL_OUT_QUBITS_35]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled S† (phase gate with -π/2)
    // CHECK: %[[NEG_PI2_EX4:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_43:.*]], %[[POS_CTRL_OUT_QUBITS_44:.*]] = mqtopt.p(%[[NEG_PI2_EX4]] static [] mask [false]) %[[OUT_QUBITS_40]] ctrl %[[POS_CTRL_OUT_QUBITS_41]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled T
    // CHECK: %[[OUT_QUBITS_46:.*]], %[[POS_CTRL_OUT_QUBITS_47:.*]] = mqtopt.t(static [] mask []) %[[OUT_QUBITS_43]] ctrl %[[POS_CTRL_OUT_QUBITS_44]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled T† (phase gate with -π/4)
    // CHECK: %[[NEG_PI4_EX4:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: %[[OUT_QUBITS_49:.*]], %[[POS_CTRL_OUT_QUBITS_50:.*]] = mqtopt.p(%[[NEG_PI4_EX4]] static [] mask [false]) %[[OUT_QUBITS_46]] ctrl %[[POS_CTRL_OUT_QUBITS_47]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Reinsertion
    // CHECK: %[[Q0_I64_2:.*]] = tensor.extract %[[C0]][] : tensor<i64>
    // CHECK: %[[Q0_IDX_2:.*]] = arith.index_cast %[[Q0_I64_2]] : i64 to index
    // CHECK: memref.store %[[OUT_QUBITS_49]], %[[ALLOC]][%[[Q0_IDX_2]]] : memref<2x!mqtopt.Qubit>
    // CHECK: %[[C1_EX2:.*]] = tensor.extract %[[C1]][] : tensor<i64>
    // CHECK: %[[C1_IDX_2:.*]] = arith.index_cast %[[C1_EX2]] : i64 to index
    // CHECK: memref.store %[[POS_CTRL_OUT_QUBITS_50]], %[[ALLOC]][%[[C1_IDX_2]]] : memref<2x!mqtopt.Qubit>

    // CHECK: memref.dealloc %[[ALLOC]] : memref<2x!mqtopt.Qubit>
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Clifford: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_to_catalyst = """
    // Allocate and extract first qubit
    // CHECK: %[[ALLOC:.*]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: %[[Q0_I64:.*]] = arith.index_cast %{{.*}} : i64 to index
    // CHECK: %[[Q0_IDX:.*]] = arith.index_cast %[[Q0_I64]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[ALLOC]][%[[Q0_IDX]]] : !quantum.reg -> !quantum.bit

    // Uncontrolled Clifford+T gates
    // Hadamard
    // CHECK: %[[H0:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit

    // SX decomposed to RZ -> RY -> RZ with 2 gphase
    // CHECK: %[[PI2:.*]] = stablehlo.constant dense<1.5707963267948966> : tensor<f64>
    // CHECK: %[[PI2_EX1:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[SX_RZ1:.*]] = quantum.custom "RZ"(%[[PI2_EX1]]) %[[H0]] : !quantum.bit
    // CHECK: %[[PI2_EX2:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[SX_RY:.*]] = quantum.custom "RY"(%[[PI2_EX2]]) %[[SX_RZ1]] : !quantum.bit
    // CHECK: %[[NEG_PI2:.*]] = stablehlo.constant dense<-1.5707963267948966> : tensor<f64>
    // CHECK: %[[NEG_PI2_EX1:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[SX_RZ2:.*]] = quantum.custom "RZ"(%[[NEG_PI2_EX1]]) %[[SX_RY]] : !quantum.bit
    // CHECK: %[[NEG_PI4:.*]] = stablehlo.constant dense<-0.78539816339744828> : tensor<f64>
    // CHECK: %[[NEG_PI4_EX1:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: quantum.gphase(%[[NEG_PI4_EX1]]) :
    // CHECK: %[[NEG_PI4_EX2:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: quantum.gphase(%[[NEG_PI4_EX2]]) :

    // SX† decomposed to RZ -> RY -> RZ
    // CHECK: %[[NEG_PI2_EX2:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[SXD_RZ1:.*]] = quantum.custom "RZ"(%[[NEG_PI2_EX2]]) %[[SX_RZ2]] : !quantum.bit
    // CHECK: %[[PI2_EX3:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[SXD_RY:.*]] = quantum.custom "RY"(%[[PI2_EX3]]) %[[SXD_RZ1]] : !quantum.bit
    // CHECK: %[[PI2_EX4:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[SXD_RZ2:.*]] = quantum.custom "RZ"(%[[PI2_EX4]]) %[[SXD_RY]] : !quantum.bit

    // S, S†, T, T†
    // CHECK: %[[S:.*]] = quantum.custom "S"() %[[SXD_RZ2]] : !quantum.bit
    // CHECK: %[[SD:.*]] = quantum.custom "S"() %[[S]] : !quantum.bit
    // CHECK: %[[T:.*]] = quantum.custom "T"() %[[SD]] : !quantum.bit
    // CHECK: %[[TD:.*]] = quantum.custom "T"() %[[T]] : !quantum.bit

    // Extract control qubit
    // CHECK: %[[C1_I64:.*]] = arith.index_cast %{{.*}} : i64 to index
    // CHECK: %[[C1_IDX:.*]] = arith.index_cast %[[C1_I64]] : index to i64
    // CHECK: %[[CTRL0:.*]] = quantum.extract %[[ALLOC]][%[[C1_IDX]]] : !quantum.reg -> !quantum.bit

    // Controlled Clifford+T gates
    // Controlled Hadamard
    // CHECK: %[[TRUE1:.*]] = arith.constant true
    // CHECK: %[[FALSE1:.*]] = arith.constant false
    // CHECK: %[[CH_T:.*]], %[[CH_C:.*]] = quantum.custom "Hadamard"() %[[TD]] ctrls(%[[CTRL0]]) ctrlvals(%[[TRUE1]]) : !quantum.bit ctrls !quantum.bit

    // Controlled SX decomposed to CRZ -> CRY -> CRZ with controlled gphase
    // CHECK: %[[PI2_EX5:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[TRUE2:.*]] = arith.constant true
    // CHECK: %[[FALSE2:.*]] = arith.constant false
    // CHECK: %[[CSX_RZ1_T:.*]], %[[CSX_RZ1_C:.*]] = quantum.custom "CRZ"(%[[PI2_EX5]]) %[[CH_T]] ctrls(%[[CH_C]]) ctrlvals(%[[TRUE2]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[PI2_EX6:.*]] = tensor.extract %[[PI2]][] : tensor<f64>
    // CHECK: %[[TRUE3:.*]] = arith.constant true
    // CHECK: %[[FALSE3:.*]] = arith.constant false
    // CHECK: %[[CSX_RY_T:.*]], %[[CSX_RY_C:.*]] = quantum.custom "CRY"(%[[PI2_EX6]]) %[[CSX_RZ1_T]] ctrls(%[[CSX_RZ1_C]]) ctrlvals(%[[TRUE3]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[NEG_PI2_EX3:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[TRUE4:.*]] = arith.constant true
    // CHECK: %[[FALSE4:.*]] = arith.constant false
    // CHECK: %[[CSX_RZ2_T:.*]], %[[CSX_RZ2_C:.*]] = quantum.custom "CRZ"(%[[NEG_PI2_EX3]]) %[[CSX_RY_T]] ctrls(%[[CSX_RY_C]]) ctrlvals(%[[TRUE4]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[NEG_PI4_EX3:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: %[[TRUE5:.*]] = arith.constant true
    // CHECK: %[[FALSE5:.*]] = arith.constant false
    // CHECK: %[[GPHASE:.*]] = quantum.gphase(%[[NEG_PI4_EX3]]) ctrls(%[[CSX_RZ2_C]]) ctrlvals(%[[TRUE5]]) : ctrls !quantum.bit

    // Controlled S
    // CHECK: %[[TRUE6:.*]] = arith.constant true
    // CHECK: %[[FALSE6:.*]] = arith.constant false
    // CHECK: %[[CS_T:.*]], %[[CS_C:.*]] = quantum.custom "S"() %[[CSX_RZ2_T]] ctrls(%[[CSX_RZ2_C]]) ctrlvals(%[[TRUE6]]) : !quantum.bit ctrls !quantum.bit

    // Controlled S† (as ControlledPhaseShift)
    // CHECK: %[[NEG_PI2_EX4:.*]] = tensor.extract %[[NEG_PI2]][] : tensor<f64>
    // CHECK: %[[TRUE7:.*]] = arith.constant true
    // CHECK: %[[FALSE7:.*]] = arith.constant false
    // CHECK: %[[CSD_T:.*]], %[[CSD_C:.*]] = quantum.custom "ControlledPhaseShift"(%[[NEG_PI2_EX4]]) %[[CS_T]] ctrls(%[[CS_C]]) ctrlvals(%[[TRUE7]]) : !quantum.bit ctrls !quantum.bit

    // Controlled T
    // CHECK: %[[TRUE8:.*]] = arith.constant true
    // CHECK: %[[FALSE8:.*]] = arith.constant false
    // CHECK: %[[CT_T:.*]], %[[CT_C:.*]] = quantum.custom "T"() %[[CSD_T]] ctrls(%[[CSD_C]]) ctrlvals(%[[TRUE8]]) : !quantum.bit ctrls !quantum.bit

    // Controlled T† (as ControlledPhaseShift)
    // CHECK: %[[NEG_PI4_EX4:.*]] = tensor.extract %[[NEG_PI4]][] : tensor<f64>
    // CHECK: %[[TRUE9:.*]] = arith.constant true
    // CHECK: %[[FALSE9:.*]] = arith.constant false
    // CHECK: %[[CTD_T:.*]], %[[CTD_C:.*]] = quantum.custom "ControlledPhaseShift"(%[[NEG_PI4_EX4]]) %[[CT_T]] ctrls(%[[CT_C]]) ctrlvals(%[[TRUE9]]) : !quantum.bit ctrls !quantum.bit

    // Reinsertion of target and control qubits
    // CHECK: %[[INS1_I64:.*]] = arith.index_cast %{{.*}} : i64 to index
    // CHECK: %[[INS1_IDX:.*]] = arith.index_cast %[[INS1_I64]] : index to i64
    // CHECK: %[[INS1:.*]] = quantum.insert %[[ALLOC]][%[[INS1_IDX]]], %[[CTD_T]] : !quantum.reg, !quantum.bit
    // CHECK: %[[INS2_I64:.*]] = arith.index_cast %{{.*}} : i64 to index
    // CHECK: %[[INS2_IDX:.*]] = arith.index_cast %[[INS2_I64]] : index to i64
    // CHECK: %[[INS2:.*]] = quantum.insert %[[ALLOC]][%[[INS2_IDX]]], %[[CTD_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[ALLOC]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Clifford: MQTOpt to CatalystQuantum")


def test_pauli_gates_roundtrip() -> None:
    """Test roundtrip conversion of Pauli gates.

    Mirrors: quantum_pauli.mlir
    Gates: X, Y, Z, I, and their controlled variants (CNOT, CY, CZ, Toffoli)
    Structure:
    1. Uncontrolled Pauli gates (X, Y, Z, I)
    2. Controlled Pauli gates (using qml.ctrl on Pauli gates)
    3. Two-qubit controlled gates (CNOT, CY, CZ)
    4. Toffoli (CCX)
    5. Controlled two-qubit gates (controlled CNOT, CY, CZ, Toffoli)

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=4))
    def circuit() -> None:
        # Uncontrolled Pauli gates
        qml.PauliX(wires=0)
        qml.PauliY(wires=0)
        qml.PauliZ(wires=0)
        qml.Identity(wires=0)

        # Controlled Pauli gates (single control) - use qml.ctrl on Pauli gates
        qml.ctrl(qml.PauliX(wires=0), control=1)
        qml.ctrl(qml.PauliY(wires=0), control=1)
        qml.ctrl(qml.PauliZ(wires=0), control=1)
        # Why is `qml.ctrl(qml.Identity(wires=0), control=1)` not supported by Catalyst?

        # Two-qubit controlled gates (explicit CNOT, CY, CZ gate names)
        qml.CNOT(wires=[1, 0]) # First qubit is control, second is target
        qml.CY(wires=[1, 0])
        qml.CZ(wires=[1, 0])

        # Toffoli (also CCX)
        qml.Toffoli(wires=[0, 1, 2])

        # Controlled multi-qubit gates (adding extra controls)
        qml.ctrl(qml.CNOT(wires=[0, 1]), control=2)
        qml.ctrl(qml.CY(wires=[0, 1]), control=2)
        qml.ctrl(qml.CZ(wires=[0, 1]), control=2)
        qml.ctrl(qml.Toffoli(wires=[0, 1, 2]), control=3)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // CHECK: func.func public @circuit()
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4x!mqtopt.Qubit>
    // CHECK: %[[Q0_IDX:.*]] = arith.index_cast
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][%[[Q0_IDX]]] : memref<4x!mqtopt.Qubit>
    
    // Uncontrolled Pauli gates on Q0
    // CHECK: %[[X:.*]] = mqtopt.x(static [] mask []) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[Y:.*]] = mqtopt.y(static [] mask []) %[[X]] : !mqtopt.Qubit
    // CHECK: %[[Z:.*]] = mqtopt.z(static [] mask []) %[[Y]] : !mqtopt.Qubit
    // CHECK: %[[I:.*]] = mqtopt.i(static [] mask []) %[[Z]] : !mqtopt.Qubit
    
    // Load Q1 for controlled Pauli gates
    // CHECK: %[[Q1_IDX:.*]] = arith.index_cast
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][%[[Q1_IDX]]] : memref<4x!mqtopt.Qubit>
    
    // Controlled Pauli gates (qml.ctrl on Pauli gates)
    // CHECK: %[[CX1_T:.*]], %[[CX1_C:.*]] = mqtopt.x(static [] mask []) %[[I]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CY1_T:.*]], %[[CY1_C:.*]] = mqtopt.y(static [] mask []) %[[CX1_T]] ctrl %[[CX1_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CZ1_T:.*]], %[[CZ1_C:.*]] = mqtopt.z(static [] mask []) %[[CY1_T]] ctrl %[[CY1_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    
    // Two-qubit controlled gates (CNOT, CY, CZ)
    // CHECK: %[[CNOT_T:.*]], %[[CNOT_C:.*]] = mqtopt.x(static [] mask []) %[[CZ1_C]] ctrl %[[CZ1_T]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CY_T:.*]], %[[CY_C:.*]] = mqtopt.y(static [] mask []) %[[CNOT_T]] ctrl %[[CNOT_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CZ_T:.*]], %[[CZ_C:.*]] = mqtopt.z(static [] mask []) %[[CY_T]] ctrl %[[CY_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    
    // Load Q2 for Toffoli
    // CHECK: %[[Q2_IDX:.*]] = arith.index_cast
    // CHECK: %[[Q2:.*]] = memref.load %[[ALLOC]][%[[Q2_IDX]]] : memref<4x!mqtopt.Qubit>
    
    // Toffoli (X with 2 controls: Q0, Q1 -> Q2)
    // CHECK: %[[TOF_T:.*]], %[[TOF_C:.*]]:2 = mqtopt.x(static [] mask []) %[[Q2]] ctrl %[[CZ_C]], %[[CZ_T]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    
    // Controlled CNOT (adds Q2 as control to CNOT on Q0, Q1)
    // CHECK: %[[CCNOT_T:.*]], %[[CCNOT_C:.*]]:2 = mqtopt.x(static [] mask []) %[[TOF_C]]#1 ctrl %[[TOF_T]], %[[TOF_C]]#0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    
    // Controlled CY
    // CHECK: %[[CCY_T:.*]], %[[CCY_C:.*]]:2 = mqtopt.y(static [] mask []) %[[CCNOT_T]] ctrl %[[CCNOT_C]]#0, %[[CCNOT_C]]#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    
    // Controlled CZ
    // CHECK: %[[CCZ_T:.*]], %[[CCZ_C:.*]]:2 = mqtopt.z(static [] mask []) %[[CCY_T]] ctrl %[[CCY_C]]#0, %[[CCY_C]]#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    
    // Load Q3 for controlled Toffoli
    // CHECK: %[[Q3_IDX:.*]] = arith.index_cast
    // CHECK: %[[Q3:.*]] = memref.load %[[ALLOC]][%[[Q3_IDX]]] : memref<4x!mqtopt.Qubit>
    
    // Controlled Toffoli (X with 3 controls: Q3, Q1, Q0 -> Q2)
    // CHECK: %[[CTOF_T:.*]], %[[CTOF_C:.*]]:3 = mqtopt.x(static [] mask []) %[[CCZ_C]]#0 ctrl %[[Q3]], %[[CCZ_C]]#1, %[[CCZ_T]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit
    
    // Store qubits back
    // CHECK: memref.store %[[CTOF_C]]#1, %[[ALLOC]]
    // CHECK: memref.store %[[CTOF_C]]#2, %[[ALLOC]]
    // CHECK: memref.store %[[CTOF_T]], %[[ALLOC]]
    // CHECK: memref.store %[[CTOF_C]]#0, %[[ALLOC]]
    // CHECK: memref.dealloc %[[ALLOC]] : memref<4x!mqtopt.Qubit>
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Pauli: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_to_catalyst = """
    // CHECK: func.func public @circuit()
    // CHECK: %[[QREG:.*]] = quantum.alloc( 4) : !quantum.reg
    
    // Extract Q0
    // CHECK: %[[Q0_IDX1:.*]] = arith.index_cast
    // CHECK: %[[Q0_IDX2:.*]] = arith.index_cast %[[Q0_IDX1]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][%[[Q0_IDX2]]] : !quantum.reg -> !quantum.bit
    
    // Uncontrolled Pauli gates on Q0
    // CHECK: %[[X:.*]] = quantum.custom "PauliX"() %[[Q0]] : !quantum.bit
    // CHECK: %[[Y:.*]] = quantum.custom "PauliY"() %[[X]] : !quantum.bit
    // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[Y]] : !quantum.bit
    // CHECK: %[[I:.*]] = quantum.custom "Identity"() %[[Z]] : !quantum.bit
    
    // Extract Q1
    // CHECK: %[[Q1_IDX1:.*]] = arith.index_cast
    // CHECK: %[[Q1_IDX2:.*]] = arith.index_cast %[[Q1_IDX1]] : index to i64
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][%[[Q1_IDX2]]] : !quantum.reg -> !quantum.bit
    
    // Controlled Pauli gates (qml.ctrl on X, Y, Z) - results swap for each gate
    // CHECK: %[[TRUE1:.*]] = arith.constant true
    // CHECK: %[[FALSE1:.*]] = arith.constant false
    // CHECK: %[[CX1_T:.*]], %[[CX1_C:.*]] = quantum.custom "CNOT"() %[[I]] ctrls(%[[Q1]]) ctrlvals(%[[TRUE1]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE2:.*]] = arith.constant true
    // CHECK: %[[FALSE2:.*]] = arith.constant false
    // CHECK: %[[CY1_T:.*]], %[[CY1_C:.*]] = quantum.custom "CY"() %[[CX1_T]] ctrls(%[[CX1_C]]) ctrlvals(%[[TRUE2]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE3:.*]] = arith.constant true
    // CHECK: %[[FALSE3:.*]] = arith.constant false
    // CHECK: %[[CZ1_T:.*]], %[[CZ1_C:.*]] = quantum.custom "CZ"() %[[CY1_T]] ctrls(%[[CY1_C]]) ctrlvals(%[[TRUE3]]) : !quantum.bit ctrls !quantum.bit
    
    // Two-qubit controlled gates (CNOT, CY, CZ) - results swap
    // CHECK: %[[TRUE4:.*]] = arith.constant true
    // CHECK: %[[FALSE4:.*]] = arith.constant false
    // CHECK: %[[CNOT_T:.*]], %[[CNOT_C:.*]] = quantum.custom "CNOT"() %[[CZ1_C]] ctrls(%[[CZ1_T]]) ctrlvals(%[[TRUE4]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE5:.*]] = arith.constant true
    // CHECK: %[[FALSE5:.*]] = arith.constant false
    // CHECK: %[[CY_T:.*]], %[[CY_C:.*]] = quantum.custom "CY"() %[[CNOT_T]] ctrls(%[[CNOT_C]]) ctrlvals(%[[TRUE5]]) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TRUE6:.*]] = arith.constant true
    // CHECK: %[[FALSE6:.*]] = arith.constant false
    // CHECK: %[[CZ_T:.*]], %[[CZ_C:.*]] = quantum.custom "CZ"() %[[CY_T]] ctrls(%[[CY_C]]) ctrlvals(%[[TRUE6]]) : !quantum.bit ctrls !quantum.bit
    
    // Extract Q2 for Toffoli
    // CHECK: %[[Q2_IDX1:.*]] = arith.index_cast
    // CHECK: %[[Q2_IDX2:.*]] = arith.index_cast %[[Q2_IDX1]] : index to i64
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][%[[Q2_IDX2]]] : !quantum.reg -> !quantum.bit
    
    // Toffoli (2 controls + target)
    // CHECK: %[[TRUE7:.*]] = arith.constant true
    // CHECK: %[[FALSE7:.*]] = arith.constant false
    // CHECK: %[[TOF_T:.*]], %[[TOF_C:.*]]:2 = quantum.custom "Toffoli"() %[[Q2]] ctrls(%[[CZ_C]], %[[CZ_T]]) ctrlvals(%[[TRUE7]], %[[TRUE7]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    
    // Controlled CNOT (becomes Toffoli with 3 qubits)
    // CHECK: %[[TRUE8:.*]] = arith.constant true
    // CHECK: %[[FALSE8:.*]] = arith.constant false
    // CHECK: %[[CCNOT_T:.*]], %[[CCNOT_C:.*]]:2 = quantum.custom "Toffoli"() %[[TOF_C]]#1 ctrls(%[[TOF_T]], %[[TOF_C]]#0) ctrlvals(%[[TRUE8]], %[[TRUE8]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    
    // Controlled CY (PauliY with 2 controls)
    // CHECK: %[[TRUE9:.*]] = arith.constant true
    // CHECK: %[[FALSE9:.*]] = arith.constant false
    // CHECK: %[[CCY_T:.*]], %[[CCY_C:.*]]:2 = quantum.custom "PauliY"() %[[CCNOT_T]] ctrls(%[[CCNOT_C]]#0, %[[CCNOT_C]]#1) ctrlvals(%[[TRUE9]], %[[TRUE9]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    
    // Controlled CZ (PauliZ with 2 controls)
    // CHECK: %[[TRUE10:.*]] = arith.constant true
    // CHECK: %[[FALSE10:.*]] = arith.constant false
    // CHECK: %[[CCZ_T:.*]], %[[CCZ_C:.*]]:2 = quantum.custom "PauliZ"() %[[CCY_T]] ctrls(%[[CCY_C]]#0, %[[CCY_C]]#1) ctrlvals(%[[TRUE10]], %[[TRUE10]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    
    // Extract Q3 for controlled Toffoli
    // CHECK: %[[Q3_IDX1:.*]] = arith.index_cast
    // CHECK: %[[Q3_IDX2:.*]] = arith.index_cast %[[Q3_IDX1]] : index to i64
    // CHECK: %[[Q3:.*]] = quantum.extract %[[QREG]][%[[Q3_IDX2]]] : !quantum.reg -> !quantum.bit
    
    // Controlled Toffoli (PauliX with 3 controls)
    // CHECK: %[[TRUE11:.*]] = arith.constant true
    // CHECK: %[[FALSE11:.*]] = arith.constant false
    // CHECK: %[[CTOF_T:.*]], %[[CTOF_C:.*]]:3 = quantum.custom "PauliX"() %[[CCZ_C]]#0 ctrls(%[[Q3]], %[[CCZ_C]]#1, %[[CCZ_T]]) ctrlvals(%[[TRUE11]], %[[TRUE11]], %[[TRUE11]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
    
    // Insert qubits back to register
    // CHECK: %[[INS1_IDX1:.*]] = arith.index_cast
    // CHECK: %[[INS1_IDX2:.*]] = arith.index_cast %[[INS1_IDX1]] : index to i64
    // CHECK: %{{.*}} = quantum.insert %[[QREG]][%[[INS1_IDX2]]], %[[CTOF_C]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[INS2_IDX1:.*]] = arith.index_cast
    // CHECK: %[[INS2_IDX2:.*]] = arith.index_cast %[[INS2_IDX1]] : index to i64
    // CHECK: %{{.*}} = quantum.insert %[[QREG]][%[[INS2_IDX2]]], %[[CTOF_C]]#2 : !quantum.reg, !quantum.bit
    // CHECK: %[[INS3_IDX1:.*]] = arith.index_cast
    // CHECK: %[[INS3_IDX2:.*]] = arith.index_cast %[[INS3_IDX1]] : index to i64
    // CHECK: %{{.*}} = quantum.insert %[[QREG]][%[[INS3_IDX2]]], %[[CTOF_T]] : !quantum.reg, !quantum.bit
    // CHECK: %[[INS4_IDX1:.*]] = arith.index_cast
    // CHECK: %[[INS4_IDX2:.*]] = arith.index_cast %[[INS4_IDX1]] : index to i64
    // CHECK: %{{.*}} = quantum.insert %[[QREG]][%[[INS4_IDX2]]], %[[CTOF_C]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Pauli: MQTOpt to CatalystQuantum")


def test_parameterized_gates_roundtrip() -> None:
    """Test roundtrip conversion of parameterized rotation gates.

    Mirrors: quantum_param.mlir
    Gates: RX, RY, RZ, PhaseShift, and their controlled variants (CRX, CRY)
    Note: MLIR test does NOT include CRZ, only CRX and CRY

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        angle = 0.3

        # Non-controlled parameterized gates
        qml.RX(angle, wires=0)
        qml.RY(angle, wires=0)
        qml.RZ(angle, wires=0)
        qml.PhaseShift(angle, wires=0)

        # Controlled parameterized gates
        qml.CRX(angle, wires=[1, 0])
        qml.CRY(angle, wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[ALLOC:.*]] = memref.alloc(){{.*}}!mqtopt.Qubit
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][{{.*}}]{{.*}}!mqtopt.Qubit

    // Uncontrolled parameterized gates
    // CHECK: %[[RX:.*]] = mqtopt.rx({{.*}}) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[RY:.*]] = mqtopt.ry({{.*}}) %[[RX]] : !mqtopt.Qubit
    // CHECK: %[[RZ:.*]] = mqtopt.rz({{.*}}) %[[RY]] : !mqtopt.Qubit
    // CHECK: %[[PS:.*]] = mqtopt.p({{.*}}) %[[RZ]] : !mqtopt.Qubit

    // Load Q1 for controlled parameterized gates
    // CHECK: memref.load{{.*}}!mqtopt.Qubit

    // Controlled parameterized gates
    // CHECK: mqtopt.rx({{.*}}){{.*}}ctrl{{.*}}: !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: mqtopt.ry({{.*}}){{.*}}ctrl{{.*}}: !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Reinsertion
    // CHECK: memref.store
    // CHECK: memref.store
    // CHECK: memref.dealloc{{.*}}!mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Param: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg

    // Q0 is extracted first for uncontrolled gates
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Uncontrolled parameterized gates
    // CHECK: %[[RX:.*]] = quantum.custom "RX"({{.*}}) %[[Q0]] : !quantum.bit
    // CHECK: %[[RY:.*]] = quantum.custom "RY"({{.*}}) %[[RX]] : !quantum.bit
    // CHECK: %[[RZ:.*]] = quantum.custom "RZ"({{.*}}) %[[RY]] : !quantum.bit
    // CHECK: %[[PS:.*]] = quantum.custom "PhaseShift"({{.*}}) %[[RZ]] : !quantum.bit

    // Q1 is extracted lazily right before controlled gates
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled parameterized gates (qubits swap after each operation)
    // CRX: target=%[[PS]], control=%[[Q1]]
    // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = quantum.custom "CRX"({{.*}}) %[[PS]] ctrls(%[[Q1]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CRY: target=%[[CRX_C]] (previous control), control=%[[CRX_T]] (previous target)
    // CHECK: quantum.custom "CRY"({{.*}}) %[[CRX_C]] ctrls(%[[CRX_T]]) ctrlvals(%true{{.*}})
    // CHECK-SAME: : !quantum.bit ctrls !quantum.bit

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Param: MQTOpt to CatalystQuantum")


def test_entangling_gates_roundtrip() -> None:
    """Test roundtrip conversion of entangling/permutation gates.

    Mirrors: quantum_entangling.mlir
    Gates: SWAP, ISWAP, ISWAP†, ECR, and their controlled variants

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        # Uncontrolled permutation gates
        qml.SWAP(wires=[0, 1])
        qml.ISWAP(wires=[0, 1])
        qml.adjoint(qml.ISWAP(wires=[0, 1]))
        qml.ECR(wires=[0, 1])

        # Controlled permutation gates
        qml.CSWAP(wires=[2, 0, 1])
        qml.ctrl(qml.ISWAP(wires=[0, 1]), control=2)
        qml.ctrl(qml.adjoint(qml.ISWAP(wires=[0, 1])), control=2)
        qml.ctrl(qml.ECR(wires=[0, 1]), control=2)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: memref.alloc(){{.*}}!mqtopt.Qubit

    // Qubits loaded as needed
    // CHECK: memref.load{{.*}}!mqtopt.Qubit
    // CHECK: memref.load{{.*}}!mqtopt.Qubit

    // SWAP gate (ISWAP/ECR get decomposed)
    // CHECK: mqtopt.swap({{.*}}){{.*}}: !mqtopt.Qubit, !mqtopt.Qubit

    // Control qubit loaded
    // CHECK: memref.load{{.*}}!mqtopt.Qubit

    // Controlled swap gate
    // CHECK: mqtopt.swap({{.*}}){{.*}}ctrl{{.*}}: !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Reinsertion
    // CHECK: memref.store
    // CHECK: memref.store
    // CHECK: memref.store
    // CHECK: memref.dealloc{{.*}}!mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Entangling: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion (simplified - ISWAP/ECR are heavily decomposed)
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg

    // Qubits extracted as needed
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // SWAP is visible, but ISWAP/ECR/adjISWAP are heavily decomposed into primitives (H, S, CNOT, RZ, RY chains)
    // CHECK: quantum.custom "SWAP"() %[[Q0]], %[[Q1]] : !quantum.bit, !quantum.bit

    // After all decompositions, Q2 extracted for controlled gates
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled swap
    // CHECK: quantum.custom "CSWAP"() {{.*}} ctrls(%[[Q2]]) ctrlvals(

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Entangling: MQTOpt to CatalystQuantum")


def test_ising_gates_roundtrip() -> None:
    """Test roundtrip conversion of Ising-type gates.

    Mirrors: quantum_ising.mlir
    Gates: IsingXY, IsingXX, IsingYY, IsingZZ, and their controlled variants
    Note: IsingXY takes 2 parameters in MLIR (phi and beta)

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        angle = 0.3

        # Uncontrolled Ising gates
        qml.IsingXY(angle, wires=[0, 1])
        qml.IsingXX(angle, wires=[0, 1])
        qml.IsingYY(angle, wires=[0, 1])
        qml.IsingZZ(angle, wires=[0, 1])

        # Controlled Ising gates
        qml.ctrl(qml.IsingXY(angle, wires=[0, 1]), control=2)
        qml.ctrl(qml.IsingXX(angle, wires=[0, 1]), control=2)
        qml.ctrl(qml.IsingYY(angle, wires=[0, 1]), control=2)
        qml.ctrl(qml.IsingZZ(angle, wires=[0, 1]), control=2)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    # This works regardless of where pytest is run from (locally or CI)
    test_file_dir = Path(__file__).parent
    mlir_dir = Path.cwd()
    test_file_dir.parent / "Conversion"

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        # Fallback: list what files actually exist for debugging
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion with FileCheck
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[ALLOC:.*]] = memref.alloc(){{.*}}!mqtopt.Qubit
    // CHECK: %[[Q0:.*]] = memref.load{{.*}}!mqtopt.Qubit
    // CHECK: %[[Q1:.*]] = memref.load{{.*}}!mqtopt.Qubit

    // Uncontrolled Ising gates
    // CHECK: %[[XY_OUT:.*]]:2 = mqtopt.xx_plus_yy({{.*}}) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[XX_OUT:.*]]:2 = mqtopt.rxx({{.*}}) %[[XY_OUT]]#0, %[[XY_OUT]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[YY_OUT:.*]]:2 = mqtopt.ryy({{.*}}) %[[XX_OUT]]#0, %[[XX_OUT]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ZZ_OUT:.*]]:2 = mqtopt.rzz({{.*}}) %[[YY_OUT]]#0, %[[YY_OUT]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled Ising gates (control qubit loaded here)
    // CHECK: memref.load{{.*}}!mqtopt.Qubit
    // CHECK: mqtopt.xx_plus_yy({{.*}}){{.*}}ctrl{{.*}}
    // CHECK: mqtopt.rxx({{.*}}){{.*}}ctrl{{.*}}
    // CHECK: mqtopt.ryy({{.*}}){{.*}}ctrl{{.*}}
    // CHECK: mqtopt.rzz({{.*}}){{.*}}ctrl{{.*}}

    // Reinsertion
    // CHECK: memref.store
    // CHECK: memref.store
    // CHECK: memref.store
    // CHECK: memref.dealloc{{.*}}!mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Ising: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion with FileCheck
    # Based on mqtopt_ising.mlir reference test
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Uncontrolled Ising gates
    // IsingXY is decomposed: RZ -> IsingXY -> RZ
    // CHECK: %[[RZ0:.*]] = quantum.custom "RZ"({{.*}}) %[[Q1]] : !quantum.bit
    // CHECK: %[[XY:.*]]:2 = quantum.custom "IsingXY"({{.*}}) %[[Q0]], %[[RZ0]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"({{.*}}) %[[XY]]#1 : !quantum.bit

    // IsingXX, IsingYY, IsingZZ gates
    // CHECK: %[[XX:.*]]:2 = quantum.custom "IsingXX"({{.*}}) %[[XY]]#0, %[[RZ1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[YY:.*]]:2 = quantum.custom "IsingYY"({{.*}}) %[[XX]]#0, %[[XX]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ZZ:.*]]:2 = quantum.custom "IsingZZ"({{.*}}) %[[YY]]#0, %[[YY]]#1 : !quantum.bit, !quantum.bit

    // Controlled Ising gates (with ctrls)
    // Extract control qubit
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled IsingXY: RZ(ctrl) -> IsingXY(ctrl) -> RZ(ctrl)
    // CHECK: %[[CRZ0:.*]], %[[CTRL1:.*]] = quantum.custom "RZ"({{.*}}) %[[ZZ]]#1 ctrls(%[[Q2]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CXY:.*]]:2, %[[CTRL2:.*]] = quantum.custom "IsingXY"({{.*}}) %[[ZZ]]#0, %[[CRZ0]]
    // CHECK-SAME: ctrls(%[[CTRL1]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ1:.*]], %[[CTRL3:.*]] = quantum.custom "RZ"({{.*}}) %[[CXY]]#1 ctrls(%[[CTRL2]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Controlled IsingXX, IsingYY, IsingZZ
    // CHECK: %[[CXX:.*]]:2, %[[CTRL4:.*]] = quantum.custom "IsingXX"({{.*}}) %[[CXY]]#0, %[[CRZ1]]
    // CHECK-SAME: ctrls(%[[CTRL3]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CYY:.*]]:2, %[[CTRL5:.*]] = quantum.custom "IsingYY"({{.*}}) %[[CXX]]#0, %[[CXX]]#1
    // CHECK-SAME: ctrls(%[[CTRL4]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CZZ:.*]]:2, %[[CTRL6:.*]] = quantum.custom "IsingZZ"({{.*}}) %[[CYY]]#0, %[[CYY]]#1
    // CHECK-SAME: ctrls(%[[CTRL5]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CZZ]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CZZ]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CTRL6]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Ising: MQTOpt to CatalystQuantum")

    # Remove all intermediate files created during the test
    for mlir_file in mlir_dir.glob("*.mlir"):
        mlir_file.unlink()


def test_mqtopt_roundtrip() -> None:
    """Execute the full roundtrip including MQT Core IR.

    Executes the conversion passes to and from MQTOpt dialect AND
    the roundtrip through MQT Core IR.
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[0, 1])

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # This will execute the pass and return the final MLIR
    mlir_opt = module.mlir_opt
    assert mlir_opt


def test_debug_roundtrip() -> None:
    """Test roundtrip conversion.
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[0, 1]) # first wire is control

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    # This works regardless of where pytest is run from (locally or CI)
    test_file_dir = Path(__file__).parent
    mlir_dir = Path.cwd()
    test_file_dir.parent / "Conversion"

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        # Fallback: list what files actually exist for debugging
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion with FileCheck
    check_to_mqtopt = """
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Ising: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion with FileCheck
    # Based on mqtopt_ising.mlir reference test
    check_to_catalyst = """
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Ising: MQTOpt to CatalystQuantum")

    # Remove all intermediate files created during the test
    for mlir_file in mlir_dir.glob("*.mlir"):
        mlir_file.unlink()