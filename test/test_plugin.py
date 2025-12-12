# Copyright (c) 2025 Chair for Design Automation, TUM
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
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pennylane as qml
import pytest
from catalyst.passes import apply_pass

from mqt.core.plugins.catalyst import get_device

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _cleanup_mlir_files() -> Generator[None, None, None]:
    """Clean up MLIR files before and after each test to ensure test isolation.

    Yields:
        None
    """
    _cleanup_mlir_artifacts()
    yield
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


def test_paulix_roundtrip() -> None:
    """Test roundtrip conversion of the PauliX gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        # Non-controlled
        qml.X(wires=0)
        qml.PauliX(wires=0)
        # Controlled
        qml.ctrl(qml.PauliX(wires=0), control=1)
        qml.CNOT(wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify original CatalystQuantum
    check_mlir_before = """
      //CHECK: %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
      //CHECK: %out_qubits_2 = quantum.custom "PauliX"() %out_qubits : !quantum.bit
      //CHECK: %out_qubits_5:2 = quantum.custom "CNOT"() %2, %out_qubits_2 : !quantum.bit, !quantum.bit
      //CHECK: %out_qubits_6:2 = quantum.custom "CNOT"() %out_qubits_5#0, %out_qubits_5#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PauliX: CatalystQuantum")

    # Verify CatalystQuantum → MQTOpt conversion
    check_after_mqtopt = """
      //CHECK: %out_qubits = mqtopt.x(static [] mask []) %1 : !mqtopt.Qubit
      //CHECK: %out_qubits_2 = mqtopt.x(static [] mask []) %out_qubits : !mqtopt.Qubit
      //CHECK: %out_qubits_5, %pos_ctrl_out_qubits = mqtopt.x(static [] mask []) %out_qubits_2 ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
      //CHECK: %out_qubits_6, %pos_ctrl_out_qubits_7 = mqtopt.x(static [] mask []) %out_qubits_5 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PauliX: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_after_catalyst = """
        //CHECK: %out_qubits = quantum.custom "PauliX"() %3 : !quantum.bit
        //CHECK: %out_qubits_2 = quantum.custom "PauliX"() %out_qubits : !quantum.bit
        //CHECK: %out_qubits_5, %out_ctrl_qubits = quantum.custom "CNOT"() %out_qubits_2 ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_8, %out_ctrl_qubits_9 = quantum.custom "CNOT"() %out_qubits_5 ctrls(%out_ctrl_qubits) ctrlvals(%true_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PauliX: MQTOpt to CatalystQuantum")


def test_pauliy_roundtrip() -> None:
    """Test roundtrip conversion of the PauliY gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        # Non-controlled
        qml.Y(wires=0)
        qml.PauliY(wires=0)
        # Controlled
        qml.ctrl(qml.PauliY(wires=0), control=1)
        qml.CY(wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify original CatalystQuantum
    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "PauliY"() %1 : !quantum.bit
        //CHECK: %out_qubits_2 = quantum.custom "PauliY"() %out_qubits : !quantum.bit
        //CHECK: %out_qubits_5:2 = quantum.custom "CY"() %2, %out_qubits_2 : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_6:2 = quantum.custom "CY"() %out_qubits_5#0, %out_qubits_5#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PauliY: CatalystQuantum")

    # Verify CatalystQuantum → MQTOpt conversion
    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.y(static [] mask []) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_2 = mqtopt.y(static [] mask []) %out_qubits : !mqtopt.Qubit
        //CHECK: %out_qubits_5, %pos_ctrl_out_qubits = mqtopt.y(static [] mask []) %out_qubits_2 ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_6, %pos_ctrl_out_qubits_7 = mqtopt.y(static [] mask []) %out_qubits_5 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PauliY: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_after_catalyst = """
        //CHECK: %out_qubits = quantum.custom "PauliY"() %3 : !quantum.bit
        //CHECK: %out_qubits_2 = quantum.custom "PauliY"() %out_qubits : !quantum.bit
        //CHECK: %out_qubits_5, %out_ctrl_qubits = quantum.custom "CY"() %out_qubits_2 ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_8, %out_ctrl_qubits_9 = quantum.custom "CY"() %out_qubits_5 ctrls(%out_ctrl_qubits) ctrlvals(%true_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PauliY: MQTOpt to CatalystQuantum")


def test_pauliz_roundtrip() -> None:
    """Test roundtrip conversion of the PauliZ gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        # Non-controlled
        qml.Z(wires=0)
        qml.PauliZ(wires=0)
        # Controlled
        qml.ctrl(qml.PauliZ(wires=0), control=1)
        qml.CZ(wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify original CatalystQuantum
    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "PauliZ"() %1 : !quantum.bit
        //CHECK: %out_qubits_2 = quantum.custom "PauliZ"() %out_qubits : !quantum.bit
        //CHECK: %out_qubits_5:2 = quantum.custom "CZ"() %2, %out_qubits_2 : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_6:2 = quantum.custom "CZ"() %out_qubits_5#0, %out_qubits_5#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PauliZ: CatalystQuantum")

    # Verify CatalystQuantum → MQTOpt conversion
    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.z(static [] mask []) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_2 = mqtopt.z(static [] mask []) %out_qubits : !mqtopt.Qubit
        //CHECK: %out_qubits_5, %pos_ctrl_out_qubits = mqtopt.z(static [] mask []) %out_qubits_2 ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_6, %pos_ctrl_out_qubits_7 = mqtopt.z(static [] mask []) %out_qubits_5 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PauliZ: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_after_catalyst = """
        //CHECK: %out_qubits = quantum.custom "PauliZ"() %3 : !quantum.bit
        //CHECK: %out_qubits_2 = quantum.custom "PauliZ"() %out_qubits : !quantum.bit
        //CHECK: %out_qubits_5, %out_ctrl_qubits = quantum.custom "CZ"() %out_qubits_2 ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_8, %out_ctrl_qubits_9 = quantum.custom "CZ"() %out_qubits_5 ctrls(%out_ctrl_qubits) ctrlvals(%true_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PauliZ: MQTOpt to CatalystQuantum")


def test_hadamard_roundtrip() -> None:
    """Test roundtrip conversion of the Hadamard gate.

    Raises:
            FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.Hadamard(wires=0)
        qml.ctrl(qml.Hadamard(wires=0), control=1)
        qml.CH(wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
        //CHECK: %out_qubits_6, %out_ctrl_qubits = quantum.custom "Hadamard"() %out_qubits ctrls(%2) ctrlvals(%extracted_5) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_8, %out_ctrl_qubits_9 = quantum.custom "Hadamard"() %out_qubits_6 ctrls(%out_ctrl_qubits) ctrlvals(%extracted_7) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "Hadamard: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.h(static [] mask []) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_6, %pos_ctrl_out_qubits = mqtopt.h(static [] mask []) %out_qubits ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_8, %pos_ctrl_out_qubits_9 = mqtopt.h(static [] mask []) %out_qubits_6 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "Hadamard: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %out_qubits = quantum.custom "Hadamard"() %3 : !quantum.bit
        //CHECK: %out_qubits_6, %out_ctrl_qubits = quantum.custom "Hadamard"() %out_qubits ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_10, %out_ctrl_qubits_11 = quantum.custom "Hadamard"() %out_qubits_6 ctrls(%out_ctrl_qubits) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "Hadamard: MQTOpt to CatalystQuantum")


def test_s_gate_roundtrip() -> None:
    """Test roundtrip conversion of the S gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.S(wires=0)
        qml.adjoint(qml.S(wires=0))
        qml.ctrl(qml.S(wires=0), control=1)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
      //CHECK: %out_qubits = quantum.custom "S"() %1 : !quantum.bit
      //CHECK: %out_qubits_2 = quantum.custom "S"() %out_qubits adj : !quantum.bit
      //CHECK: %out_qubits_7, %out_ctrl_qubits = quantum.custom "S"() %out_qubits_2 ctrls(%2) ctrlvals(%extracted_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "S: CatalystQuantum")

    check_after_mqtopt = """
      //CHECK: %out_qubits = mqtopt.s(static [] mask []) %1 : !mqtopt.Qubit
      //CHECK: %out_qubits_2 = mqtopt.sdg(static [] mask []) %out_qubits : !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "S: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
      //CHECK: %out_qubits = quantum.custom "S"() %3 : !quantum.bit
      //CHECK: %out_qubits_2 = quantum.custom "S"() %out_qubits adj : !quantum.bit
      //CHECK: %out_qubits_7, %out_ctrl_qubits = quantum.custom "S"() %out_qubits_2 ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "S: MQTOpt to CatalystQuantum")


def test_t_gate_roundtrip() -> None:
    """Test roundtrip conversion of the T gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.T(wires=0)
        qml.adjoint(qml.T(wires=0))
        qml.ctrl(qml.T(wires=0), control=1)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "T"() %1 : !quantum.bit
        //CHECK: %out_qubits_2 = quantum.custom "T"() %out_qubits adj : !quantum.bit
        //CHECK: %out_qubits_7, %out_ctrl_qubits = quantum.custom "T"() %out_qubits_2 ctrls(%2) ctrlvals(%extracted_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "T: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.t(static [] mask []) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_2 = mqtopt.tdg(static [] mask []) %out_qubits : !mqtopt.Qubit
        //CHECK: %out_qubits_7, %pos_ctrl_out_qubits = mqtopt.t(static [] mask []) %out_qubits_2 ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "T: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
      //CHECK: %out_qubits = quantum.custom "T"() %3 : !quantum.bit
      //CHECK: %out_qubits_2 = quantum.custom "T"() %out_qubits adj : !quantum.bit
      //CHECK: %out_qubits_7, %out_ctrl_qubits = quantum.custom "T"() %out_qubits_2 ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "T: MQTOpt to CatalystQuantum")


def test_rx_gate_roundtrip() -> None:
    """Test roundtrip conversion of the RX gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.RX(0.5, wires=0)
        qml.ctrl(qml.RX(0.5, wires=0), control=1)
        qml.CRX(0.5, wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "RX"({{.*}}) %1 : !quantum.bit
        //CHECK: %out_qubits_6:2 = quantum.custom "CRX"({{.*}}) %2, %out_qubits : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_8:2 = quantum.custom "CRX"(%extracted_7) %out_qubits_6#0, %out_qubits_6#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "RX: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.rx({{.*}}) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_6, %pos_ctrl_out_qubits = mqtopt.rx({{.*}}) %out_qubits ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_8, %pos_ctrl_out_qubits_9 = mqtopt.rx({{.*}}) %out_qubits_6 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "RX: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
      //CHECK: %out_qubits = quantum.custom "RX"({{.*}}) %3 : !quantum.bit
      //CHECK: %out_qubits_6, %out_ctrl_qubits = quantum.custom "CRX"({{.*}}) %out_qubits ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
      //CHECK: %out_qubits_10, %out_ctrl_qubits_11 = quantum.custom "CRX"(%extracted_7) %out_qubits_6 ctrls(%out_ctrl_qubits) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "RX: MQTOpt to CatalystQuantum")


def test_ry_gate_roundtrip() -> None:
    """Test roundtrip conversion of the RY gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.RY(0.5, wires=0)
        qml.ctrl(qml.RY(0.5, wires=0), control=1)
        qml.CRY(0.5, wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "RY"({{.*}}) %1 : !quantum.bit
        //CHECK: %out_qubits_6:2 = quantum.custom "CRY"({{.*}}) %2, %out_qubits : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_8:2 = quantum.custom "CRY"(%extracted_7) %out_qubits_6#0, %out_qubits_6#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "RY: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.ry({{.*}}) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_6, %pos_ctrl_out_qubits = mqtopt.ry({{.*}}) %out_qubits ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_8, %pos_ctrl_out_qubits_9 = mqtopt.ry(%extracted_7 static [] mask [false]) %out_qubits_6 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "RY: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %out_qubits = quantum.custom "RY"({{.*}}) %3 : !quantum.bit
        //CHECK: %out_qubits_6, %out_ctrl_qubits = quantum.custom "CRY"({{.*}}) %out_qubits ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_10, %out_ctrl_qubits_11 = quantum.custom "CRY"(%extracted_7) %out_qubits_6 ctrls(%out_ctrl_qubits) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "RY: MQTOpt to CatalystQuantum")


def test_rz_gate_roundtrip() -> None:
    """Test roundtrip conversion of the RZ gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.RZ(0.5, wires=0)
        qml.ctrl(qml.RZ(0.5, wires=0), control=1)
        qml.CRZ(0.5, wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "RZ"({{.*}}) %1 : !quantum.bit
        //CHECK: %out_qubits_6:2 = quantum.custom "CRZ"({{.*}}) %2, %out_qubits : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_8:2 = quantum.custom "CRZ"(%extracted_7) %out_qubits_6#0, %out_qubits_6#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "RZ: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.rz({{.*}}) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_6, %pos_ctrl_out_qubits = mqtopt.rz({{.*}}) %out_qubits ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_8, %pos_ctrl_out_qubits_9 = mqtopt.rz(%extracted_7 static [] mask [false]) %out_qubits_6 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "RZ: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %out_qubits = quantum.custom "RZ"({{.*}}) %3 : !quantum.bit
        //CHECK: %out_qubits_6, %out_ctrl_qubits = quantum.custom "CRZ"({{.*}}) %out_qubits ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_10, %out_ctrl_qubits_11 = quantum.custom "CRZ"(%extracted_7) %out_qubits_6 ctrls(%out_ctrl_qubits) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "RZ: MQTOpt to CatalystQuantum")


def test_phaseshift_gate_roundtrip() -> None:
    """Test roundtrip conversion of the PhaseShift gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.PhaseShift(0.5, wires=0)
        qml.ctrl(qml.PhaseShift(0.5, wires=0), control=1)
        qml.ControlledPhaseShift(0.5, wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits = quantum.custom "PhaseShift"({{.*}}) %1 : !quantum.bit
        //CHECK: %out_qubits_6:2 = quantum.custom "ControlledPhaseShift"({{.*}}) %2, %out_qubits : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_8:2 = quantum.custom "ControlledPhaseShift"(%extracted_7) %out_qubits_6#0, %out_qubits_6#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PhaseShift: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits = mqtopt.p({{.*}}) %1 : !mqtopt.Qubit
        //CHECK: %out_qubits_6, %pos_ctrl_out_qubits = mqtopt.p({{.*}}) %out_qubits ctrl %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_8, %pos_ctrl_out_qubits_9 = mqtopt.p({{.*}} static [] mask [false]) %out_qubits_6 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PhaseShift: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %out_qubits = quantum.custom "PhaseShift"({{.*}}) %3 : !quantum.bit
        //CHECK: %out_qubits_6, %out_ctrl_qubits = quantum.custom "ControlledPhaseShift"({{.*}}) %out_qubits ctrls(%6) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_10, %out_ctrl_qubits_11 = quantum.custom "ControlledPhaseShift"({{.*}}) %out_qubits_6 ctrls(%out_ctrl_qubits) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PhaseShift: MQTOpt to CatalystQuantum")


def test_swap_gate_roundtrip() -> None:
    """Test roundtrip conversion of the SWAP gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found

    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.SWAP(wires=[0, 1])
        qml.ctrl(qml.SWAP(wires=[0, 1]), control=2)
        qml.CSWAP(wires=[2, 0, 1])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits:2 = quantum.custom "SWAP"() %1, %2 : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_5:3 = quantum.custom "CSWAP"() %3, %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_6:3 = quantum.custom "CSWAP"() %out_qubits_5#0, %out_qubits_5#1, %out_qubits_5#2 : !quantum.bit, !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "SWAP: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits:2 = mqtopt.swap(static [] mask []) %1, %3 : !mqtopt.Qubit, !mqtopt.Qubit
        //CHECK: %out_qubits_5:2, %pos_ctrl_out_qubits = mqtopt.swap(static [] mask []) %out_qubits#0, %out_qubits#1 ctrl %5 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %out_qubits_6:2, %pos_ctrl_out_qubits_7 = mqtopt.swap(static [] mask []) %out_qubits_5#0, %out_qubits_5#1 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "SWAP: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %out_qubits:2 = quantum.custom "SWAP"() %3, %6 : !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_5:2, %out_ctrl_qubits = quantum.custom "CSWAP"() %out_qubits#0, %out_qubits#1 ctrls(%9) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
        //CHECK: %out_qubits_8:2, %out_ctrl_qubits_9 = quantum.custom "CSWAP"() %out_qubits_5#0, %out_qubits_5#1 ctrls(%out_ctrl_qubits) ctrlvals(%true_6) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "SWAP: MQTOpt to CatalystQuantum")


def test_toffoli_gate_roundtrip() -> None:
    """Test roundtrip conversion of the Toffoli gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")  # type: ignore[untyped-decorator]
    @apply_pass("mqt.catalystquantum-to-mqtopt")  # type: ignore[untyped-decorator]
    @qml.qnode(get_device("lightning.qubit", wires=2))  # type: ignore[untyped-decorator]
    def circuit() -> None:
        qml.Toffoli(wires=[0, 1, 2])
        qml.ctrl(qml.Toffoli(wires=[0, 1, 2]), control=3)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)  # type: ignore[untyped-decorator]
    def module() -> Any:  # noqa: ANN401
        return circuit()

    mlir_opt = module.mlir_opt
    assert mlir_opt

    mlir_dir = Path.cwd()
    catalyst_mlir = mlir_dir / "0_catalyst_module.mlir"
    mlir_to_mqtopt = mlir_dir / "1_CatalystQuantumToMQTOpt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not catalyst_mlir.exists() or not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(catalyst_mlir).open("r", encoding="utf-8") as f:
        mlir_before = f.read()
    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    check_mlir_before = """
        //CHECK: %out_qubits:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_11, %out_ctrl_qubits:3 = quantum.custom "PauliX"() %out_qubits#2 ctrls(%4, %out_qubits#0, %out_qubits#1) ctrlvals(%extracted_8, %extracted_9, %extracted_10) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "Toffoli: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %out_qubits, %pos_ctrl_out_qubits:2 = mqtopt.x(static [] mask []) %5 ctrl %1, %3 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
        //CHECK: %out_qubits_11, %pos_ctrl_out_qubits_12:3 = mqtopt.x(static [] mask []) %out_qubits ctrl %7, %pos_ctrl_out_qubits#0, %pos_ctrl_out_qubits#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "Toffoli: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %out_qubits, %out_ctrl_qubits:2 = quantum.custom "Toffoli"() %9 ctrls(%3, %6) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
        //CHECK: %out_qubits_13, %out_ctrl_qubits_14:3 = quantum.custom "PauliX"() %out_qubits ctrls(%12, %out_ctrl_qubits#0, %out_ctrl_qubits#1) ctrlvals(%true_11, %true_11, %true_11) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "Toffoli: MQTOpt to CatalystQuantum")
