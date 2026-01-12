# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _get_filecheck() -> str | None:
    """Locate the FileCheck binary. Caches the result for future calls.

    Returns:
        The path to the FileCheck binary, or None if not found.
    """
    return _find_filecheck()


def _find_filecheck() -> str | None:
    candidates: list[Path] = []

    # 1. Explicit override (if CI ever sets it)
    if os.environ.get("FILECHECK_PATH"):
        candidates.append(Path(os.environ["FILECHECK_PATH"]))

    # 2. LLVM install prefix (best signal)
    if os.environ.get("LLVM_INSTALL_PREFIX"):
        candidates.append(Path(os.environ["LLVM_INSTALL_PREFIX"]) / "bin" / "FileCheck")

    # 3. LLVM_ROOT (common in setup-mlir)
    if os.environ.get("LLVM_ROOT"):
        candidates.append(Path(os.environ["LLVM_ROOT"]) / "bin" / "FileCheck")

    # 4. CMake-style LLVM_DIR
    if os.environ.get("LLVM_DIR"):
        candidates.append(Path(os.environ["LLVM_DIR"]) / ".." / ".." / ".." / "bin" / "FileCheck")

    # 5. CMake-style MLIR_DIR
    if os.environ.get("MLIR_DIR"):
        candidates.append(Path(os.environ["MLIR_DIR"]) / ".." / ".." / ".." / "bin" / "FileCheck")

    # 6. PATH (last resort)
    path_hit = shutil.which("FileCheck")
    if path_hit:
        candidates.append(Path(path_hit))

    for c in candidates:
        if c.exists() and c.is_file():
            return str(c.resolve())

    return None


def _run_filecheck(
    mlir_content: str,
    check_patterns: str,
    test_name: str = "test",
) -> None:
    """Run FileCheck on MLIR content using CHECK patterns from a string.

    Args:
        mlir_content: The MLIR output to verify
        check_patterns: String containing FileCheck directives (lines starting with // CHECK)
        test_name: Name of the test (for error messages)

    Raises:
        RuntimeError: If FileCheck is not found
        AssertionError: If FileCheck validation fails
    """
    filecheck = _get_filecheck()

    if filecheck is None:
        msg = (
            "FileCheck not found.\n"
            "Tried FILECHECK_PATH, LLVM_INSTALL_PREFIX, LLVM_ROOT, "
            "LLVM_DIR, MLIR_DIR, and PATH.\n"
            "Ensure LLVM is available or disable FileCheck-based tests."
        )
        raise RuntimeError(msg)

    with tempfile.NamedTemporaryFile(
        encoding="utf-8",
        mode="w",
        suffix=".mlir",
        delete=False,
    ) as check_file:
        check_file.write(check_patterns)
        check_file_path = check_file.name

    try:
        result = subprocess.run(  # noqa: S603
            [filecheck, check_file_path, "--allow-unused-prefixes"],
            input=mlir_content.encode(),
            capture_output=True,
            check=False,
            timeout=30,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace") if result.stderr else "Unknown error"
            msg = (
                f"FileCheck failed for {test_name}:\n"
                f"{stderr}\n\n"
                f"MLIR Output (first 2000 chars):\n"
                f"{mlir_content[:2000]}..."
            )
            raise AssertionError(msg)
    finally:
        Path(check_file_path).unlink()


def test_paulix_roundtrip() -> None:
    """Test roundtrip conversion of the PauliX gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
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

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    # Verify original CatalystQuantum
    check_mlir_before = """
      //CHECK: %[[Q0_1:.*]] = quantum.custom "PauliX"() %[[Q0_0:.*]] : !quantum.bit
      //CHECK: %[[Q0_2:.*]] = quantum.custom "PauliX"() %[[Q0_1:.*]] : !quantum.bit
      //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CNOT"() %[[Q1_0:.*]], %[[Q0_1:.*]] : !quantum.bit, !quantum.bit
      //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CNOT"() %[[Q10_0:.*]]#0, %[[Q10_0:.*]]#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PauliX: CatalystQuantum")

    # Verify CatalystQuantum → MQTOpt conversion
    check_after_mqtopt = """
      //CHECK: %[[Q0_1:.*]] = mqtopt.x(static [] mask []) %[[Q0_0:.*]] : !mqtopt.Qubit
      //CHECK: %[[Q0_2:.*]] = mqtopt.x(static [] mask []) %[[Q0_1:.*]] : !mqtopt.Qubit
      //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = mqtopt.x(static [] mask []) %[[Q0_2:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
      //CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = mqtopt.x(static [] mask []) %[[Q0_3:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PauliX: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "PauliX"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]] = quantum.custom "PauliX"() %[[Q0_1:.*]] : !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = quantum.custom "CNOT"() %[[Q0_2:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = quantum.custom "CNOT"() %[[Q0_3:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%true_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PauliX: MQTOpt to CatalystQuantum")


def test_pauliy_roundtrip() -> None:
    """Test roundtrip conversion of the PauliY gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
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

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    # Verify original CatalystQuantum
    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "PauliY"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]] = quantum.custom "PauliY"() %[[Q0_1:.*]] : !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CY"() %[[Q1_0:.*]], %[[Q0_2:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CY"() %[[Q10_0:.*]]#0, %[[Q10_0:.*]]#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PauliY: CatalystQuantum")

    # Verify CatalystQuantum → MQTOpt conversion
    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.y(static [] mask []) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]] = mqtopt.y(static [] mask []) %[[Q0_1:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = mqtopt.y(static [] mask []) %[[Q0_2:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = mqtopt.y(static [] mask []) %[[Q0_3:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PauliY: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "PauliY"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]] = quantum.custom "PauliY"() %[[Q0_1:.*]] : !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = quantum.custom "CY"() %[[Q0_2:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = quantum.custom "CY"() %[[Q0_3:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%true_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PauliY: MQTOpt to CatalystQuantum")


def test_pauliz_roundtrip() -> None:
    """Test roundtrip conversion of the PauliZ gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
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

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    # Verify original CatalystQuantum
    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "PauliZ"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]] = quantum.custom "PauliZ"() %[[Q0_1:.*]] : !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CZ"() %[[Q1_0:.*]], %[[Q0_2:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CZ"() %[[Q10_0:.*]]#0, %[[Q10_0:.*]]#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PauliZ: CatalystQuantum")

    # Verify CatalystQuantum → MQTOpt conversion
    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.z(static [] mask []) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]] = mqtopt.z(static [] mask []) %[[Q0_1:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = mqtopt.z(static [] mask []) %[[Q0_2:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = mqtopt.z(static [] mask []) %[[Q0_3:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PauliZ: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "PauliZ"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]] = quantum.custom "PauliZ"() %[[Q0_1:.*]] : !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = quantum.custom "CZ"() %[[Q0_2:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_4:.*]], %[[Q1_2:.*]] = quantum.custom "CZ"() %[[Q0_3:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%true_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PauliZ: MQTOpt to CatalystQuantum")


def test_hadamard_roundtrip() -> None:
    """Test roundtrip conversion of the Hadamard gate.

    Raises:
            FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.Hadamard(wires=0)
        qml.ctrl(qml.Hadamard(wires=0), control=1)
        qml.CH(wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "Hadamard"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = quantum.custom "Hadamard"() %[[Q0_1:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%extracted_5) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = quantum.custom "Hadamard"() %[[Q0_2:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%extracted_7) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "Hadamard: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.h(static [] mask []) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = mqtopt.h(static [] mask []) %[[Q0_1:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = mqtopt.h(static [] mask []) %[[Q0_2:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "Hadamard: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "Hadamard"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = quantum.custom "Hadamard"() %[[Q0_1:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = quantum.custom "Hadamard"() %[[Q0_2:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "Hadamard: MQTOpt to CatalystQuantum")


def test_s_gate_roundtrip() -> None:
    """Test roundtrip conversion of the S gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.S(wires=0)
        qml.adjoint(qml.S(wires=0))
        qml.ctrl(qml.S(wires=0), control=1)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
      //CHECK: %[[Q0_1:.*]] = quantum.custom "S"() %[[Q0_0:.*]] : !quantum.bit
      //CHECK: %[[Q0_2:.*]] = quantum.custom "S"() %[[Q0_1:.*]] adj : !quantum.bit
      //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = quantum.custom "S"() %[[Q0_2:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%extracted_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "S: CatalystQuantum")

    check_after_mqtopt = """
      //CHECK: %[[Q0_1:.*]] = mqtopt.s(static [] mask []) %[[Q0_0:.*]] : !mqtopt.Qubit
      //CHECK: %[[Q0_2:.*]] = mqtopt.sdg(static [] mask []) %[[Q0_1:.*]] : !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "S: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
      //CHECK: %[[Q0_1:.*]] = quantum.custom "S"() %[[Q0_0:.*]] : !quantum.bit
      //CHECK: %[[Q0_2:.*]] = quantum.custom "S"() %[[Q0_1:.*]] adj : !quantum.bit
      //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = quantum.custom "S"() %[[Q0_2:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "S: MQTOpt to CatalystQuantum")


def test_t_gate_roundtrip() -> None:
    """Test roundtrip conversion of the T gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.T(wires=0)
        qml.adjoint(qml.T(wires=0))
        qml.ctrl(qml.T(wires=0), control=1)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "T"() %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]] = quantum.custom "T"() %[[Q0_1:.*]] adj : !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = quantum.custom "T"() %[[Q0_2:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%extracted_6) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "T: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.t(static [] mask []) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]] = mqtopt.tdg(static [] mask []) %[[Q0_1:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = mqtopt.t(static [] mask []) %[[Q0_2:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "T: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
      //CHECK: %[[Q0_1:.*]] = quantum.custom "T"() %[[Q0_0:.*]] : !quantum.bit
      //CHECK: %[[Q0_2:.*]] = quantum.custom "T"() %[[Q0_1:.*]] adj : !quantum.bit
      //CHECK: %[[Q0_3:.*]], %[[Q1_1:.*]] = quantum.custom "T"() %[[Q0_2:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "T: MQTOpt to CatalystQuantum")


def test_rx_gate_roundtrip() -> None:
    """Test roundtrip conversion of the RX gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        qml.RX(0.5, wires=0)
        qml.ctrl(qml.RX(0.5, wires=0), control=1)
        qml.CRX(0.5, wires=[1, 0])
        qml.ctrl(qml.CRX(0.5, wires=[1, 0]), control=2)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "RX"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CRX"({{.*}}) %[[Q1_0:.*]], %[[Q0_1:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CRX"(%extracted_7) %[[Q10_0:.*]]#0, %[[Q10_0:.*]]#1 : !quantum.bit, !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q21_0:.*]]:2 = quantum.custom "RX"(%extracted_12) %[[Q10_0:.*]]#1 ctrls(%3, %[[Q10_0:.*]]#0) ctrlvals(%extracted_13, %extracted_14) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "RX: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.rx({{.*}}) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = mqtopt.rx({{.*}}) %[[Q0_1:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = mqtopt.rx({{.*}}) %[[Q0_2:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_4:.*]], %[[Q12:.*]]:2 = mqtopt.rx(%[[THETA:.*]] static [] mask [false]) %[[Q0_3:.*]] ctrl %[[Q1_2:.*]], %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit"""
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "RX: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "RX"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = quantum.custom "CRX"({{.*}}) %[[Q0_1:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals([[TRUE0:.*]]) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = quantum.custom "CRX"(%extracted_7) %[[Q0_2:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals([[TRUE0:.*]]) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_4:.*]], %[[Q12_3:.*]]:2 = quantum.custom "RX"(%[[THETA:.*]]) %[[Q0_3:.*]] ctrls(%[[Q1_2:.*]], %[[Q1_1:.*]]) ctrlvals(%[[TRUE0:.*]], %[[TRUE1:.*]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "RX: MQTOpt to CatalystQuantum")


def test_ry_gate_roundtrip() -> None:
    """Test roundtrip conversion of the RY gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        qml.RY(0.5, wires=0)
        qml.ctrl(qml.RY(0.5, wires=0), control=1)
        qml.CRY(0.5, wires=[1, 0])
        qml.ctrl(qml.CRY(0.5, wires=[1, 0]), control=2)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "RY"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CRY"({{.*}}) %[[Q1_0:.*]], %[[Q0_1:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CRY"(%extracted_7) %[[Q10_0:.*]]#0, %[[Q10_0:.*]]#1 : !quantum.bit, !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q21_0:.*]]:2 = quantum.custom "RY"(%extracted_12) %[[Q10_0:.*]]#1 ctrls(%3, %[[Q10_0:.*]]#0) ctrlvals(%extracted_13, %extracted_14) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "RY: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.ry({{.*}}) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = mqtopt.ry({{.*}}) %[[Q0_1:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = mqtopt.ry(%extracted_7 static [] mask [false]) %[[Q0_2:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_4:.*]], %[[Q12:.*]]:2 = mqtopt.ry(%[[THETA:.*]] static [] mask [false]) %[[Q0_3:.*]] ctrl %[[Q1_2:.*]], %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "RY: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "RY"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = quantum.custom "CRY"({{.*}}) %[[Q0_1:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = quantum.custom "CRY"(%extracted_7) %[[Q0_2:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_4:.*]], %[[Q12_3:.*]]:2 = quantum.custom "RY"(%[[THETA:.*]]) %[[Q0_3:.*]] ctrls(%[[Q1_2:.*]], %[[Q1_1:.*]]) ctrlvals(%[[TRUE0:.*]], %[[TRUE1:.*]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "RY: MQTOpt to CatalystQuantum")


def test_rz_gate_roundtrip() -> None:
    """Test roundtrip conversion of the RZ gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        qml.RZ(0.5, wires=0)
        qml.ctrl(qml.RZ(0.5, wires=0), control=1)
        qml.CRZ(0.5, wires=[1, 0])
        qml.ctrl(qml.CRZ(0.5, wires=[1, 0]), control=2)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "RZ"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CRZ"({{.*}}) %[[Q1_0:.*]], %[[Q0_1:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "CRZ"(%extracted_7) %[[Q10_0:.*]]#0, %[[Q10_0:.*]]#1 : !quantum.bit, !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q21_0:.*]]:2 = quantum.custom "RZ"(%extracted_12) %[[Q10_0:.*]]#1 ctrls(%3, %[[Q10_0:.*]]#0) ctrlvals(%extracted_13, %extracted_14) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "RZ: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.rz({{.*}}) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = mqtopt.rz({{.*}}) %[[Q0_1:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = mqtopt.rz(%extracted_7 static [] mask [false]) %[[Q0_2:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_4:.*]], %[[Q12:.*]]:2 = mqtopt.rz(%[[THETA:.*]] static [] mask [false]) %[[Q0_3:.*]] ctrl %[[Q1_2:.*]], %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "RZ: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "RZ"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = quantum.custom "CRZ"({{.*}}) %[[Q0_1:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = quantum.custom "CRZ"(%extracted_7) %[[Q0_2:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_4:.*]], %[[Q12_3:.*]]:2 = quantum.custom "RZ"(%[[THETA:.*]]) %[[Q0_3:.*]] ctrls(%[[Q1_2:.*]], %[[Q1_1:.*]]) ctrlvals(%[[TRUE0:.*]], %[[TRUE1:.*]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "RZ: MQTOpt to CatalystQuantum")


def test_phaseshift_gate_roundtrip() -> None:
    """Test roundtrip conversion of the PhaseShift gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.PhaseShift(0.5, wires=0)
        qml.ctrl(qml.PhaseShift(0.5, wires=0), control=1)
        qml.ControlledPhaseShift(0.5, wires=[1, 0])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "PhaseShift"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "ControlledPhaseShift"({{.*}}) %[[Q1_0:.*]], %[[Q0_1:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q10_0:.*]]:2 = quantum.custom "ControlledPhaseShift"(%extracted_7) %[[Q10_0:.*]]#0, %[[Q10_0:.*]]#1 : !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "PhaseShift: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q0_1:.*]] = mqtopt.p({{.*}}) %[[Q0_0:.*]] : !mqtopt.Qubit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = mqtopt.p({{.*}}) %[[Q0_1:.*]] ctrl %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = mqtopt.p({{.*}} static [] mask [false]) %[[Q0_2:.*]] ctrl %[[Q1_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "PhaseShift: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %[[Q0_1:.*]] = quantum.custom "PhaseShift"({{.*}}) %[[Q0_0:.*]] : !quantum.bit
        //CHECK: %[[Q0_2:.*]], %[[Q1_1:.*]] = quantum.custom "ControlledPhaseShift"({{.*}}) %[[Q0_1:.*]] ctrls(%[[Q1_0:.*]]) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q0_3:.*]], %[[Q1_2:.*]] = quantum.custom "ControlledPhaseShift"({{.*}}) %[[Q0_2:.*]] ctrls(%[[Q1_1:.*]]) ctrlvals(%true_8) : !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "PhaseShift: MQTOpt to CatalystQuantum")


def test_swap_gate_roundtrip() -> None:
    """Test roundtrip conversion of the SWAP gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found

    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        qml.SWAP(wires=[0, 1])
        qml.ctrl(qml.SWAP(wires=[0, 1]), control=2)
        qml.CSWAP(wires=[2, 0, 1])

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q01_0:.*]]:2 = quantum.custom "SWAP"() %[[Q0_0:.*]], %[[Q1_0:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q201_0:.*]]:3 = quantum.custom "CSWAP"() %[[Q2_0:.*]], %[[Q01_0:.*]]#0, %[[Q01_0:.*]]#1 : !quantum.bit, !quantum.bit, !quantum.bit
        //CHECK: %[[Q201_0:.*]]:3 = quantum.custom "CSWAP"() %[[Q201_0:.*]]#0, %[[Q201_0:.*]]#1, %[[Q201_0:.*]]#2 : !quantum.bit, !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "SWAP: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q01_1:.*]]:2 = mqtopt.swap(static [] mask []) %[[Q0_0:.*]], %[[Q1_0:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
        //CHECK: %[[Q01_2:.*]]:2, %[[Q2_1:.*]] = mqtopt.swap(static [] mask []) %[[Q01_1:.*]]#0, %[[Q01_1:.*]]#1 ctrl %[[Q2_0:.*]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        //CHECK: %[[Q01_3:.*]]:2, %[[Q2_2:.*]] = mqtopt.swap(static [] mask []) %[[Q01_2:.*]]#0, %[[Q01_2:.*]]#1 ctrl %[[Q2_1:.*]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "SWAP: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %[[Q01_1:.*]]:2 = quantum.custom "SWAP"() %[[Q0_0:.*]], %[[Q1_0:.*]] : !quantum.bit, !quantum.bit
        //CHECK: %[[Q01_2:.*]]:2, %[[Q2_1:.*]] = quantum.custom "CSWAP"() %[[Q01_1:.*]]#0, %[[Q01_1:.*]]#1 ctrls(%[[Q2_0:.*]]) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
        //CHECK: %[[Q01_3:.*]]:2, %[[Q2_2:.*]] = quantum.custom "CSWAP"() %[[Q01_2:.*]]#0, %[[Q01_2:.*]]#1 ctrls(%[[Q2_1:.*]]) ctrlvals(%true_7) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "SWAP: MQTOpt to CatalystQuantum")


def test_toffoli_gate_roundtrip() -> None:
    """Test roundtrip conversion of the Toffoli gate.

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=4))
    def circuit() -> None:
        qml.Toffoli(wires=[0, 1, 2])
        qml.ctrl(qml.Toffoli(wires=[0, 1, 2]), control=3)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
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

    mlir_before = Path(catalyst_mlir).read_text(encoding="utf-8")
    mlir_after_mqtopt = Path(mlir_to_mqtopt).read_text(encoding="utf-8")
    mlir_after_roundtrip = Path(mlir_to_catalyst).read_text(encoding="utf-8")

    check_mlir_before = """
        //CHECK: %[[Q012_0:.*]]:3 = quantum.custom "Toffoli"() %[[Q0_0:.*]], %[[Q1_0:.*]], %[[Q2_0:.*]] : !quantum.bit, !quantum.bit, !quantum.bit
        //CHECK: %[[Q2_1:.*]], %[[Q301_0:.*]]:3 = quantum.custom "PauliX"() %[[Q012_0:.*]]#2 ctrls(%[[Q3_0:.*]], %[[Q012_0:.*]]#0, %[[Q012_0:.*]]#1) ctrlvals(%extracted_9, %extracted_10, %extracted_11) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_before, check_mlir_before, "Toffoli: CatalystQuantum")

    check_after_mqtopt = """
        //CHECK: %[[Q2_1:.*]], %[[Q01_1:.*]]:2 = mqtopt.x(static [] mask []) %[[Q2_0:.*]] ctrl %[[Q0_0:.*]], %[[Q1_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
        //CHECK: %[[Q2_2:.*]], %[[Q301_1:.*]]:3 = mqtopt.x(static [] mask []) %[[Q2_1:.*]] ctrl %[[Q3_0:.*]], %[[Q01_1:.*]]#0, %[[Q01_1:.*]]#1 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit
    """
    _run_filecheck(mlir_after_mqtopt, check_after_mqtopt, "Toffoli: CatalystQuantum to MQTOpt")

    check_after_catalyst = """
        //CHECK: %[[Q2_1:.*]], %[[Q01_1:.*]]:2 = quantum.custom "Toffoli"() %[[Q2_0:.*]] ctrls(%[[Q0_0:.*]], %[[Q1_0:.*]]) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit
        //CHECK: %[[Q2_2:.*]], %[[Q301_1:.*]]:3 = quantum.custom "PauliX"() %[[Q2_1:.*]] ctrls(%[[Q3_0:.*]], %[[Q01_1:.*]]#0, %[[Q01_1:.*]]#1) ctrlvals(%true_12, %true_12, %true_12) : !quantum.bit ctrls !quantum.bit, !quantum.bit, !quantum.bit
    """
    _run_filecheck(mlir_after_roundtrip, check_after_catalyst, "Toffoli: MQTOpt to CatalystQuantum")
