# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for the CatalystQuantum, QCO, and QC conversion passes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pennylane as qml
import pytest
from catalyst.passes import apply_pass

from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path, get_device

if TYPE_CHECKING:
    from collections.abc import Sequence


DIRECT_PIPELINE = [
    ("Init", ["builtin.module(canonicalize)"]),
    ("ToQCO", ["builtin.module(catalystquantum-to-qco)"]),
    ("ToCatalystQuantum", ["builtin.module(qco-to-catalystquantum)"]),
]

QC_PIPELINE = [
    ("Init", ["builtin.module(canonicalize)"]),
    ("ToQCO", ["builtin.module(catalystquantum-to-qco)"]),
    ("ToQC", ["builtin.module(qco-to-qc)"]),
    ("BackToQCO", ["builtin.module(qc-to-qco)"]),
    ("ToCatalystQuantum", ["builtin.module(qco-to-catalystquantum)"]),
]

MQT_PLUGIN_PATH = get_catalyst_plugin_abs_path()


@pytest.fixture
def _isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Catalyst intermediate files outside the source tree."""
    monkeypatch.chdir(tmp_path)


def _read_stages(pipeline: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
    """Read the Catalyst pipeline stages.

    Returns:
        The contents of the requested pipeline-stage files.

    Raises:
        FileNotFoundError: If a requested stage file is missing.
        RuntimeError: If Catalyst did not create exactly one workspace.
    """
    workspaces = [path for path in Path.cwd().glob("module*") if path.is_dir()]
    if len(workspaces) != 1:
        msg = f"Expected one Catalyst workspace, found: {[path.name for path in workspaces]}"
        raise RuntimeError(msg)
    paths = [workspaces[0] / f"{index}_After{name}.mlir" for index, (name, _) in enumerate(pipeline, start=1)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        msg = f"Missing Catalyst intermediates: {missing}"
        raise FileNotFoundError(msg)
    return [path.read_text(encoding="utf-8") for path in paths]


@pytest.mark.usefixtures("_isolated_working_directory")
def test_direct_qco_round_trip() -> None:
    """Execute and inspect CatalystQuantum to QCO to CatalystQuantum."""

    @apply_pass("mqt.qco-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-qco")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> qml.measurements.ExpectationMP:
        qml.Hadamard(wires=0)
        qml.RX(0.25, wires=0)
        qml.ctrl(qml.PauliX, control=1, control_values=[False])(wires=0)
        qml.adjoint(qml.S)(wires=0)
        qml.SWAP(wires=[0, 2])
        return qml.expval(qml.PauliZ(wires=0))  # ty: ignore[invalid-argument-type]

    @qml.qjit(
        target="mlir",
        pipelines=DIRECT_PIPELINE,
        keep_intermediate=2,
        pass_plugins={MQT_PLUGIN_PATH},
        dialect_plugins={MQT_PLUGIN_PATH},
    )
    def module() -> Any:  # ruff: ignore[any-type]
        return circuit()

    assert module.mlir_opt
    catalyst, qco, round_trip = _read_stages(DIRECT_PIPELINE)

    assert "quantum.alloc" in catalyst
    assert "quantum.device" in catalyst
    assert 'qco.alloc("qreg0", 3, 0)' in qco
    assert 'qco.alloc("qreg0", 3, 1)' in qco
    assert 'qco.alloc("qreg0", 3, 2)' in qco
    assert "qco.h" in qco
    assert "qco.rx" in qco
    assert "qco.ctrl" in qco
    assert "catalyst.control_values = array<i1: false>" in qco
    assert "qco.inv" in qco
    assert "qco.swap" in qco
    assert "func.func private @__mqt_catalyst_qco_qubit_bridge(!qco.qubit)" in qco
    assert "attributes {catalyst.qco_qubit_bridge}" in qco
    assert "call @__mqt_catalyst_qco_qubit_bridge" in qco
    assert "builtin.unrealized_conversion_cast" not in qco
    assert "quantum.namedobs" in qco
    assert "quantum.expval" in qco
    assert "quantum.alloc" in round_trip
    assert "quantum.device" in round_trip
    assert "quantum.namedobs" in round_trip
    assert "quantum.expval" in round_trip
    assert 'quantum.custom "Hadamard"' in round_trip
    assert 'quantum.custom "PauliX"' in round_trip
    assert "ctrlvals(" in round_trip
    assert "arith.constant false" in round_trip
    assert 'quantum.custom "RX"' in round_trip
    assert 'quantum.custom "SWAP"' in round_trip
    assert "__mqt_catalyst_qco_qubit_bridge" not in round_trip
    assert "builtin.unrealized_conversion_cast" not in round_trip
    assert "qco." not in round_trip
    assert "!qco.qubit" not in round_trip


@pytest.mark.usefixtures("_isolated_working_directory")
def test_qc_chained_round_trip() -> None:
    """Execute and inspect the QCO to QC to QCO path."""

    @apply_pass("mqt.qco-to-catalystquantum")
    @apply_pass("mqt.qc-to-qco")
    @apply_pass("mqt.qco-to-qc")
    @apply_pass("mqt.catalystquantum-to-qco")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> qml.measurements.ExpectationMP:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(0.5, wires=1)
        return qml.expval(qml.PauliZ(wires=1))  # ty: ignore[invalid-argument-type]

    @qml.qjit(
        target="mlir",
        pipelines=QC_PIPELINE,
        keep_intermediate=2,
        pass_plugins={MQT_PLUGIN_PATH},
        dialect_plugins={MQT_PLUGIN_PATH},
    )
    def module() -> Any:  # ruff: ignore[any-type]
        return circuit()

    assert module.mlir_opt
    catalyst, first_qco, qc, second_qco, round_trip = _read_stages(QC_PIPELINE)

    assert "quantum.alloc" in catalyst
    assert "qco.alloc" in first_qco
    assert "qco.ctrl" in first_qco
    assert "qc.alloc" in qc
    assert "qc.h" in qc
    assert "qc.ctrl" in qc
    assert "quantum.namedobs" in qc
    assert "quantum.expval" in qc
    assert "func.func private @__mqt_catalyst_qco_qubit_bridge(!qc.qubit)" in qc
    assert "attributes {catalyst.qco_qubit_bridge}" in qc
    assert "call @__mqt_catalyst_qco_qubit_bridge" in qc
    assert "qco.alloc" in second_qco
    assert "qco.h" in second_qco
    assert "qco.ctrl" in second_qco
    assert "quantum.namedobs" in second_qco
    assert "quantum.expval" in second_qco
    assert "func.func private @__mqt_catalyst_qco_qubit_bridge(!qco.qubit)" in second_qco
    assert "call @__mqt_catalyst_qco_qubit_bridge" in second_qco
    assert "quantum.alloc" in round_trip
    assert 'quantum.custom "Hadamard"' in round_trip
    assert 'quantum.custom "CNOT"' in round_trip
    assert 'quantum.custom "RZ"' in round_trip
    assert "quantum.namedobs" in round_trip
    assert "quantum.expval" in round_trip
    assert "__mqt_catalyst_qco_qubit_bridge" not in round_trip
    assert "builtin.unrealized_conversion_cast" not in round_trip
    assert "qco." not in round_trip
    assert "qc." not in round_trip
