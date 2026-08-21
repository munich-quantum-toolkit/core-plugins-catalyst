#!/usr/bin/env -S uv run --script --quiet
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# dependencies = ["nox"]
# ///

"""Nox sessions."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

nox.needs_version = ">=2025.10.16"
nox.options.default_venv_backend = "uv"


PYTHON_ALL_VERSIONS = ["3.12", "3.13", "3.14"]
LLVM_REVISION: str = json.loads(Path("toolchain.json").read_text(encoding="utf-8"))["llvm_revision"]

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True


@contextlib.contextmanager
def preserve_lockfile() -> Generator[None]:
    """Preserve the lockfile by moving it to a temporary directory."""
    lockfile = Path("uv.lock")
    if not lockfile.exists():
        yield
        return
    with tempfile.TemporaryDirectory() as temp_dir_name:
        shutil.move(lockfile, f"{temp_dir_name}/uv.lock")
        try:
            yield
        finally:
            shutil.move(f"{temp_dir_name}/uv.lock", "uv.lock")


@nox.session(reuse_venv=True, default=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    session.run(
        "uv",
        "sync",
        "--frozen",
        "--no-dev",
        "--no-install-project",
        env=env,
        external=True,
    )
    if shutil.which("prek") is None:
        session.install("prek")

    session.run(
        "prek",
        "run",
        "--all-files",
        *session.posargs,
        env=env,
        external=True,
    )


def _bootstrap_environment(session: nox.Session) -> dict[str, str]:
    """Bootstrap the pinned toolchain.

    Returns:
        The environment required to build and test against the toolchain.
    """
    python_executable = Path(session.virtualenv.location) / "bin" / "python"
    session.run(
        "bash",
        "scripts/bootstrap.sh",
        env={"MQT_BOOTSTRAP_PYTHON": str(python_executable)},
        external=True,
    )
    mlir_root = Path.cwd() / ".cache" / "mlir" / LLVM_REVISION
    return {
        "MLIR_DIR": str(mlir_root / "lib" / "cmake" / "mlir"),
        "MQT_MLIR_ROOT": str(mlir_root),
        "UV_PROJECT_ENVIRONMENT": session.virtualenv.location,
    }


def _run_tests(
    session: nox.Session,
    *,
    install_args: Sequence[str] = (),
    extra_command: Sequence[str] = (),
    pytest_run_args: Sequence[str] = (),
) -> None:
    env = _bootstrap_environment(session)
    if shutil.which("cmake") is None and shutil.which("cmake3") is None:
        session.install("cmake")
    if shutil.which("ninja") is None:
        session.install("ninja")

    # install build and test dependencies on top of the existing environment
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--only-group",
        "build",
        "--only-group",
        "test",
        "--no-install-package",
        "pennylane-catalyst",
        *install_args,
        env=env,
    )
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--no-dev",  # do not auto-install dev dependencies
        "--no-build-isolation-package",
        "mqt-core-plugins-catalyst",  # build the project without isolation
        "--no-install-package",
        "pennylane-catalyst",
        *install_args,
        env=env,
    )
    if extra_command:
        session.run(*extra_command, env=env)
    session.run(
        "uv",
        "run",
        "--no-sync",  # do not sync as everything is already installed
        *install_args,
        "pytest",
        *pytest_run_args,
        *session.posargs,
        "--cov-config=pyproject.toml",
        env=env,
    )
    session.run(
        "uv",
        "run",
        "--no-sync",  # do not sync as everything is already installed
        "lit",
        "-sv",
        "test/Conversion",
        env=env,
    )


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True, default=True)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    _run_tests(session)


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True, venv_backend="uv", default=True)
def minimums(session: nox.Session) -> None:
    """Test the minimum versions of dependencies."""
    with preserve_lockfile():
        _run_tests(
            session,
            install_args=["--resolution=lowest-direct"],
            pytest_run_args=["-Wdefault"],
        )
        env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
        session.run("uv", "tree", "--frozen", env=env)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Use "--non-interactive" to avoid serving. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    if serve:
        session.install("sphinx-autobuild")

    env = _bootstrap_environment(session)
    # install build and docs dependencies on top of the existing environment
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--only-group",
        "build",
        "--only-group",
        "docs",
        "--no-install-package",
        "pennylane-catalyst",
        env=env,
    )
    session.run(
        "uv",
        "sync",
        "--inexact",
        "--no-dev",
        "--no-build-isolation-package",
        "mqt-core-plugins-catalyst",
        "--no-install-package",
        "pennylane-catalyst",
        env=env,
    )

    shared_args = [
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        f"docs/_build/{args.builder}",
        *posargs,
    ]

    session.run(
        "uv",
        "run",
        "--no-sync",
        "--no-dev",  # do not auto-install dev dependencies
        "--no-build-isolation-package",
        "mqt-core-plugins-catalyst",  # build the project without isolation
        "sphinx-autobuild" if serve else "sphinx-build",
        *shared_args,
        env=env,
    )


if __name__ == "__main__":
    nox.main()
