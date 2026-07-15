# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list
of changes including minor and patch releases, please refer to the
[changelog](CHANGELOG.md).

## [Unreleased]

## [1.1.0]

### Explicit Catalyst plugin loading

Catalyst 0.15.0 migrated `apply_pass()` to PennyLane's standard transform API.
It no longer resolves the `catalyst.passes_resolution` entry point via
`name2pass()`.

Pass the plugin path explicitly to `qml.qjit()` instead:

```python
import pennylane as qml

from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path

plugin_path = str(get_catalyst_plugin_abs_path())


@qml.qjit(pass_plugins={plugin_path}, dialect_plugins={plugin_path})
def module():
    return circuit()
```

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/compare/v1.0.1...v1.1.0
