/*
 * Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/IR/DialectRegistry.h"

namespace test {
// Catalyst's Python module expects this symbol to have external linkage.
// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerTestDialect([[maybe_unused]] mlir::DialectRegistry& registry) {}
} // namespace test
