/*
 * Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CatalystQuantumToQC/CatalystQuantumToQC.h"
// #include "mlir/Conversion/MQTOptToCatalystQuantum/MQTOptToCatalystQuantum.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <llvm/Config/llvm-config.h>
#include <llvm/Support/Compiler.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>
#include <mlir/Tools/Plugins/PassPlugin.h>

using namespace mlir;

/// Dialect plugin registration mechanism.
/// Observe that it also allows registering passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {.apiVersion = MLIR_PLUGIN_API_VERSION,
          .pluginName = "QC",
          .pluginVersion = LLVM_VERSION_STRING,
          .registerDialectRegistryCallbacks = [](DialectRegistry* registry) {
            registry->insert<::mlir::qc>();
          }};
}

/// The pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {.apiVersion = MLIR_PLUGIN_API_VERSION,
          .pluginName = "QCPasses",
          .pluginVersion = LLVM_VERSION_STRING,
          .registerPassRegistryCallbacks = []() {
            // Only register the conversion passes we implement
            // Note: mqt::ir::opt::registerMQTOptPasses() is not called to avoid
            // pulling in transpilation transforms that require LLVM 21
            mlir::registerCatalystQuantumToQCPasses();
            // mqt::ir::conversions::registerMQTOptToCatalystQuantumPasses();
          }};
}
