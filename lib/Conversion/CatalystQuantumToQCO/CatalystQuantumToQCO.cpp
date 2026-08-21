/*
 * Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CatalystQuantumToQCO/CatalystQuantumToQCO.h" // NOLINT(misc-include-cleaner)

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <Quantum/IR/QuantumOps.h>
#include <Quantum/IR/QuantumTypes.h>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numbers>
#include <optional>
#include <string>
#include <utility>

namespace mqt::ir::conversions {

#define GEN_PASS_DEF_CATALYSTQUANTUMTOQCO
#include "mlir/Conversion/CatalystQuantumToQCO/CatalystQuantumToQCO.h.inc"

using namespace mlir;

namespace {

constexpr llvm::StringLiteral CONTROL_VALUES_ATTR = "catalyst.control_values";
constexpr llvm::StringLiteral GATE_NAME_ATTR = "catalyst.gate_name";
constexpr llvm::StringLiteral NATIVE_CONTROL_COUNT_ATTR =
    "catalyst.native_control_count";
constexpr llvm::StringLiteral NEGATIVE_CONTROL_WRAPPER_ATTR =
    "catalyst.negative_control_wrapper";
constexpr llvm::StringLiteral QUBIT_BRIDGE_ATTR = "catalyst.qco_qubit_bridge";
constexpr llvm::StringLiteral QUBIT_BRIDGE_SYMBOL =
    "__mqt_catalyst_qco_qubit_bridge";
constexpr llvm::StringLiteral GATE_HINT_BRIDGE_ATTR =
    "catalyst.qco_gate_hint_bridge";
constexpr llvm::StringLiteral GATE_HINT_BRIDGE_SYMBOL =
    "__mqt_catalyst_qco_gate_hint_bridge";
constexpr llvm::StringLiteral REGISTER_NAME_ATTR = "mqt.qco_register_name";
constexpr llvm::StringLiteral MEASURE_REGISTER_NAME_ATTR =
    "mqt.qco_measure_register_name";
constexpr llvm::StringLiteral MEASURE_REGISTER_SIZE_ATTR =
    "mqt.qco_measure_register_size";
constexpr llvm::StringLiteral MEASURE_REGISTER_INDEX_ATTR =
    "mqt.qco_measure_register_index";

struct GateSpec {
  constexpr GateSpec(const llvm::StringRef qcoName, const size_t numTargets,
                     const size_t numNativeControls, const size_t numParameters,
                     const bool appendIsingXYBeta = false,
                     const bool variadicTargets = false)
      : qcoName(qcoName), numTargets(numTargets),
        numNativeControls(numNativeControls), numParameters(numParameters),
        appendIsingXYBeta(appendIsingXYBeta), variadicTargets(variadicTargets) {
  }

  llvm::StringRef qcoName;
  size_t numTargets;
  size_t numNativeControls;
  size_t numParameters;
  bool appendIsingXYBeta = false;
  bool variadicTargets = false;
};

std::optional<GateSpec> lookupGate(const llvm::StringRef gateName) {
  if (gateName == "Hadamard") {
    return GateSpec{"qco.h", 1, 0, 0};
  }
  if (gateName == "Identity") {
    return GateSpec{"qco.id", 1, 0, 0};
  }
  if (gateName == "PauliX") {
    return GateSpec{"qco.x", 1, 0, 0};
  }
  if (gateName == "PauliY") {
    return GateSpec{"qco.y", 1, 0, 0};
  }
  if (gateName == "PauliZ") {
    return GateSpec{"qco.z", 1, 0, 0};
  }
  if (gateName == "S") {
    return GateSpec{"qco.s", 1, 0, 0};
  }
  if (gateName == "T") {
    return GateSpec{"qco.t", 1, 0, 0};
  }
  if (gateName == "SX") {
    return GateSpec{"qco.sx", 1, 0, 0};
  }
  if (gateName == "ECR") {
    return GateSpec{"qco.ecr", 2, 0, 0};
  }
  if (gateName == "SWAP") {
    return GateSpec{"qco.swap", 2, 0, 0};
  }
  if (gateName == "ISWAP") {
    return GateSpec{"qco.iswap", 2, 0, 0};
  }
  if (gateName == "RX") {
    return GateSpec{"qco.rx", 1, 0, 1};
  }
  if (gateName == "RY") {
    return GateSpec{"qco.ry", 1, 0, 1};
  }
  if (gateName == "RZ") {
    return GateSpec{"qco.rz", 1, 0, 1};
  }
  if (gateName == "PhaseShift") {
    return GateSpec{"qco.p", 1, 0, 1};
  }
  if (gateName == "CRX") {
    return GateSpec{"qco.rx", 1, 1, 1};
  }
  if (gateName == "CRY") {
    return GateSpec{"qco.ry", 1, 1, 1};
  }
  if (gateName == "CRZ") {
    return GateSpec{"qco.rz", 1, 1, 1};
  }
  if (gateName == "ControlledPhaseShift") {
    return GateSpec{"qco.p", 1, 1, 1};
  }
  if (gateName == "IsingXY") {
    return GateSpec{"qco.xx_plus_yy", 2, 0, 1, true};
  }
  if (gateName == "IsingXX") {
    return GateSpec{"qco.rxx", 2, 0, 1};
  }
  if (gateName == "IsingYY") {
    return GateSpec{"qco.ryy", 2, 0, 1};
  }
  if (gateName == "IsingZZ") {
    return GateSpec{"qco.rzz", 2, 0, 1};
  }
  if (gateName == "CNOT") {
    return GateSpec{"qco.x", 1, 1, 0};
  }
  if (gateName == "CY") {
    return GateSpec{"qco.y", 1, 1, 0};
  }
  if (gateName == "CZ") {
    return GateSpec{"qco.z", 1, 1, 0};
  }
  if (gateName == "Toffoli") {
    return GateSpec{"qco.x", 1, 2, 0};
  }
  if (gateName == "CSWAP") {
    return GateSpec{"qco.swap", 2, 1, 0};
  }
  if (gateName == "Barrier") {
    return GateSpec{"qco.barrier", 0, 0, 0, false, true};
  }
  return std::nullopt;
}

bool isCatalystQuantumType(const Type type) {
  return isa<catalyst::quantum::QubitType, catalyst::quantum::QuregType>(type);
}

SmallVector<Value> copyValues(const ValueRange values) {
  return {values.begin(), values.end()};
}

class Converter {
public:
  explicit Converter(ModuleOp module) : module(module), builder(module) {}

  LogicalResult run() {
    if (failed(validateBoundaries())) {
      return failure();
    }

    for (auto function : module.getOps<func::FuncOp>()) {
      if (function.empty()) {
        continue;
      }
      if (failed(convertBlock(function.getBody().front()))) {
        return failure();
      }
    }
    return success();
  }

private:
  LogicalResult validateBoundaries() {
    for (auto function : module.getOps<func::FuncOp>()) {
      for (const Type type : function.getFunctionType().getInputs()) {
        if (isCatalystQuantumType(type)) {
          return function.emitError(
              "quantum values crossing function boundaries are not supported");
        }
      }
      for (const Type type : function.getFunctionType().getResults()) {
        if (isCatalystQuantumType(type)) {
          return function.emitError(
              "quantum values crossing function boundaries are not supported");
        }
      }
      if (!function.empty() && !function.getBody().hasOneBlock()) {
        const WalkResult quantumOperation =
            function.walk([](Operation* op) -> WalkResult {
              return op->getName().getDialectNamespace() == "quantum"
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            });
        if (quantumOperation.wasInterrupted()) {
          return function.emitError(
              "quantum operations in multi-block functions are not supported");
        }
      }
    }

    const WalkResult result = module.walk([&](Operation* op) -> WalkResult {
      const llvm::StringRef name = op->getName().getStringRef();
      if (name == "quantum.operator") {
        op->emitError("quantum.operator must be decomposed before "
                      "catalystquantum-to-qco");
        return WalkResult::interrupt();
      }
      if (name == "quantum.adjoint" || name == "quantum.ctrl" ||
          name == "quantum.yield") {
        op->emitError("quantum values crossing control-flow boundaries are not "
                      "supported");
        return WalkResult::interrupt();
      }

      if (op->getName().getDialectNamespace() == "quantum") {
        auto function = op->getParentOfType<func::FuncOp>();
        if (!function || function.empty() ||
            op->getBlock() != &function.getBody().front()) {
          op->emitError("quantum values crossing control-flow boundaries are "
                        "not supported");
          return WalkResult::interrupt();
        }
      }

      if (op->getName().getDialectNamespace() != "quantum") {
        for (const Value operand : op->getOperands()) {
          if (isCatalystQuantumType(operand.getType())) {
            op->emitError("quantum values crossing control-flow boundaries are "
                          "not supported");
            return WalkResult::interrupt();
          }
        }
        for (const Value resultValue : op->getResults()) {
          if (isCatalystQuantumType(resultValue.getType())) {
            op->emitError("quantum values crossing control-flow boundaries are "
                          "not supported");
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  LogicalResult convertBlock(Block& block) {
    qubits.clear();
    registers.clear();
    registerNames.clear();
    nextRegister = 0;

    for (Operation& operation : block) {
      auto alloc = dyn_cast<catalyst::quantum::AllocOp>(&operation);
      if (!alloc) {
        continue;
      }
      const Attribute preservedName = alloc->getAttr(REGISTER_NAME_ATTR);
      if (!preservedName) {
        continue;
      }
      const auto name = dyn_cast<StringAttr>(preservedName);
      if (!name || name.getValue().empty()) {
        return alloc.emitError(
            "mqt.qco_register_name must be a nonempty string");
      }
      if (!registerNames.insert(name.getValue()).second) {
        return alloc.emitError("duplicate mqt.qco_register_name '")
               << name.getValue() << "'";
      }
    }

    SmallVector<Operation*> convertedOps;

    for (Operation& operation : llvm::make_early_inc_range(block)) {
      builder.setInsertionPoint(&operation);

      const LogicalResult result =
          llvm::TypeSwitch<Operation*, LogicalResult>(&operation)
              .Case<catalyst::quantum::AllocOp>(
                  [&](auto op) { return convertAlloc(op, convertedOps); })
              .Case<catalyst::quantum::AllocQubitOp>(
                  [&](auto op) { return convertAllocQubit(op, convertedOps); })
              .Case<catalyst::quantum::DeallocOp>(
                  [&](auto op) { return convertDealloc(op, convertedOps); })
              .Case<catalyst::quantum::DeallocQubitOp>([&](auto op) {
                return convertDeallocQubit(op, convertedOps);
              })
              .Case<catalyst::quantum::ExtractOp>(
                  [&](auto op) { return convertExtract(op, convertedOps); })
              .Case<catalyst::quantum::InsertOp>(
                  [&](auto op) { return convertInsert(op, convertedOps); })
              .Case<catalyst::quantum::CustomOp>(
                  [&](auto op) { return convertCustom(op, convertedOps); })
              .Case<catalyst::quantum::GlobalPhaseOp>(
                  [&](auto op) { return convertGlobalPhase(op, convertedOps); })
              .Case<catalyst::quantum::MeasureOp>(
                  [&](auto op) { return convertMeasure(op, convertedOps); })
              .Case<catalyst::quantum::ComputationalBasisOp>([&](auto op) {
                return convertComputationalBasis(op, convertedOps);
              })
              .Default([&](Operation* op) { return preserveOperation(op); });

      if (failed(result)) {
        return failure();
      }
    }

    for (Operation* op : llvm::reverse(convertedOps)) {
      if (!op->use_empty()) {
        return op->emitError(
            "unsupported quantum use remains after conversion");
      }
      op->erase();
    }
    return success();
  }

  FailureOr<Value> resolveQubit(const Value value, Operation* user) {
    if (isa<qco::QubitType>(value.getType())) {
      return value;
    }
    const auto iter = qubits.find(value);
    if (iter == qubits.end()) {
      return user->emitError("quantum value is not produced by a supported "
                             "static circuit operation");
    }
    return iter->second;
  }

  FailureOr<SmallVector<Value>> resolveQubits(const ValueRange values,
                                              Operation* user) {
    SmallVector<Value> resolved;
    resolved.reserve(values.size());
    for (const Value value : values) {
      auto mapped = resolveQubit(value, user);
      if (failed(mapped)) {
        return failure();
      }
      resolved.push_back(*mapped);
    }
    return resolved;
  }

  LogicalResult convertAlloc(catalyst::quantum::AllocOp op,
                             SmallVectorImpl<Operation*>& convertedOps) {
    const auto sizeAttr = op.getNqubitsAttrAttr();
    if (!sizeAttr) {
      return op.emitError("dynamic register sizes are not supported");
    }

    const uint64_t size = sizeAttr.getValue().getZExtValue();
    if (size == 0) {
      return op.emitError("zero-sized registers are not supported");
    }
    std::string registerName;
    if (const Attribute preservedName = op->getAttr(REGISTER_NAME_ATTR)) {
      registerName = cast<StringAttr>(preservedName).getValue().str();
    } else {
      do {
        registerName =
            (llvm::Twine("qreg") + llvm::Twine(nextRegister++)).str();
      } while (!registerNames.insert(registerName).second);
    }
    SmallVector<Value> values;
    values.reserve(size);
    for (uint64_t index = 0; index < size; ++index) {
      auto alloc = qco::AllocOp::create(
          builder, op.getLoc(), builder.getStringAttr(registerName),
          builder.getI64IntegerAttr(static_cast<int64_t>(size)),
          builder.getI64IntegerAttr(static_cast<int64_t>(index)));
      values.push_back(alloc.getResult());
    }
    registers[op.getQreg()] = std::move(values);
    convertedOps.push_back(op);
    return success();
  }

  LogicalResult convertAllocQubit(catalyst::quantum::AllocQubitOp op,
                                  SmallVectorImpl<Operation*>& convertedOps) {
    auto alloc = qco::AllocOp::create(builder, op.getLoc());
    qubits[op.getQubit()] = alloc.getResult();
    convertedOps.push_back(op);
    return success();
  }

  LogicalResult convertDealloc(catalyst::quantum::DeallocOp op,
                               SmallVectorImpl<Operation*>& convertedOps) {
    const auto iter = registers.find(op.getQreg());
    if (iter == registers.end()) {
      return op.emitError("register is not produced by a supported static "
                          "allocation");
    }
    for (const Value qubit : iter->second) {
      qco::DeallocOp::create(builder, op.getLoc(), qubit);
    }
    convertedOps.push_back(op);
    return success();
  }

  LogicalResult convertDeallocQubit(catalyst::quantum::DeallocQubitOp op,
                                    SmallVectorImpl<Operation*>& convertedOps) {
    auto qubit = resolveQubit(op.getQubit(), op);
    if (failed(qubit)) {
      return failure();
    }
    qco::DeallocOp::create(builder, op.getLoc(), *qubit);
    convertedOps.push_back(op);
    return success();
  }

  static FailureOr<uint64_t> getStaticIndex(Operation* op, IntegerAttr attr,
                                            const Value dynamicIndex,
                                            const uint64_t registerSize) {
    if (!attr) {
      if (dynamicIndex) {
        return op->emitError("dynamic register indices are not supported");
      }
      return op->emitError("register index is missing");
    }
    const uint64_t index = attr.getValue().getZExtValue();
    if (index >= registerSize) {
      return op->emitError("register index is out of bounds");
    }
    return index;
  }

  LogicalResult convertExtract(catalyst::quantum::ExtractOp op,
                               SmallVectorImpl<Operation*>& convertedOps) {
    const auto iter = registers.find(op.getQreg());
    if (iter == registers.end()) {
      return op.emitError("register is not produced by a supported static "
                          "allocation");
    }
    auto index = getStaticIndex(op, op.getIdxAttrAttr(), op.getIdx(),
                                iter->second.size());
    if (failed(index)) {
      return failure();
    }
    qubits[op.getQubit()] = iter->second[*index];
    convertedOps.push_back(op);
    return success();
  }

  LogicalResult convertInsert(catalyst::quantum::InsertOp op,
                              SmallVectorImpl<Operation*>& convertedOps) {
    const auto iter = registers.find(op.getInQreg());
    if (iter == registers.end()) {
      return op.emitError("register is not produced by a supported static "
                          "allocation");
    }
    auto index = getStaticIndex(op, op.getIdxAttrAttr(), op.getIdx(),
                                iter->second.size());
    if (failed(index)) {
      return failure();
    }
    auto qubit = resolveQubit(op.getQubit(), op);
    if (failed(qubit)) {
      return failure();
    }
    SmallVector<Value> updated = iter->second;
    updated[*index] = *qubit;
    registers[op.getOutQreg()] = std::move(updated);
    convertedOps.push_back(op);
    return success();
  }

  static FailureOr<SmallVector<bool>>
  resolveControlValues(const ValueRange values, Operation* op) {
    SmallVector<bool> resolved;
    resolved.reserve(values.size());
    for (const Value value : values) {
      auto constant = value.getDefiningOp<arith::ConstantOp>();
      if (!constant) {
        return op->emitError("dynamic control values are not supported");
      }
      const auto attr = constant.getValue();
      if (const auto boolAttr = dyn_cast<BoolAttr>(attr)) {
        resolved.push_back(boolAttr.getValue());
      } else if (const auto integerAttr = dyn_cast<IntegerAttr>(attr)) {
        resolved.push_back(!integerAttr.getValue().isZero());
      } else {
        return op->emitError("control values must be boolean constants");
      }
    }
    return resolved;
  }

  SmallVector<Value> createBaseUnitary(const Location loc,
                                       const llvm::StringRef qcoName,
                                       const llvm::StringRef catalystName,
                                       const ValueRange targets,
                                       const ValueRange parameters) {
    OperationState state(loc, qcoName);
    state.addOperands(targets);
    state.addOperands(parameters);
    state.addTypes(targets.getTypes());
    state.addAttribute(GATE_NAME_ATTR, builder.getStringAttr(catalystName));
    Operation* operation = builder.create(state);
    return copyValues(operation->getResults());
  }

  Value createNegativeControlX(const Location loc, const Value input) {
    auto results = createBaseUnitary(loc, "qco.x", "PauliX", ValueRange{input},
                                     ValueRange{});
    Operation* wrapper = results.front().getDefiningOp();
    wrapper->setAttr(NEGATIVE_CONTROL_WRAPPER_ATTR, builder.getUnitAttr());
    return results.front();
  }

  struct ConvertedUnitary {
    SmallVector<Value> controls;
    SmallVector<Value> targets;
  };

  ConvertedUnitary
  createUnitary(const Location loc, const llvm::StringRef qcoName,
                const llvm::StringRef catalystName, SmallVector<Value> controls,
                const ValueRange targets, const ValueRange parameters,
                const ArrayRef<bool> controlValues,
                const size_t nativeControlCount, const bool adjoint) {
    for (size_t index = 0; index < controls.size(); ++index) {
      if (!controlValues[index]) {
        controls[index] = createNegativeControlX(loc, controls[index]);
      }
    }

    auto createBody = [&](const ValueRange bodyTargets) {
      return createBaseUnitary(loc, qcoName, catalystName, bodyTargets,
                               parameters);
    };
    auto createPossiblyInvertedBody = [&](const ValueRange bodyTargets) {
      if (!adjoint) {
        return createBody(bodyTargets);
      }
      auto inverse = qco::InvOp::create(builder, loc, bodyTargets, createBody);
      inverse->setAttr(GATE_NAME_ATTR, builder.getStringAttr(catalystName));
      return copyValues(inverse.getResults());
    };

    SmallVector<Value> controlResults;
    SmallVector<Value> targetResults;
    if (controls.empty()) {
      targetResults = createPossiblyInvertedBody(targets);
    } else {
      func::CallOp::create(
          builder, loc,
          getOrCreateGateHintBridge(loc, catalystName, nativeControlCount),
          ValueRange{});
      auto controlled = qco::CtrlOp::create(builder, loc, controls, targets,
                                            createPossiblyInvertedBody);
      controlled->setAttr(
          CONTROL_VALUES_ATTR,
          DenseBoolArrayAttr::get(builder.getContext(), controlValues));
      controlled->setAttr(GATE_NAME_ATTR, builder.getStringAttr(catalystName));
      controlled->setAttr(
          NATIVE_CONTROL_COUNT_ATTR,
          builder.getI64IntegerAttr(static_cast<int64_t>(nativeControlCount)));
      controlResults = copyValues(controlled.getControlsOut());
      targetResults = copyValues(controlled.getTargetsOut());
    }

    for (size_t index = 0; index < controlResults.size(); ++index) {
      if (!controlValues[index]) {
        controlResults[index] =
            createNegativeControlX(loc, controlResults[index]);
      }
    }
    return {.controls = std::move(controlResults),
            .targets = std::move(targetResults)};
  }

  LogicalResult convertCustom(catalyst::quantum::CustomOp op,
                              SmallVectorImpl<Operation*>& convertedOps) {
    const llvm::StringRef gateName = op.getGateName();
    const auto spec = lookupGate(gateName);
    if (!spec) {
      return op.emitError("unsupported Catalyst gate '") << gateName << "'";
    }

    const size_t numTargets =
        spec->variadicTargets ? op.getInQubits().size() : spec->numTargets;
    if (op.getInQubits().size() != numTargets + spec->numNativeControls) {
      return op.emitError("gate has an unexpected number of qubits");
    }
    if (op.getParams().size() != spec->numParameters) {
      return op.emitError("gate has an unexpected number of parameters");
    }

    auto inputQubits = resolveQubits(op.getInQubits(), op);
    auto additionalControls = resolveQubits(op.getInCtrlQubits(), op);
    auto additionalControlValues =
        resolveControlValues(op.getInCtrlValues(), op);
    if (failed(inputQubits) || failed(additionalControls) ||
        failed(additionalControlValues)) {
      return failure();
    }
    if (additionalControls->size() != additionalControlValues->size()) {
      return op.emitError("control qubits and values must have equal length");
    }

    const ValueRange nativeControls =
        ValueRange(*inputQubits).take_front(spec->numNativeControls);
    SmallVector<Value> controls(nativeControls.begin(), nativeControls.end());
    controls.append(additionalControls->begin(), additionalControls->end());
    const ValueRange targets =
        ValueRange(*inputQubits).drop_front(spec->numNativeControls);

    SmallVector<bool> controlValues(spec->numNativeControls, true);
    controlValues.append(additionalControlValues->begin(),
                         additionalControlValues->end());

    SmallVector<Value> parameters(op.getParams().begin(), op.getParams().end());
    if (spec->appendIsingXYBeta) {
      parameters.push_back(
          arith::ConstantOp::create(builder, op.getLoc(),
                                    builder.getF64FloatAttr(std::numbers::pi))
              .getResult());
    }

    const ConvertedUnitary converted = createUnitary(
        op.getLoc(), spec->qcoName, gateName, std::move(controls), targets,
        parameters, controlValues, spec->numNativeControls, op.getAdjoint());

    if (converted.controls.size() !=
        spec->numNativeControls + additionalControls->size()) {
      return op.emitError("internal control result count mismatch");
    }
    for (size_t index = 0; index < spec->numNativeControls; ++index) {
      qubits[op.getOutQubits()[index]] = converted.controls[index];
    }
    for (size_t index = 0; index < converted.targets.size(); ++index) {
      qubits[op.getOutQubits()[spec->numNativeControls + index]] =
          converted.targets[index];
    }
    for (size_t index = 0; index < additionalControls->size(); ++index) {
      qubits[op.getOutCtrlQubits()[index]] =
          converted.controls[spec->numNativeControls + index];
    }

    convertedOps.push_back(op);
    return success();
  }

  LogicalResult convertGlobalPhase(catalyst::quantum::GlobalPhaseOp op,
                                   SmallVectorImpl<Operation*>& convertedOps) {
    auto controls = resolveQubits(op.getInCtrlQubits(), op);
    auto controlValues = resolveControlValues(op.getInCtrlValues(), op);
    if (failed(controls) || failed(controlValues)) {
      return failure();
    }
    if (controls->size() != controlValues->size()) {
      return op.emitError("control qubits and values must have equal length");
    }

    const ConvertedUnitary converted = createUnitary(
        op.getLoc(), "qco.gphase", "GlobalPhase", std::move(*controls),
        ValueRange{}, ValueRange{op.getAngle()}, *controlValues, 0,
        !op.getAdjoint());
    for (const auto [oldValue, newValue] :
         llvm::zip(op.getOutCtrlQubits(), converted.controls)) {
      qubits[oldValue] = newValue;
    }
    convertedOps.push_back(op);
    return success();
  }

  LogicalResult convertMeasure(catalyst::quantum::MeasureOp op,
                               SmallVectorImpl<Operation*>& convertedOps) {
    if (op.getPostselectAttr()) {
      return op.emitError("postselected measurements are not supported");
    }
    auto input = resolveQubit(op.getInQubit(), op);
    if (failed(input)) {
      return failure();
    }
    const Attribute nameAttr = op->getAttr(MEASURE_REGISTER_NAME_ATTR);
    const Attribute sizeAttr = op->getAttr(MEASURE_REGISTER_SIZE_ATTR);
    const Attribute indexAttr = op->getAttr(MEASURE_REGISTER_INDEX_ATTR);
    const bool hasName = static_cast<bool>(nameAttr);
    const bool hasSize = static_cast<bool>(sizeAttr);
    const bool hasIndex = static_cast<bool>(indexAttr);
    if (hasName != hasSize || hasName != hasIndex) {
      return op.emitError("QCO measurement register metadata must be all "
                          "present or all absent");
    }

    qco::MeasureOp measure;
    if (hasName) {
      const auto name = dyn_cast<StringAttr>(nameAttr);
      const auto size = dyn_cast<IntegerAttr>(sizeAttr);
      const auto index = dyn_cast<IntegerAttr>(indexAttr);
      if (!name || name.getValue().empty() || !size || !index ||
          !size.getType().isSignlessInteger(64) ||
          !index.getType().isSignlessInteger(64) || size.getInt() <= 0 ||
          index.getInt() < 0 || index.getInt() >= size.getInt()) {
        return op.emitError("malformed QCO measurement register metadata");
      }
      measure = qco::MeasureOp::create(builder, op.getLoc(), *input, name, size,
                                       index);
    } else {
      measure = qco::MeasureOp::create(builder, op.getLoc(), *input);
    }
    op.getMres().replaceAllUsesWith(measure.getResult());
    qubits[op.getOutQubit()] = measure.getQubitOut();
    convertedOps.push_back(op);
    return success();
  }

  func::FuncOp getOrCreateQubitBridge(const Location loc) {
    if (qubitBridge) {
      return qubitBridge;
    }

    std::string symbol = QUBIT_BRIDGE_SYMBOL.str();
    uint64_t suffix = 0;
    while (SymbolTable::lookupSymbolIn(module, symbol) != nullptr) {
      symbol = (llvm::Twine(QUBIT_BRIDGE_SYMBOL) + "_" + llvm::Twine(++suffix))
                   .str();
    }

    const OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    const auto functionType = builder.getFunctionType(
        TypeRange{qco::QubitType::get(builder.getContext())},
        TypeRange{catalyst::quantum::QubitType::get(builder.getContext())});
    qubitBridge = func::FuncOp::create(builder, loc, symbol, functionType);
    qubitBridge.setPrivate();
    qubitBridge->setAttr(QUBIT_BRIDGE_ATTR, builder.getUnitAttr());
    return qubitBridge;
  }

  func::FuncOp getOrCreateGateHintBridge(const Location loc,
                                         const StringRef gateName,
                                         const size_t nativeControlCount) {
    for (func::FuncOp bridge : gateHintBridges) {
      if (bridge->getAttrOfType<StringAttr>(GATE_NAME_ATTR).getValue() ==
              gateName &&
          bridge->getAttrOfType<IntegerAttr>(NATIVE_CONTROL_COUNT_ATTR)
                  .getInt() == static_cast<int64_t>(nativeControlCount)) {
        return bridge;
      }
    }

    std::string symbol = GATE_HINT_BRIDGE_SYMBOL.str();
    uint64_t suffix = 0;
    while (SymbolTable::lookupSymbolIn(module, symbol) != nullptr) {
      symbol =
          (llvm::Twine(GATE_HINT_BRIDGE_SYMBOL) + "_" + llvm::Twine(++suffix))
              .str();
    }

    const OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto bridge =
        func::FuncOp::create(builder, loc, symbol,
                             builder.getFunctionType(TypeRange{}, TypeRange{}));
    bridge.setPrivate();
    bridge->setAttr(GATE_HINT_BRIDGE_ATTR, builder.getUnitAttr());
    bridge->setAttr(GATE_NAME_ATTR, builder.getStringAttr(gateName));
    bridge->setAttr(
        NATIVE_CONTROL_COUNT_ATTR,
        builder.getI64IntegerAttr(static_cast<int64_t>(nativeControlCount)));
    gateHintBridges.push_back(bridge);
    return bridge;
  }

  Value createQubitBridge(const Location loc, const Value value) {
    auto call = func::CallOp::create(builder, loc, getOrCreateQubitBridge(loc),
                                     ValueRange{value});
    return call.getResult(0);
  }

  LogicalResult
  convertComputationalBasis(catalyst::quantum::ComputationalBasisOp op,
                            SmallVectorImpl<Operation*>& convertedOps) {
    if (!op.getQreg()) {
      return preserveOperation(op);
    }
    const auto iter = registers.find(op.getQreg());
    if (iter == registers.end()) {
      return op.emitError("observable register is not produced by a supported "
                          "static allocation");
    }

    SmallVector<Value> castQubits;
    castQubits.reserve(iter->second.size());
    for (const Value qubit : iter->second) {
      castQubits.push_back(createQubitBridge(op.getLoc(), qubit));
    }

    OperationState state(op.getLoc(), op->getName());
    state.addOperands(castQubits);
    state.addTypes(op->getResultTypes());
    for (const NamedAttribute attr : op->getAttrs()) {
      if (attr.getName() != "operandSegmentSizes") {
        state.addAttribute(attr.getName(), attr.getValue());
      }
    }
    state.addAttribute("operandSegmentSizes",
                       builder.getDenseI32ArrayAttr(
                           {static_cast<int32_t>(castQubits.size()), 0}));
    Operation* replacement = builder.create(state);
    op.getObs().replaceAllUsesWith(replacement->getResult(0));
    convertedOps.push_back(op);
    return success();
  }

  LogicalResult preserveOperation(Operation* op) {
    if (op->getName().getDialectNamespace() != "quantum") {
      return success();
    }

    for (const Value result : op->getResults()) {
      if (isCatalystQuantumType(result.getType())) {
        return op->emitError("high-level quantum operation must be decomposed "
                             "before catalystquantum-to-qco");
      }
    }
    for (OpOperand& operand : op->getOpOperands()) {
      if (isa<catalyst::quantum::QuregType>(operand.get().getType())) {
        return op->emitError("register-mode quantum operation must be "
                             "decomposed before catalystquantum-to-qco");
      }
      if (!isa<catalyst::quantum::QubitType>(operand.get().getType())) {
        continue;
      }
      auto resolved = resolveQubit(operand.get(), op);
      if (failed(resolved)) {
        return failure();
      }
      operand.set(createQubitBridge(op->getLoc(), *resolved));
    }
    return success();
  }

  ModuleOp module;
  OpBuilder builder;
  llvm::DenseMap<Value, Value> qubits;
  llvm::DenseMap<Value, SmallVector<Value>> registers;
  llvm::StringSet<> registerNames;
  func::FuncOp qubitBridge;
  SmallVector<func::FuncOp> gateHintBridges;
  uint64_t nextRegister = 0;
};

struct CatalystQuantumToQCO final
    : impl::CatalystQuantumToQCOBase<CatalystQuantumToQCO> {
  using CatalystQuantumToQCOBase::CatalystQuantumToQCOBase;

  void runOnOperation() override {
    if (failed(Converter(getOperation()).run())) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mqt::ir::conversions
