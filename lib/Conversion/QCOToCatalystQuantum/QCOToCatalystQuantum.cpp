/*
 * Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCOToCatalystQuantum/QCOToCatalystQuantum.h" // NOLINT(misc-include-cleaner)

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <Quantum/IR/QuantumOps.h>
#include <Quantum/IR/QuantumTypes.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <numbers>
#include <optional>
#include <string>
#include <utility>

namespace mqt::ir::conversions {

#define GEN_PASS_DEF_QCOTOCATALYSTQUANTUM
#include "mlir/Conversion/QCOToCatalystQuantum/QCOToCatalystQuantum.h.inc"

using namespace mlir;

namespace {

constexpr StringLiteral CONTROL_VALUES_ATTR = "catalyst.control_values";
constexpr StringLiteral GATE_NAME_ATTR = "catalyst.gate_name";
constexpr StringLiteral NATIVE_CONTROL_COUNT_ATTR =
    "catalyst.native_control_count";
constexpr StringLiteral NEGATIVE_CONTROL_WRAPPER_ATTR =
    "catalyst.negative_control_wrapper";
constexpr StringLiteral REGISTER_NAME_ATTR = "mqt.qco_register_name";
constexpr StringLiteral MEASURE_REGISTER_NAME_ATTR =
    "mqt.qco_measure_register_name";
constexpr StringLiteral MEASURE_REGISTER_SIZE_ATTR =
    "mqt.qco_measure_register_size";
constexpr StringLiteral MEASURE_REGISTER_INDEX_ATTR =
    "mqt.qco_measure_register_index";

[[nodiscard]] bool isQCOType(const Type type) {
  return isa<qco::QubitType>(type);
}

[[nodiscard]] bool isQCOOperation(Operation* op) {
  return op->getName().getDialectNamespace() ==
         qco::QCODialect::getDialectNamespace();
}

[[nodiscard]] bool hasQCOValue(Operation* op) {
  return llvm::any_of(op->getOperandTypes(), isQCOType) ||
         llvm::any_of(op->getResultTypes(), isQCOType);
}

[[nodiscard]] bool isQubitBridgeCast(Operation* op) {
  auto cast = dyn_cast<UnrealizedConversionCastOp>(op);
  return cast && cast.getNumOperands() == 1 && cast.getNumResults() == 1 &&
         isa<qco::QubitType>(cast.getOperand(0).getType()) &&
         isa<catalyst::quantum::QubitType>(cast.getResult(0).getType());
}

LogicalResult emitBoundaryError(Operation* op) {
  return op->emitError(
      "QCO qubits across function or control-flow boundaries are not "
      "supported");
}

LogicalResult validateBoundaryAndOperation(Operation* op) {
  if (auto func = dyn_cast<func::FuncOp>(op);
      func && (llvm::any_of(func.getFunctionType().getInputs(), isQCOType) ||
               llvm::any_of(func.getFunctionType().getResults(), isQCOType))) {
    return emitBoundaryError(op);
  }

  if (isQubitBridgeCast(op)) {
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc || !parentFunc.getBody().hasOneBlock() ||
        op->getBlock() != &parentFunc.getBody().front()) {
      return emitBoundaryError(op);
    }
    for (Operation* user : op->getUsers()) {
      if (user->getBlock() != op->getBlock() ||
          user->getName().getDialectNamespace() != "quantum" ||
          llvm::any_of(user->getResultTypes(), [](const Type type) {
            return isa<catalyst::quantum::QubitType,
                       catalyst::quantum::QuregType>(type);
          })) {
        return emitBoundaryError(user);
      }
    }
    return success();
  }

  if (!isQCOOperation(op)) {
    if (hasQCOValue(op)) {
      return emitBoundaryError(op);
    }
    return success();
  }

  if (isa<qco::StaticOp>(op)) {
    return op->emitError("qco.static hardware qubits are not supported");
  }

  if (!isa<qco::AllocOp, qco::DeallocOp, qco::MeasureOp, qco::YieldOp>(op) &&
      !isa<qco::UnitaryOpInterface>(op)) {
    return op->emitError("unsupported QCO operation: ") << op->getName();
  }

  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || !func.getBody().hasOneBlock()) {
    return emitBoundaryError(op);
  }

  if (!op->getParentOfType<qco::CtrlOp>() &&
      !op->getParentOfType<qco::InvOp>() &&
      op->getBlock() != &func.getBody().front()) {
    return emitBoundaryError(op);
  }

  if (op->hasAttr(CONTROL_VALUES_ATTR) &&
      !isa<DenseBoolArrayAttr>(op->getAttr(CONTROL_VALUES_ATTR))) {
    return op->emitError("malformed catalyst.control_values metadata");
  }
  if (op->hasAttr(GATE_NAME_ATTR) &&
      !isa<StringAttr>(op->getAttr(GATE_NAME_ATTR))) {
    return op->emitError("malformed catalyst.gate_name metadata");
  }
  if (op->hasAttr(NATIVE_CONTROL_COUNT_ATTR) &&
      !isa<IntegerAttr>(op->getAttr(NATIVE_CONTROL_COUNT_ATTR))) {
    return op->emitError("malformed catalyst.native_control_count metadata");
  }
  if (op->hasAttr(NATIVE_CONTROL_COUNT_ATTR) &&
      (!isa<qco::CtrlOp>(op) || !op->hasAttr(GATE_NAME_ATTR))) {
    return op->emitError(
        "catalyst.native_control_count must accompany a gate name on qco.ctrl");
  }
  if (op->hasAttr(NEGATIVE_CONTROL_WRAPPER_ATTR)) {
    if (!isa<UnitAttr>(op->getAttr(NEGATIVE_CONTROL_WRAPPER_ATTR)) ||
        !isa<qco::XOp>(op) || op->getNumOperands() != 1 ||
        op->getNumResults() != 1) {
      return op->emitError("malformed negative-control wrapper");
    }
    if (!func || op->getBlock() != &func.getBody().front()) {
      return op->emitError(
          "negative-control wrappers must be top-level operations");
    }
  }
  return success();
}

LogicalResult preflight(Operation* root) {
  const WalkResult result = root->walk([&](Operation* op) -> WalkResult {
    if (failed(validateBoundaryAndOperation(op))) {
      return WalkResult::interrupt();
    }

    for (Region& region : op->getRegions()) {
      for (Block& block : region) {
        if (!llvm::any_of(block.getArgumentTypes(), isQCOType)) {
          continue;
        }
        if (!isa<qco::CtrlOp, qco::InvOp>(op)) {
          (void)emitBoundaryError(op);
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

struct GateHints {
  std::optional<std::string> gateName;
  std::optional<size_t> nativeControlCount;
};

struct CatalystGateHintSpec {
  CatalystGateHintSpec(const StringRef qcoSymbol, const size_t numTargets,
                       const size_t numParameters,
                       const size_t numNativeControls,
                       const bool variadicTargets = false)
      : qcoSymbol(qcoSymbol), numTargets(numTargets),
        numParameters(numParameters), numNativeControls(numNativeControls),
        variadicTargets(variadicTargets) {}

  StringRef qcoSymbol;
  size_t numTargets;
  size_t numParameters;
  size_t numNativeControls;
  bool variadicTargets = false;
};

[[nodiscard]] std::optional<CatalystGateHintSpec>
lookupCatalystGateHint(const StringRef gateName) {
  if (gateName == "Identity") {
    return CatalystGateHintSpec{"id", 1, 0, 0};
  }
  if (gateName == "PauliX") {
    return CatalystGateHintSpec{"x", 1, 0, 0};
  }
  if (gateName == "PauliY") {
    return CatalystGateHintSpec{"y", 1, 0, 0};
  }
  if (gateName == "PauliZ") {
    return CatalystGateHintSpec{"z", 1, 0, 0};
  }
  if (gateName == "Hadamard") {
    return CatalystGateHintSpec{"h", 1, 0, 0};
  }
  if (gateName == "S") {
    return CatalystGateHintSpec{"s", 1, 0, 0};
  }
  if (gateName == "T") {
    return CatalystGateHintSpec{"t", 1, 0, 0};
  }
  if (gateName == "SX") {
    return CatalystGateHintSpec{"sx", 1, 0, 0};
  }
  if (gateName == "RX") {
    return CatalystGateHintSpec{"rx", 1, 1, 0};
  }
  if (gateName == "RY") {
    return CatalystGateHintSpec{"ry", 1, 1, 0};
  }
  if (gateName == "RZ") {
    return CatalystGateHintSpec{"rz", 1, 1, 0};
  }
  if (gateName == "PhaseShift") {
    return CatalystGateHintSpec{"p", 1, 1, 0};
  }
  if (gateName == "SWAP") {
    return CatalystGateHintSpec{"swap", 2, 0, 0};
  }
  if (gateName == "ISWAP") {
    return CatalystGateHintSpec{"iswap", 2, 0, 0};
  }
  if (gateName == "ECR") {
    return CatalystGateHintSpec{"ecr", 2, 0, 0};
  }
  if (gateName == "IsingXX") {
    return CatalystGateHintSpec{"rxx", 2, 1, 0};
  }
  if (gateName == "IsingYY") {
    return CatalystGateHintSpec{"ryy", 2, 1, 0};
  }
  if (gateName == "IsingZZ") {
    return CatalystGateHintSpec{"rzz", 2, 1, 0};
  }
  if (gateName == "IsingXY") {
    return CatalystGateHintSpec{"xx_plus_yy", 2, 2, 0};
  }
  if (gateName == "CNOT") {
    return CatalystGateHintSpec{"x", 1, 0, 1};
  }
  if (gateName == "CY") {
    return CatalystGateHintSpec{"y", 1, 0, 1};
  }
  if (gateName == "CZ") {
    return CatalystGateHintSpec{"z", 1, 0, 1};
  }
  if (gateName == "CRX") {
    return CatalystGateHintSpec{"rx", 1, 1, 1};
  }
  if (gateName == "CRY") {
    return CatalystGateHintSpec{"ry", 1, 1, 1};
  }
  if (gateName == "CRZ") {
    return CatalystGateHintSpec{"rz", 1, 1, 1};
  }
  if (gateName == "ControlledPhaseShift") {
    return CatalystGateHintSpec{"p", 1, 1, 1};
  }
  if (gateName == "Toffoli") {
    return CatalystGateHintSpec{"x", 1, 0, 2};
  }
  if (gateName == "CSWAP") {
    return CatalystGateHintSpec{"swap", 2, 0, 1};
  }
  if (gateName == "Barrier") {
    return CatalystGateHintSpec{"barrier", 0, 0, 0, true};
  }
  if (gateName == "GlobalPhase") {
    return CatalystGateHintSpec{"gphase", 0, 1, 0};
  }
  return std::nullopt;
}

struct Emission {
  Emission() = default;
  Emission(SmallVector<Value> outputs, SmallVector<Value> updatedControls)
      : outputs(std::move(outputs)),
        updatedControls(std::move(updatedControls)) {}

  SmallVector<Value> outputs;
  SmallVector<Value> updatedControls;
};

struct RegisterGroup {
  std::string name;
  uint64_t size = 0;
  SmallVector<qco::AllocOp> allocations;
  SmallVector<bool> deallocated;
  SmallVector<Value> currentValues;
  Value qreg;
  size_t numDeallocated = 0;
};

struct RegisterSlot {
  size_t group = 0;
  uint64_t index = 0;
};

class BlockConverter final {
public:
  explicit BlockConverter(Block& block)
      : block(&block), builder(block.getParentOp()) {}

  LogicalResult convert() {
    if (failed(collectRegisters())) {
      return failure();
    }
    if (failed(identifyNegativeControlSandwiches())) {
      return failure();
    }

    SmallVector<Operation*> convertedOps;
    for (Operation& op : *block) {
      if (isQCOOperation(&op) || isQubitBridgeCast(&op)) {
        convertedOps.push_back(&op);
      }
    }

    for (Operation* op : convertedOps) {
      builder.setInsertionPoint(op);
      if (auto alloc = dyn_cast<qco::AllocOp>(op)) {
        if (failed(convertAlloc(alloc))) {
          return failure();
        }
      } else if (auto dealloc = dyn_cast<qco::DeallocOp>(op)) {
        if (failed(convertDealloc(dealloc))) {
          return failure();
        }
      } else if (auto measure = dyn_cast<qco::MeasureOp>(op)) {
        if (failed(convertMeasure(measure))) {
          return failure();
        }
      } else if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op)) {
        if (failed(convertBridgeCast(cast))) {
          return failure();
        }
      } else if (isa<qco::UnitaryOpInterface>(op)) {
        if (failed(convertUnitary(op))) {
          return failure();
        }
      } else {
        return op->emitError("unsupported top-level QCO operation");
      }
    }

    for (const RegisterGroup& group : groups) {
      if (group.numDeallocated != 0 && group.numDeallocated != group.size) {
        qco::AllocOp allocation = group.allocations.front();
        return allocation.emitError(
            "partially deallocated QCO register cannot be reconstructed");
      }
    }

    for (Operation* op : llvm::reverse(convertedOps)) {
      if (!op->use_empty()) {
        return op->emitError("unsupported QCO use remains after conversion");
      }
      op->erase();
    }
    return success();
  }

private:
  struct NegativeControlSandwich {
    qco::XOp before;
    qco::XOp after;
  };

  [[nodiscard]] std::optional<NegativeControlSandwich>
  findNegativeControlSandwich(qco::CtrlOp ctrl, const size_t index) const {
    const Value controlIn = ctrl.getControlsIn()[index];
    auto before = controlIn.getDefiningOp<qco::XOp>();
    if (!before || before->getBlock() != block ||
        before->getNumOperands() != 1 || before->getNumResults() != 1 ||
        !controlIn.hasOneUse()) {
      return std::nullopt;
    }

    const Value controlOut = ctrl.getControlsOut()[index];
    if (!controlOut.hasOneUse()) {
      return std::nullopt;
    }
    auto after = dyn_cast<qco::XOp>(controlOut.use_begin()->getOwner());
    if (!after || after->getBlock() != block || after == before ||
        after->getNumOperands() != 1 || after->getNumResults() != 1) {
      return std::nullopt;
    }

    auto beforeUnitary = cast<qco::UnitaryOpInterface>(before.getOperation());
    auto afterUnitary = cast<qco::UnitaryOpInterface>(after.getOperation());
    if (beforeUnitary.getNumTargets() != 1 ||
        afterUnitary.getNumTargets() != 1 ||
        before->getResult(0) != controlIn ||
        afterUnitary.getInputTarget(0) != controlOut) {
      return std::nullopt;
    }
    return NegativeControlSandwich{.before = before, .after = after};
  }

  LogicalResult identifyNegativeControlSandwiches() {
    for (Operation& operation : *block) {
      auto ctrl = dyn_cast<qco::CtrlOp>(&operation);
      if (!ctrl) {
        continue;
      }

      const auto controlValues =
          ctrl->getAttrOfType<DenseBoolArrayAttr>(CONTROL_VALUES_ATTR);
      if (controlValues &&
          static_cast<size_t>(controlValues.size()) != ctrl.getNumControls()) {
        return ctrl.emitError(
            "catalyst.control_values size does not match qco.ctrl controls");
      }

      SmallVector<bool> inferred(ctrl.getNumControls(), false);
      bool hasInference = false;
      for (size_t index = 0; index < ctrl.getNumControls(); ++index) {
        auto sandwich = findNegativeControlSandwich(ctrl, index);
        if (!sandwich) {
          if (controlValues && !controlValues.asArrayRef()[index]) {
            return ctrl.emitError(
                "negative catalyst.control_values metadata requires an "
                "X-control-X sandwich");
          }
          continue;
        }

        const bool explicitlyNegative =
            controlValues && !controlValues.asArrayRef()[index];
        if (negativeControlWrappers.contains(sandwich->before.getOperation()) ||
            negativeControlWrappers.contains(sandwich->after.getOperation())) {
          if (explicitlyNegative) {
            return ctrl.emitError(
                "negative-control wrapper is shared by more than one "
                "qco.ctrl");
          }
          continue;
        }

        const bool beforeTagged =
            sandwich->before->hasAttr(NEGATIVE_CONTROL_WRAPPER_ATTR);
        const bool afterTagged =
            sandwich->after->hasAttr(NEGATIVE_CONTROL_WRAPPER_ATTR);
        if (beforeTagged != afterTagged ||
            (beforeTagged &&
             (!controlValues || controlValues.asArrayRef()[index]))) {
          return ctrl.emitError("malformed negative-control wrapper metadata");
        }

        if (!controlValues || explicitlyNegative) {
          negativeControlWrappers.insert(sandwich->before.getOperation());
          negativeControlWrappers.insert(sandwich->after.getOperation());
        }
        if (!controlValues) {
          inferred[index] = true;
          hasInference = true;
        }
      }
      if (hasInference) {
        inferredNegativeControls[ctrl.getOperation()] = std::move(inferred);
      }
    }

    for (Operation& operation : *block) {
      if (operation.hasAttr(NEGATIVE_CONTROL_WRAPPER_ATTR) &&
          !negativeControlWrappers.contains(&operation)) {
        return operation.emitError("malformed negative-control wrapper");
      }
    }
    return success();
  }

  LogicalResult collectRegisters() {
    llvm::StringMap<size_t> groupByName;
    const auto allocationCount =
        static_cast<uint64_t>(llvm::count_if(*block, [](Operation& operation) {
          return isa<qco::AllocOp>(operation);
        }));

    for (Operation& operation : *block) {
      auto alloc = dyn_cast<qco::AllocOp>(&operation);
      if (!alloc) {
        continue;
      }

      const bool hasName = static_cast<bool>(alloc.getRegisterNameAttr());
      const bool hasSize = static_cast<bool>(alloc.getRegisterSizeAttr());
      const bool hasIndex = static_cast<bool>(alloc.getRegisterIndexAttr());
      if (hasName != hasSize || hasName != hasIndex) {
        return alloc.emitError(
            "qco.alloc register metadata must provide register_name, "
            "register_size, and register_index together");
      }
      if (!hasName) {
        scalarRoots[alloc.getResult()] = alloc.getResult();
        continue;
      }

      const StringRef name = alloc.getRegisterNameAttr().getValue();
      const uint64_t size = alloc.getRegisterSize().value();
      const uint64_t index = alloc.getRegisterIndex().value();
      if (name.empty() || size == 0 || index >= size ||
          size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          size > allocationCount) {
        return alloc.emitError("malformed qco.alloc register metadata");
      }

      size_t groupIndex = 0;
      if (auto existing = groupByName.find(name);
          existing != groupByName.end()) {
        groupIndex = existing->second;
        if (groups[groupIndex].size != size) {
          return alloc.emitError(
                     "inconsistent qco.alloc register_size for register '")
                 << name << "'";
        }
      } else {
        groupIndex = groups.size();
        groupByName[name] = groupIndex;
        RegisterGroup group;
        group.name = name.str();
        group.size = size;
        group.allocations.resize(static_cast<size_t>(size));
        group.deallocated.resize(static_cast<size_t>(size), false);
        group.currentValues.resize(static_cast<size_t>(size));
        groups.push_back(std::move(group));
      }

      RegisterGroup& group = groups[groupIndex];
      if (group.allocations[static_cast<size_t>(index)]) {
        return alloc.emitError("duplicate qco.alloc register_index ")
               << index << " for register '" << name << "'";
      }
      group.allocations[static_cast<size_t>(index)] = alloc;
      registerSlots[alloc.getResult()] = {
          .group = groupIndex,
          .index = index,
      };
    }

    for (const RegisterGroup& group : groups) {
      if (llvm::any_of(group.allocations,
                       [](qco::AllocOp op) { return !op; })) {
        for (qco::AllocOp allocation : group.allocations) {
          if (allocation) {
            return allocation.emitError("qco.alloc register is incomplete: ")
                   << group.name;
          }
        }
        return failure();
      }
    }
    return success();
  }

  FailureOr<Value> lookup(Value value) { return lookup(value, emptyLocal); }

  FailureOr<Value> lookup(Value value,
                          const llvm::DenseMap<Value, Value>& local) {
    if (const auto it = local.find(value); it != local.end()) {
      return it->second;
    }
    if (const auto it = values.find(value); it != values.end()) {
      return it->second;
    }
    if (!isQCOType(value.getType())) {
      return value;
    }
    return failure();
  }

  void propagateLineage(Value input, Value output) {
    if (const auto it = registerSlots.find(input); it != registerSlots.end()) {
      const RegisterSlot slot = it->second;
      registerSlots[output] = slot;
    }
    if (const auto it = scalarRoots.find(input); it != scalarRoots.end()) {
      const Value root = it->second;
      scalarRoots[output] = root;
    }
  }

  void recordCurrentValue(Value input, Value output, Value converted) {
    propagateLineage(input, output);
    if (const auto slotIt = registerSlots.find(output);
        slotIt != registerSlots.end()) {
      const RegisterSlot slot = slotIt->second;
      groups[slot.group].currentValues[static_cast<size_t>(slot.index)] =
          converted;
    }
    if (const auto rootIt = scalarRoots.find(output);
        rootIt != scalarRoots.end()) {
      currentScalarValues[rootIt->second] = converted;
    }
  }

  FailureOr<Value> lookupCurrentValue(Value value) {
    if (const auto slotIt = registerSlots.find(value);
        slotIt != registerSlots.end()) {
      const RegisterSlot slot = slotIt->second;
      if (groups[slot.group].deallocated[static_cast<size_t>(slot.index)]) {
        return failure();
      }
      Value current =
          groups[slot.group].currentValues[static_cast<size_t>(slot.index)];
      if (current) {
        return current;
      }
      return failure();
    }
    if (const auto rootIt = scalarRoots.find(value);
        rootIt != scalarRoots.end()) {
      if (deallocatedScalarRoots.contains(rootIt->second)) {
        return failure();
      }
      if (const auto currentIt = currentScalarValues.find(rootIt->second);
          currentIt != currentScalarValues.end()) {
        return currentIt->second;
      }
      return failure();
    }
    return lookup(value);
  }

  LogicalResult convertAlloc(qco::AllocOp op) {
    if (const auto slotIt = registerSlots.find(op.getResult());
        slotIt != registerSlots.end()) {
      const RegisterSlot slot = slotIt->second;
      RegisterGroup& group = groups[slot.group];
      if (!group.qreg) {
        auto alloc = catalyst::quantum::AllocOp::create(
            builder, op.getLoc(),
            catalyst::quantum::QuregType::get(builder.getContext()), Value{},
            builder.getI64IntegerAttr(static_cast<int64_t>(group.size)));
        alloc->setAttr(REGISTER_NAME_ATTR, builder.getStringAttr(group.name));
        group.qreg = alloc.getQreg();
      }
      auto extract = catalyst::quantum::ExtractOp::create(
          builder, op.getLoc(),
          catalyst::quantum::QubitType::get(builder.getContext()), group.qreg,
          Value{}, builder.getI64IntegerAttr(static_cast<int64_t>(slot.index)));
      values[op.getResult()] = extract.getQubit();
      group.currentValues[static_cast<size_t>(slot.index)] = extract.getQubit();
      return success();
    }

    auto alloc = catalyst::quantum::AllocQubitOp::create(builder, op.getLoc());
    values[op.getResult()] = alloc.getQubit();
    currentScalarValues[op.getResult()] = alloc.getQubit();
    return success();
  }

  LogicalResult convertDealloc(qco::DeallocOp op) {
    auto mapped = lookup(op.getQubit());
    if (failed(mapped)) {
      return op.emitError("qco.dealloc consumes an unmapped qubit");
    }

    if (const auto slotIt = registerSlots.find(op.getQubit());
        slotIt != registerSlots.end()) {
      const RegisterSlot slot = slotIt->second;
      RegisterGroup& group = groups[slot.group];
      const auto index = static_cast<size_t>(slot.index);
      if (group.deallocated[index]) {
        return op.emitError("register qubit is deallocated more than once");
      }
      auto insert = catalyst::quantum::InsertOp::create(
          builder, op.getLoc(),
          catalyst::quantum::QuregType::get(builder.getContext()), group.qreg,
          Value{}, builder.getI64IntegerAttr(static_cast<int64_t>(slot.index)),
          *mapped);
      group.qreg = insert.getOutQreg();
      group.deallocated[index] = true;
      ++group.numDeallocated;
      if (group.numDeallocated == group.size) {
        catalyst::quantum::DeallocOp::create(builder, op.getLoc(), group.qreg);
      }
      return success();
    }

    if (!scalarRoots.contains(op.getQubit())) {
      return op.emitError("qco.dealloc qubit has no allocation origin");
    }
    const Value root = scalarRoots[op.getQubit()];
    if (!deallocatedScalarRoots.insert(root).second) {
      return op.emitError("scalar qubit is deallocated more than once");
    }
    catalyst::quantum::DeallocQubitOp::create(builder, op.getLoc(), *mapped);
    return success();
  }

  LogicalResult convertMeasure(qco::MeasureOp op) {
    auto input = lookup(op.getQubitIn());
    if (failed(input)) {
      return op.emitError("qco.measure consumes an unmapped qubit");
    }
    const bool hasName = static_cast<bool>(op.getRegisterNameAttr());
    const bool hasSize = static_cast<bool>(op.getRegisterSizeAttr());
    const bool hasIndex = static_cast<bool>(op.getRegisterIndexAttr());
    if (hasName != hasSize || hasName != hasIndex) {
      return op.emitError(
          "qco.measure register metadata must be all present or all absent");
    }
    if (hasName && op.getRegisterNameAttr().getValue().empty()) {
      return op.emitError("qco.measure register_name must be nonempty");
    }

    auto measure = catalyst::quantum::MeasureOp::create(
        builder, op.getLoc(), builder.getI1Type(),
        catalyst::quantum::QubitType::get(builder.getContext()), *input,
        /*postselect=*/nullptr);
    if (hasName) {
      measure->setAttr(MEASURE_REGISTER_NAME_ATTR, op.getRegisterNameAttr());
      measure->setAttr(MEASURE_REGISTER_SIZE_ATTR, op.getRegisterSizeAttr());
      measure->setAttr(MEASURE_REGISTER_INDEX_ATTR, op.getRegisterIndexAttr());
    }
    values[op.getQubitOut()] = measure.getOutQubit();
    recordCurrentValue(op.getQubitIn(), op.getQubitOut(),
                       measure.getOutQubit());
    op.getResult().replaceAllUsesWith(measure.getMres());
    return success();
  }

  LogicalResult convertBridgeCast(UnrealizedConversionCastOp op) {
    auto mapped = lookupCurrentValue(op.getOperand(0));
    if (failed(mapped)) {
      return op.emitError(
          "observable bridge qubit is unavailable at this program point");
    }
    op->getResult(0).replaceAllUsesWith(*mapped);
    return success();
  }

  LogicalResult convertUnitary(Operation* op) {
    auto unitary = cast<qco::UnitaryOpInterface>(op);

    if (negativeControlWrappers.contains(op)) {
      auto input = lookup(unitary.getInputTarget(0));
      if (failed(input)) {
        return op->emitError(
            "negative-control wrapper consumes an unmapped qubit");
      }
      values[op->getResult(0)] = *input;
      recordCurrentValue(unitary.getInputTarget(0), op->getResult(0), *input);
      return success();
    }

    const llvm::DenseMap<Value, Value> local;
    auto emission = emitUnitary(op, local, {}, {}, false, GateHints{});
    if (failed(emission)) {
      return failure();
    }
    if (emission->outputs.size() != op->getNumResults()) {
      return op->emitError("QCO unitary result arity cannot be reconstructed");
    }

    for (auto [source, converted] :
         llvm::zip_equal(op->getResults(), emission->outputs)) {
      values[source] = converted;
      recordCurrentValue(unitary.getInputForOutput(source), source, converted);
    }
    return success();
  }

  LogicalResult cloneClassicalPreamble(Block& body, Operation* unitary,
                                       llvm::DenseMap<Value, Value>& local) {
    for (Operation& nested : body.without_terminator()) {
      if (&nested == unitary) {
        continue;
      }
      if (isQCOOperation(&nested) || nested.getNumRegions() != 0 ||
          !isMemoryEffectFree(&nested)) {
        return nested.emitError("unsupported operation in QCO modifier region");
      }

      IRMapping mapping;
      for (const auto& [source, converted] : local) {
        mapping.map(source, converted);
      }
      Operation* clone = builder.clone(nested, mapping);
      for (auto [source, converted] :
           llvm::zip_equal(nested.getResults(), clone->getResults())) {
        local[source] = converted;
      }
    }
    return success();
  }

  static FailureOr<GateHints> readHints(Operation* op, GateHints inherited) {
    if (auto gate = op->getAttrOfType<StringAttr>(GATE_NAME_ATTR)) {
      if (gate.getValue().empty() ||
          (inherited.gateName && *inherited.gateName != gate.getValue())) {
        (void)op->emitError("inconsistent catalyst.gate_name metadata");
        return failure();
      }
      inherited.gateName = gate.getValue().str();
    }
    if (auto count =
            op->getAttrOfType<IntegerAttr>(NATIVE_CONTROL_COUNT_ATTR)) {
      if (!count.getType().isSignlessInteger(64)) {
        (void)op->emitError("malformed catalyst.native_control_count metadata");
        return failure();
      }
      const int64_t value = count.getInt();
      if (value < 0 ||
          std::cmp_greater(value, std::numeric_limits<size_t>::max())) {
        (void)op->emitError("malformed catalyst.native_control_count metadata");
        return failure();
      }
      const auto nativeControlCount = static_cast<size_t>(value);
      if (inherited.nativeControlCount &&
          *inherited.nativeControlCount != nativeControlCount) {
        (void)op->emitError(
            "inconsistent catalyst.native_control_count metadata");
        return failure();
      }
      inherited.nativeControlCount = nativeControlCount;
    }
    return inherited;
  }

  static LogicalResult
  validateGateHints(Operation* op, const StringRef qcoSymbol,
                    const size_t numTargets, const size_t numParameters,
                    const size_t numControls, const GateHints& hints) {
    if (!hints.gateName && !hints.nativeControlCount) {
      return success();
    }
    if (!hints.gateName) {
      return op->emitError(
          "catalyst.native_control_count is missing catalyst.gate_name");
    }

    const auto spec = lookupCatalystGateHint(*hints.gateName);
    const size_t nativeControlCount = hints.nativeControlCount.value_or(0);
    if (!spec || spec->qcoSymbol != qcoSymbol ||
        (!spec->variadicTargets && spec->numTargets != numTargets) ||
        spec->numParameters != numParameters ||
        spec->numNativeControls != nativeControlCount ||
        nativeControlCount > numControls) {
      return op->emitError("catalyst gate metadata is inconsistent with ")
             << op->getName();
    }
    return success();
  }

  FailureOr<Emission> emitUnitary(Operation* op,
                                  llvm::DenseMap<Value, Value> local,
                                  ArrayRef<Value> inheritedControls,
                                  ArrayRef<bool> inheritedControlValues,
                                  const bool inverted, GateHints hints) {
    auto updatedHints = readHints(op, std::move(hints));
    if (failed(updatedHints)) {
      return failure();
    }

    if (auto ctrl = dyn_cast<qco::CtrlOp>(op)) {
      if (ctrl.getControlsIn().size() != ctrl.getControlsOut().size()) {
        (void)ctrl.emitError("malformed qco.ctrl control arity");
        return failure();
      }

      SmallVector<bool> ownControlValues(ctrl.getNumControls(), true);
      if (auto attr =
              ctrl->getAttrOfType<DenseBoolArrayAttr>(CONTROL_VALUES_ATTR)) {
        if (static_cast<size_t>(attr.size()) != ctrl.getNumControls()) {
          (void)ctrl.emitError(
              "catalyst.control_values size does not match qco.ctrl controls");
          return failure();
        }
        ownControlValues.assign(attr.asArrayRef().begin(),
                                attr.asArrayRef().end());
      }
      if (const auto inferred = inferredNegativeControls.find(op);
          inferred != inferredNegativeControls.end()) {
        for (const auto [index, isNegative] :
             llvm::enumerate(inferred->second)) {
          if (isNegative) {
            ownControlValues[index] = false;
          }
        }
      }

      SmallVector<Value> allControls(inheritedControls);
      for (const Value control : ctrl.getControlsIn()) {
        auto mapped = lookup(control, local);
        if (failed(mapped)) {
          (void)ctrl.emitError("qco.ctrl consumes an unmapped control qubit");
          return failure();
        }
        allControls.push_back(*mapped);
      }
      SmallVector<bool> allControlValues(inheritedControlValues);
      allControlValues.append(ownControlValues);

      Block& body = ctrl.getRegion().front();
      if (body.getNumArguments() != ctrl.getTargetsIn().size()) {
        (void)ctrl.emitError("malformed qco.ctrl target aliases");
        return failure();
      }
      for (auto [argument, target] :
           llvm::zip_equal(body.getArguments(), ctrl.getTargetsIn())) {
        auto mapped = lookup(target, local);
        if (failed(mapped)) {
          (void)ctrl.emitError("qco.ctrl consumes an unmapped target qubit");
          return failure();
        }
        local[argument] = *mapped;
      }

      Operation* bodyUnitary = ctrl.getBodyUnitary().getOperation();
      if (failed(cloneClassicalPreamble(body, bodyUnitary, local))) {
        return failure();
      }
      auto bodyEmission =
          emitUnitary(bodyUnitary, local, allControls, allControlValues,
                      inverted, std::move(*updatedHints));
      if (failed(bodyEmission)) {
        return failure();
      }
      if (bodyEmission->updatedControls.size() != allControls.size()) {
        (void)ctrl.emitError(
            "qco.ctrl control result arity cannot be reconstructed");
        return failure();
      }

      for (auto [source, converted] :
           llvm::zip_equal(bodyUnitary->getResults(), bodyEmission->outputs)) {
        local[source] = converted;
      }
      auto yield = cast<qco::YieldOp>(body.back());

      Emission result;
      const ArrayRef<Value> updatedControls = bodyEmission->updatedControls;
      const ArrayRef<Value> inheritedControlResults =
          updatedControls.take_front(inheritedControls.size());
      result.updatedControls.append(inheritedControlResults.begin(),
                                    inheritedControlResults.end());
      const ArrayRef<Value> targetResults =
          updatedControls.drop_front(inheritedControls.size());
      result.outputs.append(targetResults.begin(), targetResults.end());
      for (const Value target : yield.getTargets()) {
        auto mapped = lookup(target, local);
        if (failed(mapped)) {
          (void)ctrl.emitError("qco.ctrl yields an unmapped target qubit");
          return failure();
        }
        result.outputs.push_back(*mapped);
      }
      return result;
    }

    if (auto inv = dyn_cast<qco::InvOp>(op)) {
      Block& body = inv.getRegion().front();
      if (body.getNumArguments() != inv.getQubitsIn().size()) {
        (void)inv.emitError("malformed qco.inv target aliases");
        return failure();
      }
      for (auto [argument, target] :
           llvm::zip_equal(body.getArguments(), inv.getQubitsIn())) {
        auto mapped = lookup(target, local);
        if (failed(mapped)) {
          (void)inv.emitError("qco.inv consumes an unmapped qubit");
          return failure();
        }
        local[argument] = *mapped;
      }

      Operation* bodyUnitary = inv.getBodyUnitary().getOperation();
      if (failed(cloneClassicalPreamble(body, bodyUnitary, local))) {
        return failure();
      }
      auto bodyEmission = emitUnitary(bodyUnitary, local, inheritedControls,
                                      inheritedControlValues, !inverted,
                                      std::move(*updatedHints));
      if (failed(bodyEmission)) {
        return failure();
      }
      for (auto [source, converted] :
           llvm::zip_equal(bodyUnitary->getResults(), bodyEmission->outputs)) {
        local[source] = converted;
      }

      Emission result;
      result.updatedControls = std::move(bodyEmission->updatedControls);
      auto yield = cast<qco::YieldOp>(body.back());
      for (const Value target : yield.getTargets()) {
        auto mapped = lookup(target, local);
        if (failed(mapped)) {
          (void)inv.emitError("qco.inv yields an unmapped qubit");
          return failure();
        }
        result.outputs.push_back(*mapped);
      }
      return result;
    }

    return emitBaseUnitary(op, local, inheritedControls, inheritedControlValues,
                           inverted, std::move(*updatedHints));
  }

  FailureOr<Emission> emitBaseUnitary(Operation* op,
                                      const llvm::DenseMap<Value, Value>& local,
                                      ArrayRef<Value> controls,
                                      ArrayRef<bool> controlValues,
                                      const bool inverted, GateHints hints) {
    auto unitary = dyn_cast<qco::UnitaryOpInterface>(op);
    if (!unitary) {
      (void)op->emitError("QCO modifier body is not unitary");
      return failure();
    }

    SmallVector<Value> targets;
    targets.reserve(unitary.getNumTargets());
    for (size_t i = 0; i < unitary.getNumTargets(); ++i) {
      auto mapped = lookup(unitary.getInputTarget(i), local);
      if (failed(mapped)) {
        (void)op->emitError("QCO gate consumes an unmapped target qubit");
        return failure();
      }
      targets.push_back(*mapped);
    }

    SmallVector<Value> params;
    params.reserve(unitary.getNumParams());
    for (size_t i = 0; i < unitary.getNumParams(); ++i) {
      auto mapped = lookup(unitary.getParameter(i), local);
      if (failed(mapped)) {
        (void)op->emitError("QCO gate consumes an unmapped parameter");
        return failure();
      }
      params.push_back(*mapped);
    }

    const StringRef symbol = unitary.getBaseSymbol();
    if (failed(validateGateHints(op, symbol, targets.size(), params.size(),
                                 controls.size(), hints))) {
      return failure();
    }
    if (symbol == "dcx") {
      return emitDCX(op->getLoc(), targets, controls, controlValues, inverted);
    }
    if (symbol == "rzx") {
      return emitRZX(op->getLoc(), targets, params, controls, controlValues,
                     inverted);
    }
    if (symbol == "u" || symbol == "u2") {
      return emitU(op->getLoc(), symbol, targets, params, controls,
                   controlValues, inverted);
    }
    if (symbol == "r") {
      return emitR(op->getLoc(), targets, params, controls, controlValues,
                   inverted);
    }
    if (symbol == "xx_minus_yy") {
      return emitXXPlusMinusYY(op->getLoc(), targets, params, controls,
                               controlValues, inverted,
                               /*minus=*/true);
    }
    if (symbol == "xx_plus_yy" &&
        (params.size() != 2 || !isPiConstant(params[1]))) {
      return emitXXPlusMinusYY(op->getLoc(), targets, params, controls,
                               controlValues, inverted,
                               /*minus=*/false);
    }
    if (symbol == "sx" || symbol == "sxdg") {
      return emitSX(op->getLoc(), targets, controls, controlValues,
                    inverted != (symbol == "sxdg"));
    }
    if (symbol == "ecr") {
      return emitECR(op->getLoc(), targets, controls, controlValues, inverted);
    }

    if (isa<qco::GPhaseOp>(op)) {
      if (params.size() != 1 || !targets.empty()) {
        (void)op->emitError("malformed qco.gphase");
        return failure();
      }
      const SmallVector<Value> ctrlValues =
          materializeControlValues(op->getLoc(), controlValues);
      const Type qubitType =
          catalyst::quantum::QubitType::get(builder.getContext());
      // QCO uses exp(+i theta), while Catalyst uses exp(-i theta).
      auto phase = catalyst::quantum::GlobalPhaseOp::create(
          builder, op->getLoc(), SmallVector<Type>(controls.size(), qubitType),
          params[0], !inverted, controls, ctrlValues);
      Emission result;
      result.updatedControls.append(phase.getOutCtrlQubits().begin(),
                                    phase.getOutCtrlQubits().end());
      return result;
    }

    if (symbol == "barrier") {
      return Emission{SmallVector<Value>(targets.begin(), targets.end()),
                      SmallVector<Value>(controls.begin(), controls.end())};
    }

    std::string gateName;
    bool intrinsicAdjoint = false;
    if (symbol == "id") {
      gateName = "Identity";
    } else if (symbol == "x") {
      gateName = "PauliX";
    } else if (symbol == "y") {
      gateName = "PauliY";
    } else if (symbol == "z") {
      gateName = "PauliZ";
    } else if (symbol == "h") {
      gateName = "Hadamard";
    } else if (symbol == "s" || symbol == "sdg") {
      gateName = "S";
      intrinsicAdjoint = symbol == "sdg";
    } else if (symbol == "t" || symbol == "tdg") {
      gateName = "T";
      intrinsicAdjoint = symbol == "tdg";
    } else if (symbol == "rx") {
      gateName = "RX";
    } else if (symbol == "ry") {
      gateName = "RY";
    } else if (symbol == "rz") {
      gateName = "RZ";
    } else if (symbol == "p") {
      gateName = "PhaseShift";
    } else if (symbol == "swap") {
      gateName = "SWAP";
    } else if (symbol == "iswap") {
      gateName = "ISWAP";
    } else if (symbol == "rxx") {
      gateName = "IsingXX";
    } else if (symbol == "ryy") {
      gateName = "IsingYY";
    } else if (symbol == "rzz") {
      gateName = "IsingZZ";
    } else if (symbol == "xx_plus_yy") {
      if (params.size() != 2 || !isPiConstant(params[1])) {
        (void)op->emitError(
            "qco.xx_plus_yy is only supported for Catalyst IsingXY beta=pi");
        return failure();
      }
      gateName = "IsingXY";
      params.resize(1);
    } else {
      (void)(op->emitError("unsupported QCO gate for Catalyst: ") << symbol);
      return failure();
    }

    if (hints.gateName) {
      gateName = *hints.gateName;
    }

    size_t nativeControls = hints.nativeControlCount.value_or(0);
    if (!hints.nativeControlCount && !hints.gateName) {
      if ((symbol == "x" || symbol == "y" || symbol == "z" || symbol == "rx" ||
           symbol == "ry" || symbol == "rz" || symbol == "p") &&
          controls.size() == 1 && controlValues.front()) {
        nativeControls = 1;
        if (symbol == "x") {
          gateName = "CNOT";
        } else if (symbol == "y") {
          gateName = "CY";
        } else if (symbol == "z") {
          gateName = "CZ";
        } else if (symbol == "rx") {
          gateName = "CRX";
        } else if (symbol == "ry") {
          gateName = "CRY";
        } else if (symbol == "rz") {
          gateName = "CRZ";
        } else {
          gateName = "ControlledPhaseShift";
        }
      } else if (symbol == "x" && controls.size() == 2 &&
                 llvm::all_of(controlValues,
                              [](bool value) { return value; })) {
        nativeControls = 2;
        gateName = "Toffoli";
      } else if (symbol == "swap" && controls.size() == 1 &&
                 controlValues.front()) {
        nativeControls = 1;
        gateName = "CSWAP";
      }
    }

    if (nativeControls > controls.size() ||
        !llvm::all_of(controlValues.take_front(nativeControls),
                      [](bool value) { return value; })) {
      (void)op->emitError(
          "catalyst.native_control_count does not describe positive controls");
      return failure();
    }

    SmallVector<Value> inQubits(controls.take_front(nativeControls));
    inQubits.append(targets);
    const ArrayRef<Value> modifierControls =
        controls.drop_front(nativeControls);
    const ArrayRef<bool> modifierControlValues =
        controlValues.drop_front(nativeControls);
    const SmallVector<Value> ctrlValues =
        materializeControlValues(op->getLoc(), modifierControlValues);

    auto custom = catalyst::quantum::CustomOp::create(
        builder, op->getLoc(), gateName, inQubits, modifierControls, ctrlValues,
        params, intrinsicAdjoint != inverted);
    const auto outQubitResults = custom.getOutQubits();
    const auto outControlResults = custom.getOutCtrlQubits();
    const SmallVector<Value> outQubits(outQubitResults.begin(),
                                       outQubitResults.end());
    const SmallVector<Value> outControls(outControlResults.begin(),
                                         outControlResults.end());
    if (outQubits.size() != inQubits.size() ||
        outControls.size() != modifierControls.size()) {
      (void)op->emitError("Catalyst gate result arity cannot be reconstructed");
      return failure();
    }

    Emission result;
    const ArrayRef<Value> outQubitRange = outQubits;
    const ArrayRef<Value> targetResults =
        outQubitRange.drop_front(nativeControls);
    result.outputs.append(targetResults.begin(), targetResults.end());
    const ArrayRef<Value> nativeControlResults =
        outQubitRange.take_front(nativeControls);
    result.updatedControls.append(nativeControlResults.begin(),
                                  nativeControlResults.end());
    result.updatedControls.append(outControls);
    return result;
  }

  FailureOr<Emission>
  emitPrimitive(Location loc, const StringRef gate, ArrayRef<Value> targets,
                ArrayRef<Value> params, ArrayRef<Value> controls,
                ArrayRef<bool> controlValues, const bool adjoint = false) {
    const SmallVector<Value> ctrlValues =
        materializeControlValues(loc, controlValues);
    auto custom = catalyst::quantum::CustomOp::create(
        builder, loc, gate, targets, controls, ctrlValues, params, adjoint);
    const auto outQubitResults = custom.getOutQubits();
    const auto outControlResults = custom.getOutCtrlQubits();
    SmallVector<Value> outQubits(outQubitResults.begin(),
                                 outQubitResults.end());
    SmallVector<Value> outControls(outControlResults.begin(),
                                   outControlResults.end());
    if (outQubits.size() != targets.size() ||
        outControls.size() != controls.size()) {
      (void)(emitError(loc)
             << "Catalyst gate result arity cannot be reconstructed");
      return failure();
    }
    Emission result;
    result.outputs = std::move(outQubits);
    result.updatedControls = std::move(outControls);
    return result;
  }

  FailureOr<Emission> emitPhase(Location loc, Value angle,
                                ArrayRef<Value> controls,
                                ArrayRef<bool> controlValues,
                                const bool adjoint) {
    const SmallVector<Value> ctrlValues =
        materializeControlValues(loc, controlValues);
    const Type qubitType =
        catalyst::quantum::QubitType::get(builder.getContext());
    auto phase = catalyst::quantum::GlobalPhaseOp::create(
        builder, loc, SmallVector<Type>(controls.size(), qubitType), angle,
        adjoint, controls, ctrlValues);
    const auto outControlResults = phase.getOutCtrlQubits();
    SmallVector<Value> outControls(outControlResults.begin(),
                                   outControlResults.end());
    if (outControls.size() != controls.size()) {
      (void)(emitError(loc)
             << "Catalyst global phase result arity cannot be reconstructed");
      return failure();
    }
    Emission result;
    result.updatedControls = std::move(outControls);
    return result;
  }

  FailureOr<Emission> emitSX(Location loc, ArrayRef<Value> targets,
                             ArrayRef<Value> controls,
                             ArrayRef<bool> controlValues, const bool adjoint) {
    if (targets.size() != 1) {
      (void)(emitError(loc) << "malformed qco.sx or qco.sxdg");
      return failure();
    }

    const Value halfPi = arith::ConstantOp::create(
        builder, loc, builder.getF64FloatAttr(std::numbers::pi / 2.0));
    const Value negativeQuarterPi = arith::ConstantOp::create(
        builder, loc, builder.getF64FloatAttr(-std::numbers::pi / 4.0));

    auto rotation = emitPrimitive(loc, "RX", targets, {halfPi}, controls,
                                  controlValues, adjoint);
    if (failed(rotation)) {
      return failure();
    }
    auto phase = emitPhase(loc, negativeQuarterPi, rotation->updatedControls,
                           controlValues, adjoint);
    if (failed(phase)) {
      return failure();
    }
    return Emission{{rotation->outputs[0]}, std::move(phase->updatedControls)};
  }

  FailureOr<Emission> emitECR(Location loc, ArrayRef<Value> targets,
                              ArrayRef<Value> controls,
                              ArrayRef<bool> controlValues,
                              const bool inverted) {
    if (targets.size() != 2) {
      (void)(emitError(loc) << "malformed qco.ecr");
      return failure();
    }

    const Value halfPi = arith::ConstantOp::create(
        builder, loc, builder.getF64FloatAttr(std::numbers::pi / 2.0));
    SmallVector<Value> qubits(targets);
    SmallVector<Value> currentControls(controls);

    auto apply = [&](const StringRef gate, const ArrayRef<size_t> indices,
                     const bool adjoint = false) -> LogicalResult {
      SmallVector<Value> gateTargets;
      gateTargets.reserve(indices.size());
      for (const size_t index : indices) {
        gateTargets.push_back(qubits[index]);
      }
      auto emission = emitPrimitive(loc, gate, gateTargets, {}, currentControls,
                                    controlValues, adjoint);
      if (failed(emission)) {
        return failure();
      }
      for (const auto [index, output] :
           llvm::zip_equal(indices, emission->outputs)) {
        qubits[index] = output;
      }
      currentControls = std::move(emission->updatedControls);
      return success();
    };

    auto applyRotation = [&](const StringRef gate, const size_t index,
                             const bool adjoint) -> LogicalResult {
      auto emission = emitPrimitive(loc, gate, {qubits[index]}, {halfPi},
                                    currentControls, controlValues, adjoint);
      if (failed(emission)) {
        return failure();
      }
      qubits[index] = emission->outputs[0];
      currentControls = std::move(emission->updatedControls);
      return success();
    };

    auto applySX = [&](const size_t index,
                       const bool adjoint) -> LogicalResult {
      auto emission =
          emitSX(loc, {qubits[index]}, currentControls, controlValues, adjoint);
      if (failed(emission)) {
        return failure();
      }
      qubits[index] = emission->outputs[0];
      currentControls = std::move(emission->updatedControls);
      return success();
    };

    if ((!inverted &&
         (failed(apply("PauliZ", {0})) || failed(apply("CNOT", {0, 1})) ||
          failed(applySX(1, false)) || failed(applyRotation("RX", 0, false)) ||
          failed(applyRotation("RY", 0, false)) ||
          failed(applyRotation("RX", 0, false)))) ||
        (inverted &&
         (failed(applyRotation("RX", 0, true)) ||
          failed(applyRotation("RY", 0, true)) ||
          failed(applyRotation("RX", 0, true)) || failed(applySX(1, true)) ||
          failed(apply("CNOT", {0, 1})) || failed(apply("PauliZ", {0}))))) {
      return failure();
    }
    return Emission{std::move(qubits), std::move(currentControls)};
  }

  FailureOr<Emission> emitDCX(Location loc, ArrayRef<Value> targets,
                              ArrayRef<Value> controls,
                              ArrayRef<bool> controlValues,
                              const bool inverted) {
    if (targets.size() != 2) {
      (void)(emitError(loc) << "malformed qco.dcx");
      return failure();
    }

    SmallVector<Value> qubits(targets);
    SmallVector<Value> currentControls(controls);
    constexpr std::array<std::pair<size_t, size_t>, 2> forward = {
        std::pair<size_t, size_t>{0, 1},
        std::pair<size_t, size_t>{1, 0},
    };
    constexpr std::array<std::pair<size_t, size_t>, 2> inverse = {
        std::pair<size_t, size_t>{1, 0},
        std::pair<size_t, size_t>{0, 1},
    };
    const auto& sequence = inverted ? inverse : forward;
    for (const auto [controlIndex, targetIndex] : sequence) {
      const SmallVector<Value> operands = {qubits[controlIndex],
                                           qubits[targetIndex]};
      auto cnot = emitPrimitive(loc, "CNOT", operands, {}, currentControls,
                                controlValues);
      if (failed(cnot)) {
        return failure();
      }
      qubits[controlIndex] = cnot->outputs[0];
      qubits[targetIndex] = cnot->outputs[1];
      currentControls = std::move(cnot->updatedControls);
    }
    return Emission{std::move(qubits), std::move(currentControls)};
  }

  FailureOr<Emission> emitRZX(Location loc, ArrayRef<Value> targets,
                              ArrayRef<Value> params, ArrayRef<Value> controls,
                              ArrayRef<bool> controlValues,
                              const bool inverted) {
    if (targets.size() != 2 || params.size() != 1) {
      (void)(emitError(loc) << "malformed qco.rzx");
      return failure();
    }

    auto firstH = emitPrimitive(loc, "Hadamard", {targets[1]}, {}, {}, {});
    if (failed(firstH)) {
      return failure();
    }
    const SmallVector<Value> rzzTargets = {targets[0], firstH->outputs[0]};
    auto rzz = emitPrimitive(loc, "IsingZZ", rzzTargets, params, controls,
                             controlValues, inverted);
    if (failed(rzz)) {
      return failure();
    }
    auto secondH =
        emitPrimitive(loc, "Hadamard", {rzz->outputs[1]}, {}, {}, {});
    if (failed(secondH)) {
      return failure();
    }
    return Emission{{rzz->outputs[0], secondH->outputs[0]},
                    std::move(rzz->updatedControls)};
  }

  FailureOr<Emission> emitU(Location loc, const StringRef symbol,
                            ArrayRef<Value> targets, ArrayRef<Value> params,
                            ArrayRef<Value> controls,
                            ArrayRef<bool> controlValues, const bool inverted) {
    if (targets.size() != 1 || (symbol == "u" && params.size() != 3) ||
        (symbol == "u2" && params.size() != 2)) {
      (void)(emitError(loc) << "malformed qco." << symbol);
      return failure();
    }

    Value theta;
    Value phi;
    Value lambda;
    if (symbol == "u") {
      theta = params[0];
      phi = params[1];
      lambda = params[2];
    } else {
      theta = arith::ConstantOp::create(
          builder, loc, builder.getF64FloatAttr(std::numbers::pi / 2.0));
      phi = params[0];
      lambda = params[1];
    }

    const Value sum = arith::AddFOp::create(builder, loc, phi, lambda);
    const Value half =
        arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.5));
    const Value phaseAngle = arith::MulFOp::create(builder, loc, sum, half);

    Value qubit = targets[0];
    SmallVector<Value> currentControls(controls);
    auto apply = [&](StringRef gate, Value parameter,
                     bool adjoint) -> LogicalResult {
      auto gateEmission =
          emitPrimitive(loc, gate, {qubit}, {parameter}, currentControls,
                        controlValues, adjoint);
      if (failed(gateEmission)) {
        return failure();
      }
      qubit = gateEmission->outputs[0];
      currentControls = std::move(gateEmission->updatedControls);
      return success();
    };

    if (!inverted) {
      if (failed(apply("RZ", lambda, false)) ||
          failed(apply("RY", theta, false)) ||
          failed(apply("RZ", phi, false))) {
        return failure();
      }
      auto phase =
          emitPhase(loc, phaseAngle, currentControls, controlValues, true);
      if (failed(phase)) {
        return failure();
      }
      currentControls = std::move(phase->updatedControls);
    } else {
      auto phase =
          emitPhase(loc, phaseAngle, currentControls, controlValues, false);
      if (failed(phase)) {
        return failure();
      }
      currentControls = std::move(phase->updatedControls);
      if (failed(apply("RZ", phi, true)) || failed(apply("RY", theta, true)) ||
          failed(apply("RZ", lambda, true))) {
        return failure();
      }
    }
    return Emission{{qubit}, std::move(currentControls)};
  }

  FailureOr<Emission> emitR(Location loc, ArrayRef<Value> targets,
                            ArrayRef<Value> params, ArrayRef<Value> controls,
                            ArrayRef<bool> controlValues, const bool inverted) {
    if (targets.size() != 1 || params.size() != 2) {
      (void)(emitError(loc) << "malformed qco.r");
      return failure();
    }
    const Value piHalf = arith::ConstantOp::create(
        builder, loc, builder.getF64FloatAttr(std::numbers::pi / 2.0));
    const Value firstAngle =
        arith::SubFOp::create(builder, loc, piHalf, params[1]);
    const Value lastAngle =
        arith::SubFOp::create(builder, loc, params[1], piHalf);

    Value qubit = targets[0];
    SmallVector<Value> currentControls(controls);
    auto apply = [&](StringRef gate, Value parameter,
                     bool adjoint) -> LogicalResult {
      auto gateEmission =
          emitPrimitive(loc, gate, {qubit}, {parameter}, currentControls,
                        controlValues, adjoint);
      if (failed(gateEmission)) {
        return failure();
      }
      qubit = gateEmission->outputs[0];
      currentControls = std::move(gateEmission->updatedControls);
      return success();
    };

    if ((!inverted && (failed(apply("RZ", firstAngle, false)) ||
                       failed(apply("RY", params[0], false)) ||
                       failed(apply("RZ", lastAngle, false)))) ||
        (inverted && (failed(apply("RZ", lastAngle, true)) ||
                      failed(apply("RY", params[0], true)) ||
                      failed(apply("RZ", firstAngle, true))))) {
      return failure();
    }
    return Emission{{qubit}, std::move(currentControls)};
  }

  FailureOr<Emission> emitXXPlusMinusYY(Location loc, ArrayRef<Value> targets,
                                        ArrayRef<Value> params,
                                        ArrayRef<Value> controls,
                                        ArrayRef<bool> controlValues,
                                        const bool inverted, const bool minus) {
    if (targets.size() != 2 || params.size() != 2) {
      (void)(emitError(loc)
             << "malformed qco.xx_" << (minus ? "minus" : "plus") << "_yy");
      return failure();
    }
    const Value pi = arith::ConstantOp::create(
        builder, loc, builder.getF64FloatAttr(std::numbers::pi));
    const Value piMinusBeta =
        arith::SubFOp::create(builder, loc, pi, params[1]);
    const Value betaMinusPi =
        arith::SubFOp::create(builder, loc, params[1], pi);

    SmallVector<Value> qubits(targets);
    SmallVector<Value> currentControls(controls);
    auto apply = [&](StringRef gate, ArrayRef<Value> gateTargets,
                     ArrayRef<Value> gateParams,
                     bool adjoint = false) -> FailureOr<SmallVector<Value>> {
      auto gateEmission =
          emitPrimitive(loc, gate, gateTargets, gateParams, currentControls,
                        controlValues, adjoint);
      if (failed(gateEmission)) {
        return failure();
      }
      currentControls = std::move(gateEmission->updatedControls);
      return std::move(gateEmission->outputs);
    };

    if (minus) {
      auto x = apply("PauliX", {qubits[0]}, {});
      if (failed(x)) {
        return failure();
      }
      qubits[0] = (*x)[0];
    }
    auto firstRz = apply("RZ", {qubits[1]}, {piMinusBeta});
    if (failed(firstRz)) {
      return failure();
    }
    qubits[1] = (*firstRz)[0];
    auto ising = apply("IsingXY", qubits, {params[0]}, /*adjoint=*/inverted);
    if (failed(ising)) {
      return failure();
    }
    qubits = std::move(*ising);
    auto secondRz = apply("RZ", {qubits[1]}, {betaMinusPi});
    if (failed(secondRz)) {
      return failure();
    }
    qubits[1] = (*secondRz)[0];
    if (minus) {
      auto x = apply("PauliX", {qubits[0]}, {});
      if (failed(x)) {
        return failure();
      }
      qubits[0] = (*x)[0];
    }
    return Emission{std::move(qubits), std::move(currentControls)};
  }

  SmallVector<Value> materializeControlValues(Location loc,
                                              ArrayRef<bool> values) {
    SmallVector<Value> result;
    result.reserve(values.size());
    for (const bool value : values) {
      result.push_back(arith::ConstantIntOp::create(
          builder, loc, static_cast<int64_t>(value), 1));
    }
    return result;
  }

  [[nodiscard]] static bool isPiConstant(Value value) {
    Attribute constant;
    if (!matchPattern(value, m_Constant(&constant))) {
      return false;
    }
    const auto floatAttr = dyn_cast<FloatAttr>(constant);
    if (!floatAttr) {
      return false;
    }
    return std::abs(floatAttr.getValueAsDouble() - std::numbers::pi) < 1e-12;
  }

  Block* block;
  OpBuilder builder;
  SmallVector<RegisterGroup> groups;
  llvm::DenseMap<Value, RegisterSlot> registerSlots;
  llvm::DenseMap<Value, Value> scalarRoots;
  llvm::DenseMap<Value, Value> values;
  llvm::DenseMap<Value, Value> currentScalarValues;
  llvm::DenseSet<Value> deallocatedScalarRoots;
  llvm::DenseSet<Operation*> negativeControlWrappers;
  llvm::DenseMap<Operation*, SmallVector<bool>> inferredNegativeControls;
  llvm::DenseMap<Value, Value> emptyLocal;
};

struct QCOToCatalystQuantum final
    : impl::QCOToCatalystQuantumBase<QCOToCatalystQuantum> {
  using QCOToCatalystQuantumBase::QCOToCatalystQuantumBase;

  void runOnOperation() override {
    Operation* root = getOperation();
    if (failed(preflight(root))) {
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> functions;
    root->walk([&](func::FuncOp func) {
      if (func.empty()) {
        return;
      }
      if (llvm::any_of(func.getBody().front(), [](Operation& op) {
            return isQCOOperation(&op) || isQubitBridgeCast(&op);
          })) {
        functions.push_back(func);
      }
    });

    for (func::FuncOp func : functions) {
      BlockConverter converter(func.getBody().front());
      if (failed(converter.convert())) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mqt::ir::conversions
