/*
 * Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CatalystQuantumToQC/CatalystQuantumToQC.h" // NOLINT(misc-include-cleaner)

#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numbers>
#include <utility>

namespace mqt::ir::conversions {

#define GEN_PASS_DEF_CATALYSTQUANTUMTOQC
#include "mlir/Conversion/CatalystQuantumToQC/CatalystQuantumToQC.h.inc"

using namespace mlir;
using namespace mlir::arith;

namespace {

/// Partition control qubits into positive and negative control based on their
/// control values.
/// Returns the updated control qubits and values that should be used for the
/// operation.
struct ControlPartitionResult {
  SmallVector<Value> posCtrlQubits;
  SmallVector<Value> negCtrlQubits;
};

struct ControlLists {
  SmallVector<Value> posCtrlQubits;
  SmallVector<Value> negCtrlQubits;
};

ControlLists buildControlLists(ValueRange baseControls,
                               ValueRange additionalPos,
                               ValueRange additionalNeg) {
  ControlLists lists;
  lists.posCtrlQubits.append(baseControls.begin(), baseControls.end());
  lists.posCtrlQubits.append(additionalPos.begin(), additionalPos.end());
  lists.negCtrlQubits.append(additionalNeg.begin(), additionalNeg.end());
  return lists;
}

SmallVector<Value> reorderControlledGateResults(Operation* op,
                                                size_t totalCtrlCount) {
  SmallVector<Value> reordered;
  reordered.reserve(totalCtrlCount + 1);
  for (size_t idx = 0; idx < totalCtrlCount; ++idx) {
    reordered.push_back(op->getResult(1 + idx));
  }
  reordered.push_back(op->getResult(0));
  return reordered;
}

LogicalResult partitionControlQubits(ValueRange inCtrlQubits,
                                     ValueRange inCtrlValues,
                                     ConversionPatternRewriter& rewriter,
                                     Location loc,
                                     ControlPartitionResult& result) {
  if (inCtrlQubits.size() != inCtrlValues.size()) {
    return rewriter.notifyMatchFailure(
        loc, "control qubits and control values size mismatch");
  }
  for (size_t i = 0; i < inCtrlQubits.size(); ++i) {
    bool isPosCtrl = true; // Default to positive control
    bool isConstant = false;

    // Check if control value is a compile-time constant
    if (auto constOp = inCtrlValues[i].getDefiningOp<arith::ConstantOp>()) {
      isConstant = true;
      if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue())) {
        isPosCtrl = boolAttr.getValue();
      } else if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        isPosCtrl = (intAttr.getInt() != 0);
      } else {
        return rewriter.notifyMatchFailure(
            loc, "Control value must be a boolean or integer constant");
      }
    }

    // Handle the control qubit based on whether value is constant or dynamic
    if (isConstant) {
      // Constant control value: use standard pos/neg control
      if (isPosCtrl) {
        result.posCtrlQubits.emplace_back(inCtrlQubits[i]);
      } else {
        result.negCtrlQubits.emplace_back(inCtrlQubits[i]);
      }
    } else {
      // TODO: something like if (ctrl_value == 0) { apply X }; apply_op; if
      // (ctrl_value
      // == 0) { apply X }
      rewriter.getContext()->getDiagEngine().emit(loc,
                                                  DiagnosticSeverity::Warning)
          << "Dynamic control values are not fully supported yet. Treating as "
             "positive control. Consider constant folding control values "
             "before this pass.";

      result.posCtrlQubits.emplace_back(inCtrlQubits[i]);
    }
  }
  return success();
}

struct ParameterInfo {
  SmallVector<double> staticParams;
  SmallVector<bool> paramsMask;
  SmallVector<Value> finalParams;
};

FailureOr<ParameterInfo> processParameters(catalyst::quantum::CustomOp op,
                                           ValueRange paramsValues) {
  ParameterInfo info;
  auto maskAttr = op->getAttrOfType<DenseBoolArrayAttr>("params_mask");
  auto staticParamsAttr = op->getAttrOfType<DenseF64ArrayAttr>("static_params");

  if (maskAttr && staticParamsAttr) {
    size_t staticIdx = 0;
    size_t dynamicIdx = 0;
    const int64_t maskSize = maskAttr.size();
    const size_t staticParamsCount = staticParamsAttr.size();

    for (int64_t i = 0; i < maskSize; ++i) {
      const bool isStatic = maskAttr[i];
      info.paramsMask.emplace_back(isStatic);

      if (isStatic) {
        if (staticIdx >= staticParamsCount) {
          return op.emitError("Missing static_params for static mask");
        }
        info.staticParams.emplace_back(staticParamsAttr[staticIdx++]);
      } else {
        if (dynamicIdx >= paramsValues.size()) {
          return op.emitError("Too few dynamic parameters");
        }
        info.finalParams.emplace_back(paramsValues[dynamicIdx++]);
      }
    }
  } else {
    for (auto param : paramsValues) {
      info.finalParams.push_back(param);
      info.paramsMask.push_back(false);
    }
  }
  return info;
}

} // namespace

class CatalystQuantumToQCTypeConverter final : public TypeConverter {
public:
  explicit CatalystQuantumToQCTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if needed
    addConversion([](const Type type) { return type; });

    // Convert Catalyst QubitType to QC QubitType
    addConversion([ctx](catalyst::quantum::QubitType /*type*/) -> Type {
      return qc::QubitType::get(ctx);
    });

    // Convert Catalyst QuregType to dynamic memref as placeholder
    // The actual static memref types will flow through from alloc operations
    addConversion([ctx](catalyst::quantum::QuregType /*type*/) -> Type {
      auto qubitType = qc::QubitType::get(ctx);
      return MemRefType::get(
          {ShapedType::kDynamic}, // NOLINT(misc-include-cleaner)
          qubitType);
    });

    // Target materialization: converts values during pattern application
    // Returns the input, potentially with a cast if the types don't match
    addTargetMaterialization([](OpBuilder& builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return nullptr;
      }
      if (inputs[0].getType() != resultType) {
        if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
          if (isa<MemRefType>(inputs[0].getType())) {
            return {builder.create<memref::CastOp>(loc, memrefType, inputs[0])};
          }
        }
      }
      return inputs[0];
    });
  }
};

struct ConvertQuantumAlloc final
    : OpConversionPattern<catalyst::quantum::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto nqubitsAttr = op.getNqubitsAttrAttr();

    // Prepare the result type(s)
    const auto qubitType = qc::QubitType::get(rewriter.getContext());

    if (nqubitsAttr) {
      // Static allocation
      const auto nqubits = nqubitsAttr.getValue().getZExtValue();
      const auto memrefType =
          MemRefType::get({static_cast<int64_t>(nqubits)}, qubitType);
      auto allocOp = rewriter.create<memref::AllocOp>(op.getLoc(), memrefType);
      rewriter.replaceOp(op, allocOp.getResult());
    } else if (auto nqubitsOp = op.getNqubits()) {
      // Dynamic allocation
      Value size = nqubitsOp;
      if (isa<IntegerType>(size.getType())) {
        size = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), size);
      }
      const auto memrefType =
          MemRefType::get({ShapedType::kDynamic}, qubitType);
      auto allocOp = rewriter.create<memref::AllocOp>(op.getLoc(), memrefType,
                                                      ValueRange{size});
      rewriter.replaceOp(op, allocOp.getResult());
    } else {
      return op.emitError(
          "AllocOp missing both nqubits_attr and nqubits operand");
    }

    return success();
  }
};

struct ConvertQuantumDealloc final
    : OpConversionPattern<catalyst::quantum::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s)
    Value memref = adaptor.getQreg();

    // Unwrap unrealized_conversion_cast if present
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!castOp.getInputs().empty()) {
        memref = castOp.getInputs()[0];
      }
    }

    // Create the new operation
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, memref);
    return success();
  }
};

struct ConvertQuantumMeasure final
    : OpConversionPattern<catalyst::quantum::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s)
    const auto inQubit = adaptor.getInQubit();

    // Create the new operation
    // Note: quantum.measure returns (i1, !quantum.bit)
    //       qc.measure returns i1
    auto QCOp = rewriter.create<qc::MeasureOp>(op.getLoc(), inQubit);

    // Replace with results in the correct order
    rewriter.replaceOp(op, {inQubit, QCOp.getResult()});
    return success();
  }
};

struct ConvertQuantumExtract final
    : OpConversionPattern<catalyst::quantum::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Prepare the result type(s)
    const auto qubitType = qc::QubitType::get(rewriter.getContext());

    // Get index (either from attribute or operand)
    Value indexValue;
    auto idxAttr = op.getIdxAttrAttr();

    if (idxAttr) {
      // Compile-time constant index from attribute
      const auto idx = idxAttr.getValue().getZExtValue();
      indexValue = rewriter.create<ConstantIndexOp>(op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("ExtractOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = rewriter.create<IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), idxOperand);
      } else {
        indexValue = idxOperand;
      }
    }

    // Extract operand(s)
    Value memref = adaptor.getQreg();

    // Unwrap unrealized_conversion_cast if present
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!castOp.getInputs().empty()) {
        memref = castOp.getInputs()[0];
      }
    }

    // Verify we got a memref type
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) {
      return op.emitError("Expected memref type from alloc, got: ")
             << memref.getType();
    }

    // Create the new operation
    auto loadOp = rewriter.create<memref::LoadOp>(
        op.getLoc(), qubitType, memref, ValueRange{indexValue});

    // Replace the extract operation with the loaded qubit
    rewriter.replaceOp(op, loadOp.getResult());
    return success();
  }
};

struct ConvertQuantumInsert final
    : OpConversionPattern<catalyst::quantum::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Get index (either from attribute or operand)
    Value indexValue;
    auto idxAttr = op.getIdxAttrAttr();

    if (idxAttr) {
      // Compile-time constant index from attribute
      const auto idx = idxAttr.getValue().getZExtValue();
      indexValue = rewriter.create<ConstantIndexOp>(op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("InsertOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = rewriter.create<IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), idxOperand);
      } else {
        indexValue = idxOperand;
      }
    }

    // Extract operand(s)
    Value memref = adaptor.getInQreg();

    // Unwrap unrealized_conversion_cast if present
    if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (!castOp.getInputs().empty()) {
        memref = castOp.getInputs()[0];
      }
    }

    // Create the new operation
    rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getQubit(), memref,
                                     ValueRange{indexValue});

    // In the memref model, the register is modified in-place
    rewriter.replaceOp(op, memref);
    return success();
  }
};

struct ConvertQuantumGlobalPhase final
    : OpConversionPattern<catalyst::quantum::GlobalPhaseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::GlobalPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    const auto param = adaptor.getParams();
    const auto inCtrlQubits = adaptor.getInCtrlQubits();
    const auto inCtrlValues = adaptor.getInCtrlValues();

    // Separate positive and negative control qubits
    ControlPartitionResult ctrlResult;
    if (failed(partitionControlQubits(inCtrlQubits, inCtrlValues, rewriter,
                                      op.getLoc(), ctrlResult))) {
      return failure();
    }

    const auto& inPosCtrlQubitsVec = ctrlResult.posCtrlQubits;
    const auto& inNegCtrlQubitsVec = ctrlResult.negCtrlQubits;

    // Create the parameter attributes
    const SmallVector<double> staticParamsVec;
    SmallVector<bool> paramsMaskVec;
    SmallVector<Value> finalParamValues;

    // All parameters are treated as dynamic during conversion
    // Constant folding should be done by canonicalization passes
    finalParamValues.push_back(param);
    paramsMaskVec.push_back(false);

    const auto staticParams =
        DenseF64ArrayAttr::get(rewriter.getContext(), staticParamsVec);
    const auto paramsMask =
        DenseBoolArrayAttr::get(rewriter.getContext(), paramsMaskVec);

    // Replace the original with the new operation
    rewriter.create<qc::GPhaseOp>(op.getLoc(), finalParamValues[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQuantumCustomOp final
    : OpConversionPattern<catalyst::quantum::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(catalyst::quantum::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Extract operand(s) and attribute(s)
    const auto gateName = op.getGateName();
    const auto paramsValues = adaptor.getParams();
    const auto inQubits = adaptor.getInQubits();
    const auto inCtrlQubits = adaptor.getInCtrlQubits();
    const auto inCtrlValues = adaptor.getInCtrlValues();

    // Separate positive and negative control qubits
    ControlPartitionResult ctrlResult;
    if (failed(partitionControlQubits(inCtrlQubits, inCtrlValues, rewriter,
                                      op.getLoc(), ctrlResult))) {
      return failure();
    }

    // Save controls from inCtrlQubits separately - they will be appended AFTER
    // controls from inQubits (for gates that extract controls from inQubits)
    SmallVector<Value> additionalPosCtrlQubits = ctrlResult.posCtrlQubits;
    SmallVector<Value> additionalNegCtrlQubits = ctrlResult.negCtrlQubits;

    auto paramInfoOrError = processParameters(op, paramsValues);
    if (failed(paramInfoOrError)) {
      return failure();
    }
    const auto& paramInfo = *paramInfoOrError;
    const auto& finalParamValues = paramInfo.finalParams;

    const auto staticParams =
        DenseF64ArrayAttr::get(rewriter.getContext(), paramInfo.staticParams);
    const auto paramsMask =
        DenseBoolArrayAttr::get(rewriter.getContext(), paramInfo.paramsMask);

    // Create the new operation
    Operation* qcOp = nullptr;

#define CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(GATE_TYPE)                    \
  rewriter.create<qc::GATE_TYPE##Op>(op.getLoc(), inQubits[0])

#define CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(GATE_TYPE)                    \
  rewriter.create<qc::GATE_TYPE##Op>(op.getLoc(), inQubits[0], inQubits[1])

#define CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(GATE_TYPE)                     \
  rewriter.create<qc::GATE_TYPE##Op>(op.getLoc(), inQubits[0],                 \
                                     finalParamValues[0])

#define CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(GATE_TYPE)                     \
  rewriter.create<qc::GATE_TYPE##Op>(op.getLoc(), inQubits[0], inQubits[1],    \
                                     finalParamValues[0])

    if (gateName == "Hadamard") {
      qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(H);
    } else if (gateName == "Identity") {
      qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Id);
    } else if (gateName == "PauliX") {
      qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(X);
    } else if (gateName == "PauliY") {
      qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Y);
    } else if (gateName == "PauliZ") {
      qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Z);
    } else if (gateName == "S") {
      if (op.getAdjoint()) {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Sdg);
      } else {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(S);
      }
    } else if (gateName == "T") {
      if (op.getAdjoint()) {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Tdg);
      } else {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(T);
      }
    } else if (gateName == "SX") {
      if (op.getAdjoint()) {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(SXdg);
      } else {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(SX);
      }
    } else if (gateName == "ECR") {
      qcOp = CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(ECR);
    } else if (gateName == "SWAP") {
      qcOp = CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(SWAP);
    } else if (gateName == "ISWAP") {
      qcOp = CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(iSWAP);
    } else if (gateName == "RX") {
      qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RX);
    } else if (gateName == "RY") {
      qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RY);
    } else if (gateName == "RZ") {
      qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RZ);
    } else if (gateName == "PhaseShift") {
      qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(P);
    } else if (gateName == "CRX") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1), [&]() {
            rewriter.create<qc::RXOp>(op.getLoc(), inQubits[1],
                                      finalParamValues[0]);
          });
    } else if (gateName == "CRY") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1), [&]() {
            rewriter.create<qc::RYOp>(op.getLoc(), inQubits[1],
                                      finalParamValues[0]);
          });
    } else if (gateName == "CRZ") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1), [&]() {
            rewriter.create<qc::RZOp>(op.getLoc(), inQubits[1],
                                      finalParamValues[0]);
          });
    } else if (gateName == "ControlledPhaseShift") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1), [&]() {
            rewriter.create<qc::POp>(op.getLoc(), inQubits[1],
                                     finalParamValues[0]);
          });
    } else if (gateName == "IsingXY") {
      // PennyLane IsingXY has 1 parameter (phi), OpenQASM XXPlusYY needs 2
      // (theta, beta) Relationship: IsingXY(phi) = XXPlusYY(phi, pi)
      // Add pi as second static parameter (since we add it during compilation)
      SmallVector<double> isingxyStaticParams(paramInfo.staticParams.begin(),
                                              paramInfo.staticParams.end());
      isingxyStaticParams.push_back(std::numbers::pi);

      SmallVector<bool> isingxyParamsMask(paramInfo.paramsMask.begin(),
                                          paramInfo.paramsMask.end());
      isingxyParamsMask.push_back(true); // pi is a compile-time constant

      auto isingxyStaticParamsAttr =
          DenseF64ArrayAttr::get(rewriter.getContext(), isingxyStaticParams);
      auto isingxyParamsMaskAttr =
          DenseBoolArrayAttr::get(rewriter.getContext(), isingxyParamsMask);

      qcOp = rewriter.create<qc::XXPlusYYOp>(op.getLoc(), inQubits[0],
                                             inQubits[1], finalParamValues[0],
                                             isingxyStaticParamsAttr[0]);
    } else if (gateName == "IsingXX") {
      qcOp = CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(RXX);
    } else if (gateName == "IsingYY") {
      qcOp = CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(RYY);
    } else if (gateName == "IsingZZ") {
      qcOp = CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(RZZ);
    } else if (gateName == "CNOT") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1),
          [&]() { rewriter.create<qc::XOp>(op.getLoc(), inQubits[1]); });
    } else if (gateName == "CY") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1),
          [&]() { rewriter.create<qc::YOp>(op.getLoc(), inQubits[1]); });
    } else if (gateName == "CZ") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1),
          [&]() { rewriter.create<qc::ZOp>(op.getLoc(), inQubits[1]); });
    } else if (gateName == "Toffoli") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(2),
          [&]() { rewriter.create<qc::XOp>(op.getLoc(), inQubits[2]); });
    } else if (gateName == "CSWAP") {
      qcOp = rewriter.create<qc::CtrlOp>(
          op.getLoc(), inQubits.take_front(1), [&]() {
            rewriter.create<qc::SWAPOp>(op.getLoc(), inQubits[1], inQubits[2]);
          });
    } else {
      return op.emitError("Unsupported gate: ") << gateName;
    }

#undef CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP
#undef CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP
#undef CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP
#undef CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP

    // Replace the original with the new operation
    rewriter.replaceOp(op, inQubits);
    return success();
  }
};

struct CatalystQuantumToQC final
    : impl::CatalystQuantumToQCBase<CatalystQuantumToQC> {
  using CatalystQuantumToQCBase::CatalystQuantumToQCBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<qc::QCDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<catalyst::quantum::QuantumDialect>();

    // Mark operations legal that have no equivalent in the target dialect
    target.addLegalOp<
        catalyst::quantum::DeviceInitOp, catalyst::quantum::DeviceReleaseOp,
        catalyst::quantum::NamedObsOp, catalyst::quantum::ExpvalOp,
        catalyst::quantum::FinalizeOp, catalyst::quantum::ComputationalBasisOp,
        catalyst::quantum::StateOp, catalyst::quantum::InitializeOp>();

    const CatalystQuantumToQCTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);

    patterns
        .add<ConvertQuantumAlloc, ConvertQuantumDealloc, ConvertQuantumMeasure,
             ConvertQuantumExtract, ConvertQuantumInsert,
             ConvertQuantumGlobalPhase, ConvertQuantumCustomOp>(typeConverter,
                                                                context);

    // Type conversion boilerplate to handle function signatures and control
    // flow See: https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion

    // Convert func.func signatures to use the converted types
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    // Mark func.func as legal only if signature and body types are converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation* op) {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        return typeConverter.isSignatureLegal(funcOp.getFunctionType()) &&
               typeConverter.isLegal(&funcOp.getBody());
      }
      return true; // Not a FuncOp, treat as legal (not our concern)
    });

    // Convert return ops to match the new function result types
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Mark func.return as legal only if operand types match converted types
    target.addDynamicallyLegalOp<func::ReturnOp>([&](Operation* op) {
      if (isa<func::ReturnOp>(op)) {
        return typeConverter.isLegal(op);
      }
      return true;
    });

    // Convert call sites to use the converted argument and result types
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // Mark func.call as legal only if operand and result types are converted
    target.addDynamicallyLegalOp<func::CallOp>([&](Operation* op) {
      if (isa<func::CallOp>(op)) {
        return typeConverter.isLegal(op);
      }
      return true;
    });

    // Convert control-flow ops (cf.br, cf.cond_br, etc.)
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // Mark unknown ops as legal if they don't require type conversion
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::conversions
