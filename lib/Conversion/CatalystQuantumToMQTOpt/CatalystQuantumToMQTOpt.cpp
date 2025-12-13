/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

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
#include <mlir/IR/BuiltinTypes.h>
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

#define GEN_PASS_DEF_CATALYSTQUANTUMTOMQTOPT
#include "mlir/Conversion/CatalystQuantumToMQTOpt/CatalystQuantumToMQTOpt.h.inc"

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

static ControlLists buildControlLists(ValueRange baseControls,
                                      ValueRange additionalPos,
                                      ValueRange additionalNeg) {
  ControlLists lists;
  lists.posCtrlQubits.append(baseControls.begin(), baseControls.end());
  lists.posCtrlQubits.append(additionalPos.begin(), additionalPos.end());
  lists.negCtrlQubits.append(additionalNeg.begin(), additionalNeg.end());
  return lists;
}

static SmallVector<Value> reorderControlledGateResults(Operation* op,
                                                       size_t totalCtrlCount) {
  SmallVector<Value> reordered;
  reordered.reserve(totalCtrlCount + 1);
  for (size_t idx = 0; idx < totalCtrlCount; ++idx) {
    reordered.push_back(op->getResult(1 + idx));
  }
  reordered.push_back(op->getResult(0));
  return reordered;
}

static LogicalResult partitionControlQubits(ValueRange inCtrlQubits,
                                            ValueRange inCtrlValues,
                                            ConversionPatternRewriter& rewriter,
                                            Location loc,
                                            ControlPartitionResult& result) {
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

} // namespace

class CatalystQuantumToMQTOptTypeConverter final : public TypeConverter {
public:
  explicit CatalystQuantumToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion: Allow all types to pass through unmodified if needed
    addConversion([](const Type type) { return type; });

    // Convert Catalyst QubitType to MQTOpt QubitType
    addConversion([ctx](catalyst::quantum::QubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });

    // Convert Catalyst QuregType to dynamic memref as placeholder
    // The actual static memref types will flow through from alloc operations
    addConversion([ctx](catalyst::quantum::QuregType /*type*/) -> Type {
      auto qubitType = opt::QubitType::get(ctx);
      return MemRefType::get({ShapedType::kDynamic}, qubitType);
    });

    // Target materialization: converts values during pattern application
    // Just returns the input - the actual memref is already created by alloc
    addTargetMaterialization([]([[maybe_unused]] OpBuilder& builder,
                                [[maybe_unused]] Type resultType,
                                ValueRange inputs,
                                [[maybe_unused]] Location loc) -> Value {
      if (inputs.size() == 1) {
        return inputs[0];
      }
      return nullptr;
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
    const auto qubitType = opt::QubitType::get(rewriter.getContext());

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

    // Prepare the result type(s)
    const auto qubitType = opt::QubitType::get(rewriter.getContext());
    const auto bitType = rewriter.getI1Type();

    // Create the new operation
    // Note: quantum.measure returns (i1, !quantum.bit)
    //       mqtopt.measure returns (!mqtopt.Qubit, i1)
    auto mqtoptOp = rewriter.create<opt::MeasureOp>(op.getLoc(), qubitType,
                                                    bitType, inQubit);

    // Replace with results in the correct order
    rewriter.replaceOp(op, {mqtoptOp.getResult(1), mqtoptOp.getResult(0)});
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
    const auto qubitType = opt::QubitType::get(rewriter.getContext());

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
    SmallVector<double> staticParamsVec;
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

    // Create the new operation
    auto mqtoptOp = rewriter.create<opt::GPhaseOp>(
        op.getLoc(), TypeRange{},                   // out_qubits
        ValueRange(inPosCtrlQubitsVec).getTypes(),  // pos_ctrl_out_qubits
        ValueRange(inNegCtrlQubitsVec).getTypes(),  // neg_ctrl_out_qubits
        staticParams, paramsMask, finalParamValues, // params
        ValueRange{},                               // in_qubits
        ValueRange(inPosCtrlQubitsVec),             // pos_ctrl_in_qubits
        ValueRange(inNegCtrlQubitsVec));            // neg_ctrl_in_qubits

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
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

    // Process parameters (static vs dynamic)
    SmallVector<bool> paramsMaskVec;
    SmallVector<double> staticParamsVec;
    SmallVector<Value> finalParamValues;

    auto maskAttr = op->getAttrOfType<DenseBoolArrayAttr>("params_mask");
    auto staticParamsAttr =
        op->getAttrOfType<DenseF64ArrayAttr>("static_params");

    if (maskAttr && staticParamsAttr) {
      // Preserve existing mask and static params from the op
      size_t staticIdx = 0;
      size_t dynamicIdx = 0;

      for (int64_t i = 0; i < static_cast<int64_t>(maskAttr.size()); ++i) {
        const bool isStatic = maskAttr[i];
        paramsMaskVec.emplace_back(isStatic);

        if (isStatic) {
          if (static_cast<int64_t>(staticIdx) >=
              static_cast<int64_t>(staticParamsAttr.size())) {
            return op.emitError("Missing static_params for static mask");
          }
          staticParamsVec.emplace_back(staticParamsAttr[staticIdx++]);
        } else {
          if (dynamicIdx >= paramsValues.size()) {
            return op.emitError("Too few dynamic parameters");
          }
          finalParamValues.emplace_back(paramsValues[dynamicIdx++]);
        }
      }
    } else {
      // All parameters are treated as dynamic during conversion
      // Constant folding should be done by canonicalization passes
      for (auto param : paramsValues) {
        finalParamValues.push_back(param);
        paramsMaskVec.push_back(false);
      }
    }

    const auto staticParams =
        DenseF64ArrayAttr::get(rewriter.getContext(), staticParamsVec);
    const auto paramsMask =
        DenseBoolArrayAttr::get(rewriter.getContext(), paramsMaskVec);

    // Create the new operation
    Operation* mqtoptOp = nullptr;

#define CREATE_GATE_OP(GATE_TYPE)                                              \
  rewriter.create<opt::GATE_TYPE##Op>(                                         \
      op.getLoc(), inQubits.getTypes(),                                        \
      ValueRange(additionalPosCtrlQubits).getTypes(),                          \
      ValueRange(additionalNegCtrlQubits).getTypes(), staticParams,            \
      paramsMask, finalParamValues, inQubits, additionalPosCtrlQubits,         \
      additionalNegCtrlQubits)

    if (gateName == "Hadamard") {
      mqtoptOp = CREATE_GATE_OP(H);
    } else if (gateName == "Identity") {
      mqtoptOp = CREATE_GATE_OP(I);
    } else if (gateName == "PauliX") {
      mqtoptOp = CREATE_GATE_OP(X);
    } else if (gateName == "PauliY") {
      mqtoptOp = CREATE_GATE_OP(Y);
    } else if (gateName == "PauliZ") {
      mqtoptOp = CREATE_GATE_OP(Z);
    } else if (gateName == "S") {
      if (op.getAdjoint()) {
        mqtoptOp = CREATE_GATE_OP(Sdg);
      } else {
        mqtoptOp = CREATE_GATE_OP(S);
      }
    } else if (gateName == "T") {
      if (op.getAdjoint()) {
        mqtoptOp = CREATE_GATE_OP(Tdg);
      } else {
        mqtoptOp = CREATE_GATE_OP(T);
      }
    } else if (gateName == "SX") {
      if (op.getAdjoint()) {
        mqtoptOp = CREATE_GATE_OP(SXdg);
      } else {
        mqtoptOp = CREATE_GATE_OP(SX);
      }
    } else if (gateName == "ECR") {
      mqtoptOp = CREATE_GATE_OP(ECR);
    } else if (gateName == "SWAP") {
      mqtoptOp = CREATE_GATE_OP(SWAP);
    } else if (gateName == "ISWAP") {
      if (op.getAdjoint()) {
        mqtoptOp = CREATE_GATE_OP(iSWAPdg);
      } else {
        mqtoptOp = CREATE_GATE_OP(iSWAP);
      }
    } else if (gateName == "RX") {
      mqtoptOp = CREATE_GATE_OP(RX);
    } else if (gateName == "RY") {
      mqtoptOp = CREATE_GATE_OP(RY);
    } else if (gateName == "RZ") {
      mqtoptOp = CREATE_GATE_OP(RZ);
    } else if (gateName == "PhaseShift") {
      mqtoptOp = CREATE_GATE_OP(P);
    } else if (gateName == "CRX") {
      auto controlLists = buildControlLists(
          inQubits.take_front(1), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto crxOp = rewriter.create<opt::RXOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[1], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op, reorderControlledGateResults(crxOp, ctrlCount));
      return success();
    } else if (gateName == "CRY") {
      auto controlLists = buildControlLists(
          inQubits.take_front(1), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto cryOp = rewriter.create<opt::RYOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[1], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op, reorderControlledGateResults(cryOp, ctrlCount));
      return success();
    } else if (gateName == "CRZ") {
      auto controlLists = buildControlLists(
          inQubits.take_front(1), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto crzOp = rewriter.create<opt::RZOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[1], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op, reorderControlledGateResults(crzOp, ctrlCount));
      return success();
    } else if (gateName == "ControlledPhaseShift") {
      auto controlLists = buildControlLists(
          inQubits.take_front(1), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto cpOp = rewriter.create<opt::POp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[1], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op, reorderControlledGateResults(cpOp, ctrlCount));
      return success();
    } else if (gateName == "IsingXY") {
      // PennyLane IsingXY has 1 parameter (phi), OpenQASM XXPlusYY needs 2
      // (theta, beta) Relationship: IsingXY(phi) = XXPlusYY(phi, pi)
      // Add pi as second static parameter (since we add it during compilation)
      SmallVector<double> isingxyStaticParams(staticParamsVec.begin(),
                                              staticParamsVec.end());
      isingxyStaticParams.push_back(std::numbers::pi);

      SmallVector<bool> isingxyParamsMask(paramsMaskVec.begin(),
                                          paramsMaskVec.end());
      isingxyParamsMask.push_back(true); // pi is a compile-time constant

      auto isingxyStaticParamsAttr =
          DenseF64ArrayAttr::get(rewriter.getContext(), isingxyStaticParams);
      auto isingxyParamsMaskAttr =
          DenseBoolArrayAttr::get(rewriter.getContext(), isingxyParamsMask);

      mqtoptOp = rewriter.create<opt::XXplusYYOp>(
          op.getLoc(), inQubits.getTypes(),
          ValueRange(additionalPosCtrlQubits).getTypes(),
          ValueRange(additionalNegCtrlQubits).getTypes(),
          isingxyStaticParamsAttr, isingxyParamsMaskAttr, finalParamValues,
          inQubits, additionalPosCtrlQubits, additionalNegCtrlQubits);
    } else if (gateName == "IsingXX") {
      mqtoptOp = CREATE_GATE_OP(RXX);
    } else if (gateName == "IsingYY") {
      mqtoptOp = CREATE_GATE_OP(RYY);
    } else if (gateName == "IsingZZ") {
      mqtoptOp = CREATE_GATE_OP(RZZ);
    } else if (gateName == "CNOT") {
      auto controlLists = buildControlLists(
          inQubits.take_front(1), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto cnotOp = rewriter.create<opt::XOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[1], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op, reorderControlledGateResults(cnotOp, ctrlCount));
      return success();
    } else if (gateName == "CY") {
      auto controlLists = buildControlLists(
          inQubits.take_front(1), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto cyOp = rewriter.create<opt::YOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[1], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op, reorderControlledGateResults(cyOp, ctrlCount));
      return success();
    } else if (gateName == "CZ") {
      auto controlLists = buildControlLists(
          inQubits.take_front(1), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto czOp = rewriter.create<opt::ZOp>(
          op.getLoc(), inQubits[1].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[1], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op, reorderControlledGateResults(czOp, ctrlCount));
      return success();
    } else if (gateName == "Toffoli") {
      auto controlLists = buildControlLists(
          inQubits.take_front(2), ValueRange(additionalPosCtrlQubits),
          ValueRange(additionalNegCtrlQubits));
      auto toffoliOp = rewriter.create<opt::XOp>(
          op.getLoc(), inQubits[2].getType(),
          ValueRange(controlLists.posCtrlQubits).getTypes(),
          ValueRange(controlLists.negCtrlQubits).getTypes(), staticParams,
          paramsMask, finalParamValues, inQubits[2], controlLists.posCtrlQubits,
          controlLists.negCtrlQubits);
      const size_t ctrlCount =
          controlLists.posCtrlQubits.size() + controlLists.negCtrlQubits.size();
      rewriter.replaceOp(op,
                         reorderControlledGateResults(toffoliOp, ctrlCount));
      return success();
    } else if (gateName == "CSWAP") {
      // CSWAP gate: 1 control qubit + 2 target qubits
      // inQubits[0] is control, inQubits[1] and inQubits[2] are targets
      SmallVector<Value> ctrls = {inQubits[0]};
      ctrls.append(additionalPosCtrlQubits.begin(),
                   additionalPosCtrlQubits.end());
      SmallVector<Value> negCtrls(additionalNegCtrlQubits.begin(),
                                  additionalNegCtrlQubits.end());
      auto cswapOp = rewriter.create<opt::SWAPOp>(
          op.getLoc(), ValueRange{inQubits[1], inQubits[2]},
          ValueRange(ctrls).getTypes(), ValueRange(negCtrls).getTypes(),
          staticParams, paramsMask, finalParamValues,
          ValueRange{inQubits[1], inQubits[2]}, ctrls, negCtrls);
      // MQTOpt SWAP returns (target0, target1, control) but Catalyst CSWAP
      // expects (control, target0, target1) Need to reorder: [2, 0, 1] (control
      // goes first)
      rewriter.replaceOp(op, {cswapOp.getResult(2), cswapOp.getResult(0),
                              cswapOp.getResult(1)});
      return success();
    } else {
      return op.emitError("Unsupported gate: ") << gateName;
    }

#undef CREATE_GATE_OP

    // Replace the original with the new operation
    rewriter.replaceOp(op, mqtoptOp);
    return success();
  }
};

struct CatalystQuantumToMQTOpt final
    : impl::CatalystQuantumToMQTOptBase<CatalystQuantumToMQTOpt> {
  using CatalystQuantumToMQTOptBase::CatalystQuantumToMQTOptBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<opt::MQTOptDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<catalyst::quantum::QuantumDialect>();

    // Mark operations legal that have no equivalent in the target dialect
    target.addLegalOp<
        catalyst::quantum::DeviceInitOp, catalyst::quantum::DeviceReleaseOp,
        catalyst::quantum::NamedObsOp, catalyst::quantum::ExpvalOp,
        catalyst::quantum::FinalizeOp, catalyst::quantum::ComputationalBasisOp,
        catalyst::quantum::StateOp, catalyst::quantum::InitializeOp>();

    const CatalystQuantumToMQTOptTypeConverter typeConverter(context);
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
