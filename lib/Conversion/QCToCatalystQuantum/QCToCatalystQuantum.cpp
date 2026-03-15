/*
 * Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToCatalystQuantum/QCToCatalystQuantum.h" // NOLINT(misc-include-cleaner)

#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <cstddef>
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

#define GEN_PASS_DEF_QCTOCATALYSTQUANTUM
#include "mlir/Conversion/QCToCatalystQuantum/QCToCatalystQuantum.h.inc"

using namespace mlir;
using namespace mlir::arith;

// Helper functions to reduce code duplication
namespace {

/// Helper struct to hold control qubit information
struct ControlInfo {
  SmallVector<Value> ctrlQubits;
  SmallVector<Value> ctrlValues;

  ControlInfo() noexcept = default;
};

/// Extract and concatenate control qubits and create corresponding control
/// values
ControlInfo extractControlInfo(ValueRange posCtrlQubits,
                               ValueRange negCtrlQubits,
                               ConversionPatternRewriter& rewriter,
                               Location loc) {
  ControlInfo info;

  // Concatenate controls: [pos..., neg...]  (preserve this order consistently)
  info.ctrlQubits.reserve(posCtrlQubits.size() + negCtrlQubits.size());
  info.ctrlQubits.append(posCtrlQubits.begin(), posCtrlQubits.end());
  info.ctrlQubits.append(negCtrlQubits.begin(), negCtrlQubits.end());

  if (info.ctrlQubits.empty()) {
    return info;
  }

  // Create control values: 1 for positive controls, 0 for negative controls
  const Value one =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, /*value=*/1,
                                                  /*width=*/1);
  const Value zero =
      rewriter.create<mlir::arith::ConstantIntOp>(loc, /*value=*/0,
                                                  /*width=*/1);

  info.ctrlValues.reserve(info.ctrlQubits.size());
  info.ctrlValues.append(posCtrlQubits.size(), one);  // +controls => 1
  info.ctrlValues.append(negCtrlQubits.size(), zero); // -controls => 0

  return info;
}

/// Helper function to extract operands and control info - for more complex
/// cases
template <typename OpAdaptor> struct ExtractedOperands {
  ValueRange inQubits;
  ControlInfo ctrlInfo;
};

template <typename OpAdaptor>
ExtractedOperands<OpAdaptor>
extractOperands(OpAdaptor adaptor, ConversionPatternRewriter& rewriter,
                Location loc) {
  const ValueRange inQubits = adaptor.getInQubits();
  const ValueRange posCtrlQubits = adaptor.getPosCtrlInQubits();
  const ValueRange negCtrlQubits = adaptor.getNegCtrlInQubits();

  const ControlInfo ctrlInfo =
      extractControlInfo(posCtrlQubits, negCtrlQubits, rewriter, loc);

  return {inQubits, ctrlInfo};
}

} // anonymous namespace

class QCToCatalystQuantumTypeConverter final : public TypeConverter {
public:
  explicit QCToCatalystQuantumTypeConverter(MLIRContext* ctx) {
    // Identity conversion for types that don't need transformation
    addConversion([](const Type type) { return type; });

    // Convert MemRef of QC QubitType to Catalyst QuregType
    // Also handles memrefs where the element type was already converted
    addConversion([ctx](MemRefType memrefType) -> Type {
      auto elemType = memrefType.getElementType();
      if (isa<qc::QubitType>(elemType) ||
          isa<catalyst::quantum::QubitType>(elemType)) {
        return catalyst::quantum::QuregType::get(ctx);
      }
      return memrefType;
    });

    // Convert QC QubitType to Catalyst QubitType
    addConversion([ctx](qc::QubitType /*type*/) -> Type {
      return catalyst::quantum::QubitType::get(ctx);
    });
  }
};

struct ConvertQCAlloc final : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert memrefs of qubit type
    auto memrefType = dyn_cast<BaseMemRefType>(op.getType());
    auto elemType = memrefType ? memrefType.getElementType() : Type();
    if (!memrefType || !(isa<qc::QubitType>(elemType) ||
                         isa<catalyst::quantum::QubitType>(elemType))) {
      return failure();
    }

    // Only handle ranked memrefs
    auto rankedMemrefType = dyn_cast<MemRefType>(memrefType);
    if (!rankedMemrefType) {
      return failure();
    }

    // Prepare the result type(s)
    const auto resultType =
        catalyst::quantum::QuregType::get(rewriter.getContext());

    // Get the size from memref type or dynamic operands
    Value size = nullptr;
    mlir::IntegerAttr nqubitsAttr = nullptr;

    // Check if this is a statically shaped memref
    if (rankedMemrefType.hasStaticShape() &&
        rankedMemrefType.getNumElements() >= 0) {
      // For static memref: use attribute (no operand)
      nqubitsAttr =
          rewriter.getI64IntegerAttr(rankedMemrefType.getNumElements());
    } else {
      // For dynamic memref: check if the size is actually a constant
      auto dynamicOperands = op.getDynamicSizes();
      const Value dynamicSize =
          dynamicOperands.empty() ? nullptr : dynamicOperands[0];

      if (dynamicSize) {
        // Try to recover static size from constant operand
        if (auto constOp =
                dynamicSize.getDefiningOp<arith::ConstantIndexOp>()) {
          // The size is a constant index, use it as an attribute instead
          nqubitsAttr = rewriter.getI64IntegerAttr(constOp.value());
        } else if (auto constOp =
                       dynamicSize.getDefiningOp<arith::ConstantIntOp>()) {
          // The size is a constant int, use it as an attribute instead
          nqubitsAttr = rewriter.getI64IntegerAttr(constOp.value());
        } else {
          // Truly dynamic size - use operand
          size = dynamicSize;
          // quantum.alloc expects i64, but memref size is index type
          if (mlir::isa<IndexType>(size.getType())) {
            size = rewriter.create<arith::IndexCastOp>(
                op.getLoc(), rewriter.getI64Type(), size);
          }
        }
      }
    }

    // Replace with quantum alloc operation
    rewriter.replaceOpWithNewOp<catalyst::quantum::AllocOp>(op, resultType,
                                                            size, nqubitsAttr);

    return success();
  }
};

struct ConvertQCDealloc final : OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert memrefs of qubit type
    auto memrefType = dyn_cast<BaseMemRefType>(op.getMemref().getType());
    auto elemType = memrefType ? memrefType.getElementType() : Type();
    if (!memrefType || !(isa<qc::QubitType>(elemType) ||
                         isa<catalyst::quantum::QubitType>(elemType))) {
      return failure();
    }

    // Create the new operation
    const auto catalystOp = rewriter.create<catalyst::quantum::DeallocOp>(
        op.getLoc(), TypeRange({}), adaptor.getMemref());

    // Replace the original with the new operation
    rewriter.replaceOp(op, catalystOp);
    return success();
  }
};

struct ConvertQCMeasure final : OpConversionPattern<qc::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Extract operand(s)
    auto inQubit = adaptor.getQubit();

    // Prepare the result type(s)
    auto qubitType = catalyst::quantum::QubitType::get(rewriter.getContext());
    auto bitType = rewriter.getI1Type();

    // Create the new operation
    const auto catalystOp = rewriter.create<catalyst::quantum::MeasureOp>(
        op.getLoc(), bitType, qubitType, inQubit,
        /*optional::mlir::IntegerAttr postselect=*/nullptr);

    // Replace all uses of both results and then erase the operation
    const auto catalystMeasure = catalystOp->getResult(0);
    const auto catalystQubit = catalystOp->getResult(1);
    rewriter.replaceOp(op, ValueRange{catalystQubit, catalystMeasure});
    return success();
  }
};

struct ConvertQCLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert loads of qubit type
    if (!(isa<qc::QubitType>(op.getType()) ||
          isa<catalyst::quantum::QubitType>(op.getType()))) {
      return failure();
    }

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QubitType::get(rewriter.getContext());

    // Get index (assuming single index for 1D memref)
    auto indices = adaptor.getIndices();
    Value index = indices.empty() ? nullptr : indices[0];

    // Convert index type to i64 if needed
    if (index && mlir::isa<IndexType>(index.getType())) {
      index = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                  rewriter.getI64Type(), index);
    }

    // Create the new operation
    auto catalystOp = rewriter.create<catalyst::quantum::ExtractOp>(
        op.getLoc(), resultType, adaptor.getMemref(), index, nullptr);

    // Replace the load operation with the extracted qubit
    rewriter.replaceOp(op, catalystOp.getResult());
    return success();
  }
};

struct ConvertQCStore final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert stores to memrefs with qubit element type
    auto memrefType = dyn_cast<BaseMemRefType>(op.getMemRef().getType());
    auto elemType = memrefType ? memrefType.getElementType() : Type();
    if (!memrefType || !(isa<qc::QubitType>(elemType) ||
                         isa<catalyst::quantum::QubitType>(elemType))) {
      return failure();
    }

    // Get indices (assuming single index for 1D memref)
    auto indices = adaptor.getIndices();
    Value index = indices.empty() ? nullptr : indices[0];

    // Convert index type to i64 if needed
    if (index && mlir::isa<IndexType>(index.getType())) {
      index = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                  rewriter.getI64Type(), index);
    }

    // Prepare the result type(s)
    auto resultType = catalyst::quantum::QuregType::get(rewriter.getContext());

    // Create the new operation
    rewriter.create<catalyst::quantum::InsertOp>(op.getLoc(), resultType,
                                                 adaptor.getMemref(), index,
                                                 nullptr, adaptor.getValue());

    // Erase the original store operation (store has no results to replace)
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertQCCast final : OpConversionPattern<memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert if it's a cast between qubit memrefs
    auto srcType = dyn_cast<BaseMemRefType>(op.getSource().getType());
    auto dstType = dyn_cast<BaseMemRefType>(op.getType());
    auto srcElem = srcType ? srcType.getElementType() : Type();
    auto dstElem = dstType ? dstType.getElementType() : Type();

    if (!srcType || !dstType ||
        !(isa<qc::QubitType>(srcElem) ||
          isa<catalyst::quantum::QubitType>(srcElem)) ||
        !(isa<qc::QubitType>(dstElem) ||
          isa<catalyst::quantum::QubitType>(dstElem))) {
      return failure();
    }

    // Both should convert to !quantum.reg
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

} // namespace mqt::ir::conversions
