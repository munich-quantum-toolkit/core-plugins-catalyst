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
#include "mlir/Dialect/QC/IR/QCOps.h"

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

/**
 * @brief Type converter for QC-to-Quantum conversion
 *
 * @details
 * Handles type conversion between the QC and Quantum dialects.
 * The primary conversion is from !qc.qubit to !quantum.qubit as well as memref
 * to !quantum.reg, which represents the semantic shift from reference types to
 * value types.
 *
 * Other types (integers, booleans, etc.) pass through unchanged via
 * the identity conversion.
 */
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

/**
 * @brief Converts memref.alloc to quantum.alloc
 *
 * @par Example:
 * ```mlir
 * %memref = memref.alloc(%c3) : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %qreg = quantum.alloc(%c3) : !quantum.reg
 * ```
 */
struct ConvertMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // Only convert memrefs of qubit type
    auto memrefType = dyn_cast<BaseMemRefType>(op.getType());
    auto elemType = memrefType ? memrefType.getElementType() : Type();
    if (!memrefType ||
        !llvm::isa<qc::QubitType>(op.getType().getElementType()) ||
        llvm::isa<catalyst::quantum::QubitType>(
            op.getType().getElementType())) {
      return failure();
    }

    // Only handle ranked memrefs
    auto rankedMemrefType = dyn_cast<MemRefType>(memrefType);
    if (!rankedMemrefType) {
      return failure();
    }
  }

  // Prepare the result type(s)
  const auto resultType =
      catalyst::quantum::QuregType::get(rewriter.getContext());
  if (shape[0] == ShapedType::kDynamic) {
    qtensor = rewriter.replaceOpWithNewOp<qtensor::AllocOp>(
        op, adaptor.getDynamicSizes()[0]);
  } else {
    auto size = arith::ConstantIndexOp::create(rewriter, op.getLoc(), shape[0]);
    qtensor =
        rewriter.replaceOpWithNewOp<qtensor::AllocOp>(op, size.getResult());
  }

  auto& state = getState();
  auto memref = op.getResult();
  assignMappedTensor(state, qtensor.getDefiningOp(), memref, qtensor);

  return success();
}
};

} // namespace mqt::ir::conversions
