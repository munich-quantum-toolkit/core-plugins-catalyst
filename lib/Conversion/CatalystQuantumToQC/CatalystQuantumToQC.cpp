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
#include "mlir/Dialect/QC/IR/QCOps.h"

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
            return {
                memref::CastOp::create(builder, loc, memrefType, inputs[0])};
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
      auto allocOp = memref::AllocOp::create(rewriter, op.getLoc(), memrefType);
      rewriter.replaceOp(op, allocOp.getResult());
    } else if (auto nqubitsOp = op.getNqubits()) {
      // Dynamic allocation
      Value size = nqubitsOp;
      if (isa<IntegerType>(size.getType())) {
        size = arith::IndexCastOp::create(rewriter, op.getLoc(),
                                          rewriter.getIndexType(), size);
      }
      const auto memrefType =
          MemRefType::get({ShapedType::kDynamic}, qubitType);
      auto allocOp = memref::AllocOp::create(rewriter, op.getLoc(), memrefType,
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
    auto QCOp = qc::MeasureOp::create(rewriter, op.getLoc(), inQubit);

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
      indexValue = ConstantIndexOp::create(rewriter, op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("ExtractOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = IndexCastOp::create(rewriter, op.getLoc(),
                                         rewriter.getIndexType(), idxOperand);
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
    auto loadOp = memref::LoadOp::create(rewriter, op.getLoc(), qubitType,
                                         memref, ValueRange{indexValue});

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
      indexValue = ConstantIndexOp::create(rewriter, op.getLoc(), idx);
    } else {
      // Runtime dynamic index from operand
      auto idxOperand = adaptor.getIdx();
      if (!idxOperand) {
        return op.emitError("InsertOp missing both idx_attr and idx operand");
      }

      // Convert i64 to index type if needed
      if (isa<IntegerType>(idxOperand.getType())) {
        indexValue = IndexCastOp::create(rewriter, op.getLoc(),
                                         rewriter.getIndexType(), idxOperand);
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
    memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getQubit(), memref,
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

    // Replace the original with the new operation
    qc::GPhaseOp::create(rewriter, op.getLoc(), param);
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

    // llvm::errs() << "DEBUG: Size of inQubits: " << inQubits.size() << "\n";
    // llvm::errs() << "DEBUG: Size of inCtrlQubits: " << inCtrlQubits.size() <<
    // "\n";

    // Create the new operation
    Operation* qcOp = nullptr;

#define CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(GATE_TYPE)                    \
  qc::GATE_TYPE##Op::create(rewriter, op.getLoc(), inQubits[0])

#define CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(GATE_TYPE)                    \
  qc::GATE_TYPE##Op::create(rewriter, op.getLoc(), inQubits[0], inQubits[1])

#define CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(GATE_TYPE)                     \
  qc::GATE_TYPE##Op::create(rewriter, op.getLoc(), inQubits[0], paramsValues[0])

#define CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(GATE_TYPE)                     \
  qc::GATE_TYPE##Op::create(rewriter, op.getLoc(), inQubits[0], inQubits[1],   \
                            paramsValues[0])

    if (gateName == "Hadamard") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(H);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(H);
        });
      }
    } else if (gateName == "Identity") {
      qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Id);
    } else if (gateName == "PauliX") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(X);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(X);
        });
      }
    } else if (gateName == "PauliY") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Y);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Y);
        });
      }
    } else if (gateName == "PauliZ") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Z);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Z);
        });
      }
    } else if (gateName == "S") {
      if (op.getAdjoint()) {
        if (inCtrlQubits.empty()) {
          qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Sdg);
        } else {
          qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
            CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Sdg);
          });
        }
      } else {
        if (inCtrlQubits.empty()) {
          qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(S);
        } else {
          qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
            CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(S);
          });
        }
      }
    } else if (gateName == "T") {
      if (op.getAdjoint()) {
        if (inCtrlQubits.empty()) {
          qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Tdg);
        } else {
          qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
            CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(Tdg);
          });
        }
      } else {
        if (inCtrlQubits.empty()) {
          qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(T);
        } else {
          qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
            CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(T);
          });
        }
      }
    } else if (gateName == "SX") {
      if (op.getAdjoint()) {
        if (inCtrlQubits.empty()) {
          qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(SXdg);
        } else {
          qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
            CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(SXdg);
          });
        }
      } else {
        if (inCtrlQubits.empty()) {
          qcOp = CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(SX);
        } else {
          qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
            CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP(SX);
          });
        }
      }
    } else if (gateName == "ECR") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(ECR);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(ECR);
        });
      }
    } else if (gateName == "SWAP") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(SWAP);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(SWAP);
        });
      }
    } else if (gateName == "ISWAP") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(iSWAP);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP(iSWAP);
        });
      }
    } else if (gateName == "RX") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RX);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RX);
        });
      }
    } else if (gateName == "RY") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RY);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RY);
        });
      }
    } else if (gateName == "RZ") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RZ);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(RZ);
        });
      }
    } else if (gateName == "PhaseShift") {
      if (inCtrlQubits.empty()) {
        qcOp = CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(P);
      } else {
        qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inCtrlQubits, [&]() {
          CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP(P);
        });
      }
    } else if (gateName == "CRX") {
      qcOp = qc::CtrlOp::create(
          rewriter, op.getLoc(), inQubits.take_front(1), [&]() {
            qc::RXOp::create(rewriter, op.getLoc(), inQubits[1],
                             paramsValues[0]);
          });
    } else if (gateName == "CRY") {
      qcOp = qc::CtrlOp::create(
          rewriter, op.getLoc(), inQubits.take_front(1), [&]() {
            qc::RYOp::create(rewriter, op.getLoc(), inQubits[1],
                             paramsValues[0]);
          });
    } else if (gateName == "CRZ") {
      qcOp = qc::CtrlOp::create(
          rewriter, op.getLoc(), inQubits.take_front(1), [&]() {
            qc::RZOp::create(rewriter, op.getLoc(), inQubits[1],
                             paramsValues[0]);
          });
    } else if (gateName == "ControlledPhaseShift") {
      qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inQubits.take_front(1),
                                [&]() {
                                  qc::POp::create(rewriter, op.getLoc(),
                                                  inQubits[1], paramsValues[0]);
                                });
    } else if (gateName == "IsingXY") {
      // PennyLane IsingXY has 1 parameter (phi), OpenQASM XXPlusYY needs 2
      // (theta, beta) Relationship: IsingXY(phi) = XXPlusYY(phi, pi)
      // Add pi as second parameter (since we add it during compilation)

      qcOp = qc::XXPlusYYOp::create(rewriter, op.getLoc(), inQubits[0],
                                    inQubits[1], paramsValues[0],
                                    std::numbers::pi);
    } else if (gateName == "IsingXX") {
      qcOp = CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(RXX);
    } else if (gateName == "IsingYY") {
      qcOp = CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(RYY);
    } else if (gateName == "IsingZZ") {
      qcOp = CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP(RZZ);
    } else if (gateName == "CNOT") {
      qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inQubits[0], [&]() {
        qc::XOp::create(rewriter, op.getLoc(), inQubits[1]);
      });
    } else if (gateName == "CY") {
      qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inQubits[0], [&]() {
        qc::YOp::create(rewriter, op.getLoc(), inQubits[1]);
      });
    } else if (gateName == "CZ") {
      qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inQubits[0], [&]() {
        qc::ZOp::create(rewriter, op.getLoc(), inQubits[1]);
      });
    } else if (gateName == "Toffoli") {
      qcOp = qc::CtrlOp::create(
          rewriter, op.getLoc(), inQubits.take_front(2),
          [&]() { qc::XOp::create(rewriter, op.getLoc(), inQubits[2]); });
    } else if (gateName == "CSWAP") {
      qcOp = qc::CtrlOp::create(rewriter, op.getLoc(), inQubits[0], [&]() {
        qc::SWAPOp::create(rewriter, op.getLoc(), inQubits[1], inQubits[2]);
      });
    } else {
      return op.emitError("Unsupported gate: ") << gateName;
    }

#undef CREATE_ONE_TARGET_ZERO_PARAMETER_GATE_OP
#undef CREATE_TWO_TARGET_ZERO_PARAMETER_GATE_OP
#undef CREATE_ONE_TARGET_ONE_PARAMETER_GATE_OP
#undef CREATE_TWO_TARGET_ONE_PARAMETER_GATE_OP

    llvm::SmallVector<mlir::Value> combined;
    combined.reserve(inQubits.size() + inCtrlQubits.size());
    combined.append(inQubits.begin(), inQubits.end());
    combined.append(inCtrlQubits.begin(), inCtrlQubits.end());

    mlir::ValueRange combinedRange = mlir::ValueRange(combined);
    // Replace the original with the new operation
    rewriter.replaceOp(op, combinedRange);
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
