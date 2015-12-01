//===-- CBackend.cpp - Library for converting LLVM code to C --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library converts LLVM code to C code, compilable by GCC and other C
// compilers.
//
//===----------------------------------------------------------------------===//

#include "CTargetMachine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ConstantsScanner.h"
#define MXPA_CODEGEN 1
#ifdef MXPA_CODEGEN
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/Path.h"
#endif
#include "llvm/Config/config.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/IR/CallSite.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Transforms/Scalar.h"
#include <algorithm>
#include <cstdio>
#include <set>
#include <iostream>
#include <sstream>

// Some ms header decided to define setjmp as _setjmp, undo this for this file.
#ifdef _MSC_VER
#undef setjmp
#endif
using namespace llvm;

// Switch of OpenCL C kernel compilation
static cl::opt<bool> CompileOpenCLKernel(
  "compile-opencl-kernel", cl::Hidden,
  cl::desc("Compile OpenCL C kernel"), cl::init(false));

extern "C" void LLVMInitializeCBackendTarget() {
  // Register the target.
  RegisterTargetMachine<CTargetMachine> X(TheCBackendTarget);
}

namespace {
class CBEMCAsmInfo : public MCAsmInfo {
public:
  CBEMCAsmInfo() {
    //GlobalPrefix = "";
    PrivateGlobalPrefix = "";
  }
};

/// CWriter - This class is the main chunk of code that converts an LLVM
/// module to a C translation unit.
class CWriter : public FunctionPass, public InstVisitor<CWriter> {
  formatted_raw_ostream &Out;
  IntrinsicLowering *IL;
  Mangler *Mang;
  LoopInfo *LI;
  const Module *TheModule;
  const MCAsmInfo *TAsm;
  const MCRegisterInfo *MRI;
  const MCObjectFileInfo *MOFI;
  MCContext *TCtx;
  const DataLayout *TD;

  std::map<const ConstantFP *, unsigned> FPConstantMap;
  std::set<Function *> intrinsicPrototypesAlreadyGenerated;
  std::set<const Argument *> ByValParams;
  unsigned FPCounter;
  unsigned OpaqueCounter;
  DenseMap<const Value *, unsigned> AnonValueNumbers;
  unsigned NextAnonValueNumber;

  bool OnlyNamed;
  std::vector<StructType *> StructTypes;
  DenseSet<const Value *> VisitedConstants;
  DenseSet<Type *> VisitedTypes;

  /// UnnamedStructIDs - This contains a unique ID for each struct that is
  /// either anonymous or has no name.
  DenseMap<StructType *, unsigned> UnnamedStructIDs;
  std::string bcHash;
#ifdef MXPA_CODEGEN
  ScalarEvolution *SE;
  std::map<Value *, std::set<const SCEV *> > MemoryAccesses;
  std::map<Value *, const SCEV *> MemoryElementSize;
  std::set<const SCEV *> MemoryWrites;
  AttributeSet KernelFunctionAS;
  std::set<const Function *> KernelFnCallees;
  std::set<const Function *> InlinedCallees;
  std::set<const Function *> NonInlinedCallees;
  std::vector<std::string> kernelList;
  std::vector< std::vector<unsigned int> > work_group_list;
  std::vector< std::vector<std::string> > arg_addr_space_list;
  std::vector< std::vector<std::string> > arg_access_qual_list;
  std::vector< std::vector<std::string> > arg_type_list;
  std::vector< std::vector<std::string> > arg_type_qual_list;
  std::vector< std::vector<std::string> > arg_name_list;
  std::set<const Function *> LaunchKernels;
#endif

public:
  static char ID;
  explicit CWriter(formatted_raw_ostream &o)
    : FunctionPass(ID), Out(o), IL(0), Mang(0), LI(0),
      TheModule(0), TAsm(0), MRI(0), MOFI(0), TCtx(0), TD(0),
      OpaqueCounter(0), NextAnonValueNumber(0) {
    initializeLoopInfoPass(*PassRegistry::getPassRegistry());
    FPCounter = 0;
  }

  virtual const char *getPassName() const {
    return "C backend";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfo>();
    AU.addRequired<ScalarEvolution>();
    AU.setPreservesAll();
  }

  virtual bool doInitialization(Module &M);

  bool runOnFunction(Function &F) {
#ifdef MXPA_CODEGEN
    MemoryAccesses.clear();
    MemoryElementSize.clear();
    MemoryWrites.clear();
    SE = &getAnalysis<ScalarEvolution>();
#endif
    // Do not codegen any 'available_externally' functions at all, they have
    // definitions outside the translation unit.
    if (F.hasAvailableExternallyLinkage()) {
      return false;
    }

    LI = &getAnalysis<LoopInfo>();

    // Get rid of intrinsics we can't handle.
    lowerIntrinsics(F);

    // Output all floating point constants that cannot be printed accurately.
#ifdef MXPA_CODEGEN
    Out << "\n#ifndef QUERIES\n";
#endif
    printFloatingPointConstants(F);
    printFunction(F);
#ifdef MXPA_CODEGEN
      Out << "\n#endif\n";
#endif

#ifdef MXPA_CODEGEN
    if (isKernelFunction(&F) && !CompileOpenCLKernel) {
      Out << "\n#ifdef QUERIES\n";
      printMXPABounds(F, true);
      printMXPABounds(F, false);
      printMXPAArgQueries(F);
      Out << "\n#endif\n";
    }
#endif
    return false;
  }

  virtual bool doFinalization(Module &M) {
    if (!CompileOpenCLKernel) {
      //printMXPAKernelRegister();
    }
    // Free memory...
    delete IL;
    delete TD;
    delete Mang;
    delete TCtx;
    delete TAsm;
    delete MRI;
    delete MOFI;
    FPConstantMap.clear();
    ByValParams.clear();
    intrinsicPrototypesAlreadyGenerated.clear();
    UnnamedStructIDs.clear();
    return false;
  }

  raw_ostream &printType(raw_ostream &Out, Type *Ty,
                         bool isSigned = false,
                         const std::string &VariableName = "",
                         bool IgnoreName = false,
                         const AttributeSet &PAL = AttributeSet());
  raw_ostream &printSimpleType(raw_ostream &Out, Type *Ty,
                               bool isSigned,
                               const std::string &NameSoFar = "",
                               bool isOpenCLC = false);

  void printStructReturnPointerFunctionType(raw_ostream &Out,
      const AttributeSet &PAL,
      PointerType *Ty);

  std::string getStructName(StructType *ST);

  /// writeOperandDeref - Print the result of dereferencing the specified
  /// operand with '*'.  This is equivalent to printing '*' then using
  /// writeOperand, but avoids excess syntax in some cases.
  void writeOperandDeref(Value *Operand) {
    if (isAddressExposed(Operand)) {
      // Already something with an address exposed.
      writeOperandInternal(Operand);
    } else {
      Out << "*(";
      writeOperand(Operand);
      Out << ")";
    }
  }

  bool isKernelFunction(const Function *F);
  void writeOperand(Value *Operand, bool Static = false);
  void writeInstComputationInline(Instruction &I);
  void writeOperandInternal(Value *Operand, bool Static = false);
  void writeOperandWithCast(Value *Operand, unsigned Opcode);
  void writeOperandWithCast(Value *Operand, const ICmpInst &I);
  bool writeInstructionCast(const Instruction &I);

  void writeMemoryAccess(Value *Operand, Type *OperandType,
                         bool IsVolatile, unsigned Alignment);
#ifdef MXPA_CODEGEN
  void writeBounds(const SCEV *, bool isUpperBound, Value *arg);
  void CheckBounds(const SCEV *, bool isUpperBound, char &ret);
  void writeBoundsUnknown(Value *, bool isUpperBound, Value *arg);
  void writeLoadInst(Value *Operand, Type *OperandType,
                     bool IsVolatile, unsigned Alignment, bool, Value *arg);
#endif

private :
  std::string InterpretASMConstraint(InlineAsm::ConstraintInfo &c);

  void lowerIntrinsics(Function &F);
  /// Prints the definition of the intrinsic function F. Supports the
  /// intrinsics which need to be explicitly defined in the CBackend.
  void printIntrinsicDefinition(const Function &F, raw_ostream &Out);

  void printModuleTypes();
  void printContainedStructs(Type *Ty, SmallPtrSet<Type *, 16> &);
  void printFloatingPointConstants(Function &F);
  void printFloatingPointConstantDataSequentials(
    const ConstantDataSequential *CDS);
  void printFloatingPointConstants(const Constant *C);
  void printFunctionSignature(const Function *F, bool Prototype);

  void printFunction(Function &);
#ifdef MXPA_CODEGEN
  void getKernelFnCallees(const Module &M);
  void getInlinedCallees(const Module &M);
  void getCommonToKernels(const Module &M);
  void getLaunchKernels(const Module &M);
  bool isUselessFunction(Function &F);
  bool isLaunchKernel(const Function &F);
  void printMXPAWrapper(Function &);
  void printMXPAKernelRegister();
  bool collectKernelInfo();
  void printMXPABounds(Function &F, bool Upper);
  void printMXPABndsForSlctInst(Value *V, bool Upper);
  void printMXPABndsCommCase(
    std::map<Value *, std::set<const SCEV *> >::iterator &it,
    bool Upper);
  void printMXPAArgQueries(Function &F);
  void printMXPAArgCommCase(
    std::map<Value *, std::set<const SCEV *> >::iterator &it);
  void printMXPAArgForSlctInst(Value *V);
  bool NeedHndlMXPAForSlctInst(Value *V);
  void hndlMXPABndsForSlctInst(Value *V, bool Upper);
  void hndlMXPAArgForSlctInst(Value *V);
#endif
  void printBasicBlock(BasicBlock *BB);
  void printLoop(Loop *L);

  void printCast(unsigned opcode, Type *SrcTy, Type *DstTy);
  void printConstant(Constant *CPV, bool Static);
  void printConstantWithCast(Constant *CPV, unsigned Opcode);
  bool printConstExprCast(const ConstantExpr *CE, bool Static);
  void printConstantArray(ConstantArray *CPA, bool Static);
  void printConstantVector(ConstantVector *CV, bool Static);
  void printConstantDataSequential(ConstantDataSequential *CDS, bool Static);
  void printFunctionPrivateVaraibles(Function &F);
  void printInitializer(GlobalVariable *I);

  /// isAddressExposed - Return true if the specified value's name needs to
  /// have its address taken in order to get a C value of the correct type.
  /// This happens for global variables, byval parameters, and direct allocas.
  bool isAddressExposed(const Value *V) const {
    if (const Argument *A = dyn_cast<Argument>(V)) {
      return ByValParams.count(A);
    }
    return isa<GlobalVariable>(V) || isDirectAlloca(V);
  }

  // isInlinableInst - Attempt to inline instructions into their uses to build
  // trees as much as possible.  To do this, we have to consistently decide
  // what is acceptable to inline, so that variable declarations don't get
  // printed and an extra copy of the expr is not emitted.
  //
  static bool isInlinableInst(const Instruction &I) {
    // Always inline cmp instructions, even if they are shared by multiple
    // expressions.  GCC generates horrible code if we don't.
    if (isa<CmpInst>(I)) {
      return true;
    }

    // Must be an expression, must be used exactly once.  If it is dead, we
    // emit it inline where it would go.
    if (I.getType() == Type::getVoidTy(I.getContext()) || !I.hasOneUse() ||
        isa<TerminatorInst>(I) || isa<CallInst>(I) || isa<PHINode>(I) ||
        isa<LoadInst>(I) || isa<VAArgInst>(I) || isa<InsertElementInst>(I) ||
        isa<InsertValueInst>(I))
      // Don't inline a load across a store or other bad things!
    {
      return false;
    }

    // Must not be used in inline asm, extractelement, or shufflevector.
    if (I.hasOneUse()) {
      const Instruction &User = cast<Instruction>(*I.user_back());
      if (isInlineAsm(User) || isa<ExtractElementInst>(User) ||
          isa<ShuffleVectorInst>(User)) {
        return false;
      }
    }

    // Only inline instruction it if it's use is in the same BB as the inst.
    return I.getParent() == cast<Instruction>(I.user_back())->getParent();
  }

  // isDirectAlloca - Define fixed sized allocas in the entry block as direct
  // variables which are accessed with the & operator.  This causes GCC to
  // generate significantly better code than to emit alloca calls directly.
  //
  static const AllocaInst *isDirectAlloca(const Value *V) {
    const AllocaInst *AI = dyn_cast<AllocaInst>(V);
    if (!AI) {
      return 0;
    }
    if (AI->isArrayAllocation()) {
      return 0;  // FIXME: we can also inline fixed size array allocas!
    }
    if (AI->getParent() != &AI->getParent()->getParent()->getEntryBlock()) {
      return 0;
    }
    return AI;
  }

  // isInlineAsm - Check if the instruction is a call to an inline asm chunk.
  static bool isInlineAsm(const Instruction &I) {
    if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
      return isa<InlineAsm>(CI->getCalledValue());
    }
    return false;
  }

  // Instruction visitation functions
  friend class InstVisitor<CWriter>;

  void visitReturnInst(ReturnInst &I);
  void visitBranchInst(BranchInst &I);
  void visitSwitchInst(SwitchInst &I);
  void visitIndirectBrInst(IndirectBrInst &I);
  void visitInvokeInst(InvokeInst &I) {
    llvm_unreachable("Lowerinvoke pass didn't work!");
  }
  void visitResumeInst(ResumeInst &I) {
    llvm_unreachable("DwarfEHPrepare pass didn't work!");
  }
  void visitUnreachableInst(UnreachableInst &I);

  void visitPHINode(PHINode &I);
  void visitBinaryOperator(Instruction &I);
  void visitICmpInst(ICmpInst &I);
  void visitFCmpInst(FCmpInst &I);

  void visitCastInst(CastInst &I);
  void visitSelectInst(SelectInst &I);
  void visitCallInst(CallInst &I);
  void visitInlineAsm(CallInst &I);
  bool visitBuiltinCall(CallInst &I, Intrinsic::ID ID, bool &WroteCallee);

  void visitAllocaInst(AllocaInst &I);
  void visitLoadInst(LoadInst   &I);
  void visitStoreInst(StoreInst  &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);
  void visitVAArgInst(VAArgInst &I);

  void visitInsertElementInst(InsertElementInst &I);
  void visitExtractElementInst(ExtractElementInst &I);
  void visitShuffleVectorInst(ShuffleVectorInst &SVI);

  void visitInsertValueInst(InsertValueInst &I);
  void visitExtractValueInst(ExtractValueInst &I);

  void visitInstruction(Instruction &I) {
#ifndef NDEBUG
    errs() << "C Writer does not know about " << I;
#endif
    llvm_unreachable(0);
  }

  void outputLValue(Instruction *I) {
    Out << "  " << GetValueName(I) << " = ";
  }

  bool isGotoCodeNecessary(BasicBlock *From, BasicBlock *To);
  void printPHICopiesForSuccessor(BasicBlock *CurBlock,
                                  BasicBlock *Successor, unsigned Indent);
  void printBranchToBlock(BasicBlock *CurBlock, BasicBlock *SuccBlock,
                          unsigned Indent);
  void printGEPExpression(Value *Ptr, gep_type_iterator I,
                          gep_type_iterator E, bool Static);

  std::string GetValueName(const Value *Operand);

  void run(const Module &M, bool onlyNamed);
  void incorporateType(const Module &M, Type *Ty);
  void incorporateValue(const Module &M, const Value *V);
  void incorporateMDNode(const Module &M, const MDNode *V);
};
} //End of namespace

char CWriter::ID = 0;

static std::string CBEMangle(const std::string &S) {
  std::string Result;

  for (unsigned i = 0, e = S.size(); i != e; ++i)
    if (isalnum(S[i]) || S[i] == '_') {
      Result += S[i];
    } else {
      Result += '_';
      Result += 'A' + (S[i] & 15);
      Result += 'A' + ((S[i] >> 4) & 15);
      Result += '_';
    }
  return Result;
}

std::string CWriter::getStructName(StructType *ST) {
  if (!ST->isLiteral() && !ST->getName().empty()) {
    return CBEMangle("l_" + ST->getName().str());
  }

  return "l_unnamed_" + utostr(UnnamedStructIDs[ST]);
}

/// printStructReturnPointerFunctionType - This is like printType for a struct
/// return type, except, instead of printing the type as void (*)(Struct*, ...)
/// print it as "Struct (*)(...)", for struct return functions.
void CWriter::printStructReturnPointerFunctionType(raw_ostream &Out,
    const AttributeSet &PAL,
    PointerType *TheTy) {
  FunctionType *FTy = cast<FunctionType>(TheTy->getElementType());
  std::string tstr;
  raw_string_ostream FunctionInnards(tstr);
  FunctionInnards << " (*) (";
  bool PrintedType = false;

  FunctionType::param_iterator I = FTy->param_begin(), E = FTy->param_end();
  Type *RetTy = cast<PointerType>(*I)->getElementType();
  unsigned Idx = 1;
  for (++I, ++Idx; I != E; ++I, ++Idx) {
    if (PrintedType) {
      FunctionInnards << ", ";
    }
    Type *ArgTy = *I;
    if (PAL.hasAttribute(Idx, Attribute::ByVal)) {
      assert(ArgTy->isPointerTy());
      ArgTy = cast<PointerType>(ArgTy)->getElementType();
    }
    printType(FunctionInnards, ArgTy,
              /*isSigned=*/PAL.hasAttribute(Idx, Attribute::SExt), "");
    PrintedType = true;
  }
  if (FTy->isVarArg()) {
    if (!PrintedType) {
      FunctionInnards << " int";  //dummy argument for empty vararg functs
    }
    FunctionInnards << ", ...";
  } else if (!PrintedType) {
    FunctionInnards << "void";
  }
  FunctionInnards << ')';
  printType(Out, RetTy,
            /*isSigned=*/PAL.hasAttribute(0, Attribute::SExt), FunctionInnards.str());
}

raw_ostream &
CWriter::printSimpleType(raw_ostream &Out, Type *Ty, bool isSigned,
                         const std::string &NameSoFar, bool isOpenCLC) {
  assert(((Ty->getTypeID() >= 0 && Ty->getTypeID() <= 9) || Ty->isIntegerTy() || Ty->isVectorTy()) &&
         "Invalid type for printSimpleType");
  switch (Ty->getTypeID()) {
    case Type::VoidTyID:
      return Out << "void " << NameSoFar;
    case Type::IntegerTyID: {
        unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
        if (isOpenCLC) {
          if (NumBits == 8) {
            return Out << (isSigned ? "" : "u") << "char" << NameSoFar;
          } else if (NumBits == 16) {
            return Out << (isSigned ? "" : "u") << "short" << NameSoFar;
          } else if (NumBits == 32) {
            return Out << (isSigned ? "" : "u") << "int" << NameSoFar;
          } else if (NumBits == 64) {
            return Out << (isSigned ? "" : "u") << "long" << NameSoFar;
          }
        } else {
          if (NumBits == 1) {
            return Out << "bool " << NameSoFar;
          } else if (NumBits <= 8) {
            return Out << (isSigned ? "signed" : "unsigned") << " char " << NameSoFar;
          } else if (NumBits <= 16) {
            return Out << (isSigned ? "signed" : "unsigned") << " short " << NameSoFar;
          } else if (NumBits <= 32) {
            return Out << (isSigned ? "signed" : "unsigned") << " int " << NameSoFar;
          } else if (NumBits <= 64) {
            return Out << (isSigned ? "signed" : "unsigned") << " long " << NameSoFar;
          } else {
            assert(NumBits <= 128 && "Bit widths > 128 not implemented yet");
            return Out << (isSigned ? "int4" : "uint4") << " " << NameSoFar;
          }
        }
      }
    case Type::FloatTyID:
      return Out << "float "   << NameSoFar;
    case Type::DoubleTyID:
      return Out << "double "  << NameSoFar;
      // Lacking emulation of FP80 on PPC, etc., we assume whichever of these is
      // present matches host 'long double'.
    case Type::X86_FP80TyID:
    case Type::PPC_FP128TyID:
    case Type::FP128TyID:
      return Out << "long double " << NameSoFar;

    case Type::X86_MMXTyID:
      return printSimpleType(Out, Type::getInt32Ty(Ty->getContext()), isSigned,
                             " __attribute__((vector_size(64))) " + NameSoFar);

    case Type::VectorTyID: {
        VectorType *VTy = cast<VectorType>(Ty);
        unsigned numElements = VTy->getVectorNumElements();
        std::string n = numElements > 1 ? utostr(numElements) : "";
        return printSimpleType(Out, VTy->getElementType(), isSigned,
                               " __attribute__((vector_size(" +
                               utostr(TD->getTypeAllocSize(VTy)) + " ))) " + NameSoFar);
      }

    default:
#ifndef NDEBUG
      errs() << "Unknown primitive type: " << *Ty << "\n";
#endif
      llvm_unreachable(0);
  }
}

// Pass the Type* and the variable name and this prints out the variable
// declaration.
//
raw_ostream &CWriter::printType(raw_ostream &Out, Type *Ty,
                                bool isSigned, const std::string &NameSoFar,
                                bool IgnoreName, const AttributeSet &PAL) {
  if ((Ty->getTypeID() >= 0 && Ty->getTypeID() <= 9) || Ty->isIntegerTy() || Ty->isVectorTy()) {
    printSimpleType(Out, Ty, isSigned, NameSoFar);
    return Out;
  }

  switch (Ty->getTypeID()) {
    case Type::FunctionTyID: {
        FunctionType *FTy = cast<FunctionType>(Ty);
        std::string tstr;
        raw_string_ostream FunctionInnards(tstr);
        FunctionInnards << " (" << NameSoFar << ") (";
        unsigned Idx = 1;
        for (FunctionType::param_iterator I = FTy->param_begin(),
             E = FTy->param_end(); I != E; ++I) {
          Type *ArgTy = *I;
          if (PAL.hasAttribute(Idx, Attribute::ByVal)) {
            assert(ArgTy->isPointerTy());
            ArgTy = cast<PointerType>(ArgTy)->getElementType();
          }
          if (I != FTy->param_begin()) {
            FunctionInnards << ", ";
          }
          printType(FunctionInnards, ArgTy,
                    /*isSigned=*/PAL.hasAttribute(Idx, Attribute::SExt), "");
          ++Idx;
        }
        if (FTy->isVarArg()) {
          if (!FTy->getNumParams()) {
            FunctionInnards << " int";  //dummy argument for empty vaarg functs
          }
          FunctionInnards << ", ...";
        } else if (!FTy->getNumParams()) {
          FunctionInnards << "void";
        }
        FunctionInnards << ')';
        printType(Out, FTy->getReturnType(),
                  /*isSigned=*/PAL.hasAttribute(0, Attribute::SExt), FunctionInnards.str());
        return Out;
      }
    case Type::StructTyID: {
        StructType *STy = cast<StructType>(Ty);

        // Check to see if the type is named.
        if (!IgnoreName) {
          return Out << getStructName(STy) << ' ' << NameSoFar;
        }
        Out << "struct\n";
        Out << NameSoFar + " {\n";
        unsigned Idx = 0;
        for (StructType::element_iterator I = STy->element_begin(),
             E = STy->element_end(); I != E; ++I, ++Idx) {
          Out << "  ";
#if 0
          printType(Out, *I, false, "field" + utostr(Idx++));
#endif
          bool Field_IN = (STy->getNumElements() == 1) ? IgnoreName : false;
          if ((*I)->isStructTy() && Field_IN) {
            printType(Out, *I, false, "",
                      Field_IN);
            Out << "field" + utostr(Idx);
          } else {
            printType(Out, *I, false, "field" + utostr(Idx), Field_IN);
          }
          Out << ";\n";
        }
        Out << '}';
        if (STy->isPacked()) {
          Out << " __attribute__ ((packed))";
        }
        return Out;
      }

    case Type::PointerTyID: {
        PointerType *PTy = cast<PointerType>(Ty);
        std::string ptrName = "*" + NameSoFar;

        if (PTy->getElementType()->isArrayTy() ||
            PTy->getElementType()->isVectorTy()) {
          ptrName = "(" + ptrName + ")";
        }

        uint space = PTy->getAddressSpace();
        // Print the key word of global
        if (space == 1) {
          Out << " __global ";
        } else if (space == 3) {
          Out << " __local  ";
        }

        if (!PAL.isEmpty())
          // Must be a function ptr cast!
        {
          return printType(Out, PTy->getElementType(), false, ptrName, true, PAL);
        }
        return printType(Out, PTy->getElementType(), false, ptrName);
      }

    case Type::ArrayTyID: {
        ArrayType *ATy = cast<ArrayType>(Ty);

        // Check to see if the type is named.
        if (!IgnoreName) {
          std::vector<Type *> structMembers;
          structMembers.push_back(ATy);
          return Out << getStructName(StructType::get(
                                        TheModule->getContext(), structMembers)) << ' ' << NameSoFar;
        }

        unsigned NumElements = ATy->getNumElements();
        if (NumElements == 0) {
          NumElements = 1;
        }
        // Arrays are wrapped in structs to allow them to have normal
        // value semantics (avoiding the array "decay").
        Out << " struct { ";
        printType(Out, ATy->getElementType(), false,
                  "array[" + utostr(NumElements) + "]");
        return Out << "; }" << NameSoFar << " ";
      }

    default:
      llvm_unreachable("Unhandled case in getTypeProps!");
  }
}

void CWriter::printConstantArray(ConstantArray *CPA, bool Static) {
  Out << "{ ";
  printConstant(cast<Constant>(CPA->getOperand(0)), Static);
  for (unsigned i = 1, e = CPA->getNumOperands(); i != e; ++i) {
    Out << ", ";
    printConstant(cast<Constant>(CPA->getOperand(i)), Static);
  }
  Out << " }";
}

void CWriter::printConstantVector(ConstantVector *CP, bool Static) {
  Out << "{ ";
  printConstant(cast<Constant>(CP->getOperand(0)), Static);
  for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
    Out << ", ";
    printConstant(cast<Constant>(CP->getOperand(i)), Static);
  }
  Out << " }";
}

void CWriter::printConstantDataSequential(ConstantDataSequential *CDS,
    bool Static) {
  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  //
  if (CDS->isCString()) {
    Out << '\"';
    // Keep track of whether the last number was a hexadecimal escape.
    bool LastWasHex = false;

    StringRef Bytes = CDS->getAsCString();

    // Do not include the last character, which we know is null
    for (unsigned i = 0, e = Bytes.size(); i != e; ++i) {
      unsigned char C = Bytes[i];

      // Print it out literally if it is a printable character.  The only thing
      // to be careful about is when the last letter output was a hex escape
      // code, in which case we have to be careful not to print out hex digits
      // explicitly (the C compiler thinks it is a continuation of the previous
      // character, sheesh...)
      //
      if (isprint(C) && (!LastWasHex || !isxdigit(C))) {
        LastWasHex = false;
        if (C == '"' || C == '\\') {
          Out << "\\" << (char)C;
        } else {
          Out << (char)C;
        }
      } else {
        LastWasHex = false;
        switch (C) {
          case '\n':
            Out << "\\n";
            break;
          case '\t':
            Out << "\\t";
            break;
          case '\r':
            Out << "\\r";
            break;
          case '\v':
            Out << "\\v";
            break;
          case '\a':
            Out << "\\a";
            break;
          case '\"':
            Out << "\\\"";
            break;
          case '\'':
            Out << "\\\'";
            break;
          default:
            Out << "\\x";
            Out << (char)((C / 16  < 10) ? (C / 16 + '0') : (C / 16 - 10 + 'A'));
            Out << (char)(((C & 15) < 10) ? ((C & 15) + '0') : ((C & 15) - 10 + 'A'));
            LastWasHex = true;
            break;
        }
      }
    }
    Out << '\"';
  } else {
    Out << "{ ";
    printConstant(CDS->getElementAsConstant(0), Static);
    for (unsigned i = 1, e = CDS->getNumElements(); i != e; ++i) {
      Out << ", ";
      printConstant(CDS->getElementAsConstant(i), Static);
    }
    Out << " }";
  }
}

static inline std::string ftostr(const APFloat &V) {
  std::string Buf;
  if (&V.getSemantics() == &APFloat::IEEEdouble) {
    raw_string_ostream(Buf) << V.convertToDouble();
    return Buf;
  } else if (&V.getSemantics() == &APFloat::IEEEsingle) {
    raw_string_ostream(Buf) << (double)V.convertToFloat();
    return Buf;
  }
  return "<unknown format in ftostr>"; // error
}


// isFPCSafeToPrint - Returns true if we may assume that CFP may be written out
// textually as a double (rather than as a reference to a stack-allocated
// variable). We decide this by converting CFP to a string and back into a
// double, and then checking whether the conversion results in a bit-equal
// double to the original value of CFP. This depends on us and the target C
// compiler agreeing on the conversion process (which is pretty likely since we
// only deal in IEEE FP).
//
static bool isFPCSafeToPrint(const ConstantFP *CFP) {
  bool ignored;
  // Do long doubles in hex for now.
  if (CFP->getType() != Type::getFloatTy(CFP->getContext()) &&
      CFP->getType() != Type::getDoubleTy(CFP->getContext())) {
    return false;
  }
  APFloat APF = APFloat(CFP->getValueAPF());  // copy
  if (CFP->getType() == Type::getFloatTy(CFP->getContext())) {
    APF.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &ignored);
  }
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
  char Buffer[100];
  sprintf(Buffer, "%a", APF.convertToDouble());
  if (!strncmp(Buffer, "0x", 2) ||
      !strncmp(Buffer, "-0x", 3) ||
      !strncmp(Buffer, "+0x", 3)) {
    return APF.bitwiseIsEqual(APFloat(atof(Buffer)));
  }
  return false;
#else
  std::string StrVal = ftostr(APF);

  while (StrVal[0] == ' ') {
    StrVal.erase(StrVal.begin());
  }

  // Check to make sure that the stringized number is not some string like "Inf"
  // or NaN.  Check that the string matches the "[-+]?[0-9]" regex.
  if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
      ((StrVal[0] == '-' || StrVal[0] == '+') &&
       (StrVal[1] >= '0' && StrVal[1] <= '9')))
    // Reparse stringized version!
  {
    return APF.bitwiseIsEqual(APFloat(atof(StrVal.c_str())));
  }
  return false;
#endif
}

/// Print out the casting for a cast operation. This does the double casting
/// necessary for conversion to the destination type, if necessary.
/// @brief Print a cast
void CWriter::printCast(unsigned opc, Type *SrcTy, Type *DstTy) {
  // Print the destination type cast
  bool is128Bit = false;
  if (DstTy->getTypeID() == Type::IntegerTyID) {
    unsigned NumBits = cast<IntegerType>(DstTy)->getBitWidth();
    if (NumBits == 128) {
      is128Bit = true;
    }
  }
  switch (opc) {
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::IntToPtr:
    case Instruction::Trunc:
      if (SrcTy->getTypeID() == Type::IntegerTyID) {
        unsigned n = cast<IntegerType>(SrcTy)->getBitWidth();
        if (n == 128) {
          Out << '(';
          printSimpleType(Out, DstTy, false);
          Out << "*)";
          break;
        }
      }
    case Instruction::BitCast:
      if (is128Bit) {
        Out << '(';
        printSimpleType(Out, DstTy, false);
        Out << "*)";
        break;
      }
    case Instruction::FPExt:
    case Instruction::FPTrunc: // For these the DstTy sign doesn't matter
    case Instruction::AddrSpaceCast:
      Out << '(';
      printType(Out, DstTy);
      Out << ')';
      break;
    case Instruction::ZExt:
    case Instruction::PtrToInt:
    case Instruction::FPToUI: // For these, make sure we get an unsigned dest
      Out << '(';
      printSimpleType(Out, DstTy, false);
      Out << ')';
      break;
    case Instruction::SExt:
    case Instruction::FPToSI: // For these, make sure we get a signed dest
      Out << '(';
      printSimpleType(Out, DstTy, true);
      Out << ')';
      break;
    default:
      llvm_unreachable("Invalid cast opcode");
  }

  // Print the source type cast
  switch (opc) {
    case Instruction::UIToFP:
    case Instruction::ZExt:
      Out << '(';
      printSimpleType(Out, SrcTy, false);
      Out << ')';
      break;
    case Instruction::SIToFP:
    case Instruction::SExt:
      Out << '(';
      printSimpleType(Out, SrcTy, true);
      Out << ')';
      break;
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
      // Avoid "cast to pointer from integer of different size" warnings
      Out << "(unsigned long)";
      break;
    case Instruction::Trunc:
    case Instruction::BitCast:
      if (is128Bit) {
        Out << '&';
        break;
      }
    case Instruction::FPExt:
    case Instruction::FPTrunc:
    case Instruction::FPToSI:
    case Instruction::FPToUI:
    case Instruction::AddrSpaceCast:
      break; // These don't need a source cast.
    default:
      llvm_unreachable("Invalid cast opcode");
  }
}

// printConstant - The LLVM Constant to C Constant converter.
void CWriter::printConstant(Constant *CPV, bool Static) {
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    switch (CE->getOpcode()) {
      case Instruction::Trunc:
      case Instruction::ZExt:
      case Instruction::SExt:
      case Instruction::FPTrunc:
      case Instruction::FPExt:
      case Instruction::UIToFP:
      case Instruction::SIToFP:
      case Instruction::FPToUI:
      case Instruction::FPToSI:
      case Instruction::PtrToInt:
      case Instruction::IntToPtr:
      case Instruction::BitCast:
        Out << "(";
        printCast(CE->getOpcode(), CE->getOperand(0)->getType(), CE->getType());
        if (CE->getOpcode() == Instruction::SExt &&
            CE->getOperand(0)->getType() == Type::getInt1Ty(CPV->getContext())) {
          // Make sure we really sext from bool here by subtracting from 0
          Out << "0-";
        }
        printConstant(CE->getOperand(0), Static);
        if (CE->getType() == Type::getInt1Ty(CPV->getContext()) &&
            (CE->getOpcode() == Instruction::Trunc ||
             CE->getOpcode() == Instruction::FPToUI ||
             CE->getOpcode() == Instruction::FPToSI ||
             CE->getOpcode() == Instruction::PtrToInt)) {
          // Make sure we really truncate to bool here by anding with 1
          Out << "&1u";
        }
        Out << ')';
        return;

      case Instruction::GetElementPtr:
        Out << "(";
        printGEPExpression(CE->getOperand(0), gep_type_begin(CPV),
                           gep_type_end(CPV), Static);
        Out << ")";
        return;
      case Instruction::Select:
        Out << '(';
        printConstant(CE->getOperand(0), Static);
        Out << '?';
        printConstant(CE->getOperand(1), Static);
        Out << ':';
        printConstant(CE->getOperand(2), Static);
        Out << ')';
        return;
      case Instruction::Add:
      case Instruction::FAdd:
      case Instruction::Sub:
      case Instruction::FSub:
      case Instruction::Mul:
      case Instruction::FMul:
      case Instruction::SDiv:
      case Instruction::UDiv:
      case Instruction::FDiv:
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::FRem:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
      case Instruction::ICmp:
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr: {
          Out << '(';
          bool NeedsClosingParens = printConstExprCast(CE, Static);
          printConstantWithCast(CE->getOperand(0), CE->getOpcode());
          switch (CE->getOpcode()) {
            case Instruction::Add:
            case Instruction::FAdd:
              Out << " + ";
              break;
            case Instruction::Sub:
            case Instruction::FSub:
              Out << " - ";
              break;
            case Instruction::Mul:
            case Instruction::FMul:
              Out << " * ";
              break;
            case Instruction::URem:
            case Instruction::SRem:
            case Instruction::FRem:
              Out << " % ";
              break;
            case Instruction::UDiv:
            case Instruction::SDiv:
            case Instruction::FDiv:
              Out << " / ";
              break;
            case Instruction::And:
              Out << " & ";
              break;
            case Instruction::Or:
              Out << " | ";
              break;
            case Instruction::Xor:
              Out << " ^ ";
              break;
            case Instruction::Shl:
              Out << " << ";
              break;
            case Instruction::LShr:
            case Instruction::AShr:
              Out << " >> ";
              break;
            case Instruction::ICmp:
              switch (CE->getPredicate()) {
                case ICmpInst::ICMP_EQ:
                  Out << " == ";
                  break;
                case ICmpInst::ICMP_NE:
                  Out << " != ";
                  break;
                case ICmpInst::ICMP_SLT:
                case ICmpInst::ICMP_ULT:
                  Out << " < ";
                  break;
                case ICmpInst::ICMP_SLE:
                case ICmpInst::ICMP_ULE:
                  Out << " <= ";
                  break;
                case ICmpInst::ICMP_SGT:
                case ICmpInst::ICMP_UGT:
                  Out << " > ";
                  break;
                case ICmpInst::ICMP_SGE:
                case ICmpInst::ICMP_UGE:
                  Out << " >= ";
                  break;
                default:
                  llvm_unreachable("Illegal ICmp predicate");
              }
              break;
            default:
              llvm_unreachable("Illegal opcode here!");
          }
          printConstantWithCast(CE->getOperand(1), CE->getOpcode());
          if (NeedsClosingParens) {
            Out << "))";
          }
          Out << ')';
          return;
        }
      case Instruction::FCmp: {
          Out << '(';
          bool NeedsClosingParens = printConstExprCast(CE, Static);
          if (CE->getPredicate() == FCmpInst::FCMP_FALSE) {
            Out << "0";
          } else if (CE->getPredicate() == FCmpInst::FCMP_TRUE) {
            Out << "1";
          } else {
            const char *op = 0;
            switch (CE->getPredicate()) {
              default:
                llvm_unreachable("Illegal FCmp predicate");
              case FCmpInst::FCMP_ORD:
                op = "ord";
                break;
              case FCmpInst::FCMP_UNO:
                op = "uno";
                break;
              case FCmpInst::FCMP_UEQ:
                op = "ueq";
                break;
              case FCmpInst::FCMP_UNE:
                op = "une";
                break;
              case FCmpInst::FCMP_ULT:
                op = "ult";
                break;
              case FCmpInst::FCMP_ULE:
                op = "ule";
                break;
              case FCmpInst::FCMP_UGT:
                op = "ugt";
                break;
              case FCmpInst::FCMP_UGE:
                op = "uge";
                break;
              case FCmpInst::FCMP_OEQ:
                op = "oeq";
                break;
              case FCmpInst::FCMP_ONE:
                op = "one";
                break;
              case FCmpInst::FCMP_OLT:
                op = "olt";
                break;
              case FCmpInst::FCMP_OLE:
                op = "ole";
                break;
              case FCmpInst::FCMP_OGT:
                op = "ogt";
                break;
              case FCmpInst::FCMP_OGE:
                op = "oge";
                break;
            }
            Out << "llvm_fcmp_" << op << "(";
            printConstantWithCast(CE->getOperand(0), CE->getOpcode());
            Out << ", ";
            printConstantWithCast(CE->getOperand(1), CE->getOpcode());
            Out << ")";
          }
          if (NeedsClosingParens) {
            Out << "))";
          }
          Out << ')';
          return;
        }
      default:
#ifndef NDEBUG
        errs() << "CWriter Error: Unhandled constant expression: "
               << *CE << "\n";
#endif
        llvm_unreachable(0);
    }
  } else if (isa<UndefValue>(CPV) && CPV->getType()->isSingleValueType()) {
    Out << "((";
    printType(Out, CPV->getType()); // sign doesn't matter
    Out << ")/*UNDEF*/";
    if (!CPV->getType()->isVectorTy()) {
      Out << "0)";
    } else {
      Out << "{})";
    }
    return;
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(CPV)) {
    Type *Ty = CI->getType();
    if (Ty == Type::getInt1Ty(CPV->getContext())) {
      Out << (CI->getZExtValue() ? '1' : '0');
    } else if (Ty == Type::getInt32Ty(CPV->getContext())) {
      Out << CI->getZExtValue() << 'u';
    } else if (Ty->getPrimitiveSizeInBits() > 32) {
      Out << CI->getZExtValue() << "ul";
    } else {
      Out << "((";
      printSimpleType(Out, Ty, false) << ')';
      if (CI->isMinValue(true)) {
        Out << CI->getZExtValue() << 'u';
      } else {
        Out << CI->getSExtValue();
      }
      Out << ')';
    }
    return;
  }

  switch (CPV->getType()->getTypeID()) {
    case Type::FloatTyID:
    case Type::DoubleTyID:
    case Type::X86_FP80TyID:
    case Type::PPC_FP128TyID:
    case Type::FP128TyID: {
        ConstantFP *FPC = cast<ConstantFP>(CPV);
        std::map<const ConstantFP *, unsigned>::iterator I = FPConstantMap.find(FPC);
        if (I != FPConstantMap.end()) {
          // Because of FP precision problems we must load from a stack allocated
          // value that holds the value in hex.
          Out << "\n#ifdef __OPENCL_VERSION__\n";
          Out << "(*(" << (FPC->getType() == Type::getFloatTy(CPV->getContext()) ?
                           "__constant float" :
                           "__constant double")
              << "*)&FPConstant" << I->second << ')';
          Out << "\n#else\n";
          Out << "(*(" << (FPC->getType() == Type::getFloatTy(CPV->getContext()) ?
                           "float" :
                           FPC->getType() == Type::getDoubleTy(CPV->getContext()) ?
                           "double" :
                           "long double")
              << "*)&FPConstant" << I->second << ')';
          Out << "\n#endif\n";
        } else {
          double V;
          if (FPC->getType() == Type::getFloatTy(CPV->getContext())) {
            V = FPC->getValueAPF().convertToFloat();
          } else if (FPC->getType() == Type::getDoubleTy(CPV->getContext())) {
            V = FPC->getValueAPF().convertToDouble();
          } else {
            // Long double.  Convert the number to double, discarding precision.
            // This is not awesome, but it at least makes the CBE output somewhat
            // useful.
            APFloat Tmp = FPC->getValueAPF();
            bool LosesInfo;
            Tmp.convert(APFloat::IEEEdouble, APFloat::rmTowardZero, &LosesInfo);
            V = Tmp.convertToDouble();
          }

          if (IsNAN(V)) {
            // The value is NaN

            // FIXME the actual NaN bits should be emitted.
            // The prefix for a quiet NaN is 0x7FF8. For a signalling NaN,
            // it's 0x7ff4.
            const unsigned long QuietNaN = 0x7ff8UL;
            //const unsigned long SignalNaN = 0x7ff4UL;

            // We need to grab the first part of the FP #
            char Buffer[100];

            uint64_t ll = DoubleToBits(V);
            sprintf(Buffer, "0x%llx", static_cast<long long>(ll));

            std::string Num(&Buffer[0], &Buffer[6]);
            unsigned long Val = strtoul(Num.c_str(), 0, 16);

            if (FPC->getType() == Type::getFloatTy(FPC->getContext()))
              Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "F(\""
                  << Buffer << "\") /*nan*/ ";
            else
              Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "(\""
                  << Buffer << "\") /*nan*/ ";
          } else if (IsInf(V)) {
            // The value is Inf
            if (V < 0) {
              Out << '-';
            }
            Out << "LLVM_INF" <<
                (FPC->getType() == Type::getFloatTy(FPC->getContext()) ? "F" : "")
                << " /*inf*/ ";
          } else {
            std::string Num;
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
            // Print out the constant as a floating point number.
            char Buffer[100];
            sprintf(Buffer, "%a", V);
            Num = Buffer;
#else
            Num = ftostr(FPC->getValueAPF());
#endif
            Out << Num;
          }
        }
        break;
      }

    case Type::ArrayTyID:
      // Use C99 compound expression literal initializer syntax.
      if (!Static) {
        Out << "(";
        printType(Out, CPV->getType());
        Out << ")";
      }
      Out << " { { "; // Arrays are wrapped in struct types.
      if (ConstantArray *CA = dyn_cast<ConstantArray>(CPV)) {
        printConstantArray(CA, Static);
      } else if (ConstantDataSequential *CDS =
                   dyn_cast<ConstantDataSequential>(CPV)) {
        printConstantDataSequential(CDS, Static);
      } else {
        assert(isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV));
        ArrayType *AT = cast<ArrayType>(CPV->getType());
        Out << '{';
        if (AT->getNumElements()) {
          Out << ' ';
          Constant *CZ = Constant::getNullValue(AT->getElementType());
          printConstant(CZ, Static);
          for (unsigned i = 1, e = AT->getNumElements(); i != e; ++i) {
            Out << ", ";
            printConstant(CZ, Static);
          }
        }
        Out << " }";
      }
      Out << " } }"; // Arrays are wrapped in struct types.
      break;

    case Type::VectorTyID:
      // Use C99 compound expression literal initializer syntax.
      if (!Static) {
        Out << "(";
        printType(Out, CPV->getType());
        Out << ")";
      }
      if (ConstantVector *CV = dyn_cast<ConstantVector>(CPV)) {
        printConstantVector(CV, Static);
      } else if (ConstantDataSequential *CDS =
                   dyn_cast<ConstantDataSequential>(CPV)) {
        printConstantDataSequential(CDS, Static);
      } else {
        assert(isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV));
        VectorType *VT = cast<VectorType>(CPV->getType());
        Out << "{ ";
        Constant *CZ = Constant::getNullValue(VT->getElementType());
        printConstant(CZ, Static);
        for (unsigned i = 1, e = VT->getNumElements(); i != e; ++i) {
          Out << ", ";
          printConstant(CZ, Static);
        }
        Out << " }";
      }
      break;

    case Type::StructTyID:
      // Use C99 compound expression literal initializer syntax.
      if (!Static) {
        Out << "(";
        printType(Out, CPV->getType());
        Out << ")";
      }
      if (isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV)) {
        StructType *ST = cast<StructType>(CPV->getType());
        Out << '{';
        if (ST->getNumElements()) {
          Out << ' ';
          printConstant(Constant::getNullValue(ST->getElementType(0)), Static);
          for (unsigned i = 1, e = ST->getNumElements(); i != e; ++i) {
            Out << ", ";
            printConstant(Constant::getNullValue(ST->getElementType(i)), Static);
          }
        }
        Out << " }";
      } else {
        Out << '{';
        if (CPV->getNumOperands()) {
          Out << ' ';
          printConstant(cast<Constant>(CPV->getOperand(0)), Static);
          for (unsigned i = 1, e = CPV->getNumOperands(); i != e; ++i) {
            Out << ", ";
            printConstant(cast<Constant>(CPV->getOperand(i)), Static);
          }
        }
        Out << " }";
      }
      break;

    case Type::PointerTyID:
      if (isa<ConstantPointerNull>(CPV)) {
        Out << "((";
        printType(Out, CPV->getType()); // sign doesn't matter
        Out << ")/*NULL*/0)";
        break;
      } else if (GlobalValue *GV = dyn_cast<GlobalValue>(CPV)) {
        writeOperand(GV, Static);
        break;
      }
      // FALL THROUGH
    default:
#ifndef NDEBUG
      errs() << "Unknown constant type: " << *CPV << "\n";
#endif
      llvm_unreachable(0);
  }
}

// Some constant expressions need to be casted back to the original types
// because their operands were casted to the expected type. This function takes
// care of detecting that case and printing the cast for the ConstantExpr.
bool CWriter::printConstExprCast(const ConstantExpr *CE, bool Static) {
  bool NeedsExplicitCast = false;
  Type *Ty = CE->getOperand(0)->getType();
  bool TypeIsSigned = false;
  switch (CE->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
      // We need to cast integer arithmetic so that it is always performed
      // as unsigned, to avoid undefined behavior on overflow.
    case Instruction::LShr:
    case Instruction::URem:
    case Instruction::UDiv:
      NeedsExplicitCast = true;
      break;
    case Instruction::AShr:
    case Instruction::SRem:
    case Instruction::SDiv:
      NeedsExplicitCast = true;
      TypeIsSigned = true;
      break;
    case Instruction::SExt:
      Ty = CE->getType();
      NeedsExplicitCast = true;
      TypeIsSigned = true;
      break;
    case Instruction::ZExt:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
      Ty = CE->getType();
      NeedsExplicitCast = true;
      break;
    default:
      break;
  }
  if (NeedsExplicitCast) {
    Out << "((";
    if (Ty->isIntegerTy() && Ty != Type::getInt1Ty(Ty->getContext())) {
      printSimpleType(Out, Ty, TypeIsSigned);
    } else {
      printType(Out, Ty);  // not integer, sign doesn't matter
    }
    Out << ")(";
  }
  return NeedsExplicitCast;
}

//  Print a constant assuming that it is the operand for a given Opcode. The
//  opcodes that care about sign need to cast their operands to the expected
//  type before the operation proceeds. This function does the casting.
void CWriter::printConstantWithCast(Constant *CPV, unsigned Opcode) {

  // Extract the operand's type, we'll need it.
  Type *OpTy = CPV->getType();

  // Indicate whether to do the cast or not.
  bool shouldCast = false;
  bool typeIsSigned = false;

  // Based on the Opcode for which this Constant is being written, determine
  // the new type to which the operand should be casted by setting the value
  // of OpTy. If we change OpTy, also set shouldCast to true so it gets
  // casted below.
  switch (Opcode) {
    default:
      // for most instructions, it doesn't matter
      break;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
      // We need to cast integer arithmetic so that it is always performed
      // as unsigned, to avoid undefined behavior on overflow.
    case Instruction::LShr:
    case Instruction::UDiv:
    case Instruction::URem:
      shouldCast = true;
      break;
    case Instruction::AShr:
    case Instruction::SDiv:
    case Instruction::SRem:
      shouldCast = true;
      typeIsSigned = true;
      break;
  }

  // Write out the casted constant if we should, otherwise just write the
  // operand.
  if (shouldCast) {
    Out << "((";
    printSimpleType(Out, OpTy, typeIsSigned);
    Out << ")";
    printConstant(CPV, false);
    Out << ")";
  } else {
    printConstant(CPV, false);
  }
}

std::string CWriter::GetValueName(const Value *Operand) {

#if 0
  // To Ensure
  // Resolve potential alias.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(Operand)) {
    if (const Value *V = GA->resolveAliasedGlobal(false)) {
      Operand = V;
    }
  }
#endif

  // Mangle globals with the standard mangler interface for LLC compatibility.
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(Operand)) {
    SmallString<128> Str;
    Mang->getNameWithPrefix(Str, GV, false);
    return CBEMangle(Str.str().str());
  }

  std::string Name = Operand->getName();

  if (Name.empty()) { // Assign unique names to local temporaries.
    unsigned &No = AnonValueNumbers[Operand];
    if (No == 0) {
      No = ++NextAnonValueNumber;
    }
    Name = "tmp__" + utostr(No);
  }

  std::string VarName;
  VarName.reserve(Name.capacity());

  for (std::string::iterator I = Name.begin(), E = Name.end();
       I != E; ++I) {
    char ch = *I;

    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_')) {
      char buffer[5];
      sprintf(buffer, "_%x_", ch);
      VarName += buffer;
    } else {
      VarName += ch;
    }
  }

  return "llvm_cbe_" + VarName;
}

/// writeInstComputationInline - Emit the computation for the specified
/// instruction inline, with no destination provided.
void CWriter::writeInstComputationInline(Instruction &I) {
  // We can't currently support integer types other than 1, 8, 16, 32, 64.
  // Validate this.
  Type *Ty = I.getType();
  bool isBitCast128Ty = false;
  if (isa<BitCastInst>(&I) && Ty->isIntegerTy()) {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits == 128) {
      isBitCast128Ty = true;
    }
  }
  if (!isBitCast128Ty) {
    // FIXME: Remove 24u, 48u, 96u if unnecessarily
    if (Ty->isIntegerTy() && Ty != Type::getInt1Ty(I.getContext()) &&
        Ty != Type::getInt8Ty(I.getContext()) &&
        Ty != Type::getInt16Ty(I.getContext()) &&
        Ty != Type::getInt32Ty(I.getContext()) &&
        Ty != Type::getInt64Ty(I.getContext()) /*&&
        Ty != Type::getIntNTy(I.getContext(), 24u) &&
        Ty != Type::getIntNTy(I.getContext(), 48u) &&
        Ty != Type::getIntNTy(I.getContext(), 96u)*/) {
      report_fatal_error("The C backend does not currently support integer "
                         "types of widths other than 1, 8, 16, 32, 64.\n"
                         "This is being tracked as PR 4158.");
    }
  }

  // If this is a non-trivial bool computation, make sure to truncate down to
  // a 1 bit value.  This is important because we want "add i1 x, y" to return
  // "0" when x and y are true, not "2" for example.
  bool NeedBoolTrunc = false;
  if (I.getType() == Type::getInt1Ty(I.getContext()) &&
      !isa<ICmpInst>(I) && !isa<FCmpInst>(I)) {
    NeedBoolTrunc = true;
  }

  if (NeedBoolTrunc) {
    Out << "((";
  }

  visit(I);

  if (NeedBoolTrunc) {
    Out << ")&1)";
  }
}


void CWriter::writeOperandInternal(Value *Operand, bool Static) {
  if (Instruction *I = dyn_cast<Instruction>(Operand))
    // Should we inline this instruction to build a tree?
    if (isInlinableInst(*I) && !isDirectAlloca(I)) {
      Out << '(';
      writeInstComputationInline(*I);
      Out << ')';
      return;
    }

  Constant *CPV = dyn_cast<Constant>(Operand);

  if (CPV && !isa<GlobalValue>(CPV)) {
    printConstant(CPV, Static);
  } else {
    Out << GetValueName(Operand);
  }
}

void CWriter::writeOperand(Value *Operand, bool Static) {
  bool isAddressImplicit = isAddressExposed(Operand);
  if (isAddressImplicit) {
    Out << "(&";  // Global variables are referenced as their addresses by llvm
  }

  writeOperandInternal(Operand, Static);

  if (isAddressImplicit) {
    Out << ')';
  }
}

// Some instructions need to have their result value casted back to the
// original types because their operands were casted to the expected type.
// This function takes care of detecting that case and printing the cast
// for the Instruction.
bool CWriter::writeInstructionCast(const Instruction &I) {
  Type *Ty = I.getOperand(0)->getType();
  switch (I.getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
      // We need to cast integer arithmetic so that it is always performed
      // as unsigned, to avoid undefined behavior on overflow.
    case Instruction::LShr:
    case Instruction::URem:
    case Instruction::UDiv:
      Out << "((";
      printSimpleType(Out, Ty, false);
      Out << ")(";
      return true;
    case Instruction::AShr:
    case Instruction::SRem:
    case Instruction::SDiv:
      Out << "((";
      printSimpleType(Out, Ty, true);
      Out << ")(";
      return true;
    default:
      break;
  }
  return false;
}

// Write the operand with a cast to another type based on the Opcode being used.
// This will be used in cases where an instruction has specific type
// requirements (usually signedness) for its operands.
void CWriter::writeOperandWithCast(Value *Operand, unsigned Opcode) {

  // Extract the operand's type, we'll need it.
  Type *OpTy = Operand->getType();

  // Indicate whether to do the cast or not.
  bool shouldCast = false;

  // Indicate whether the cast should be to a signed type or not.
  bool castIsSigned = false;

  // Based on the Opcode for which this Operand is being written, determine
  // the new type to which the operand should be casted by setting the value
  // of OpTy. If we change OpTy, also set shouldCast to true.
  switch (Opcode) {
    default:
      // for most instructions, it doesn't matter
      break;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
      // We need to cast integer arithmetic so that it is always performed
      // as unsigned, to avoid undefined behavior on overflow.
    case Instruction::LShr:
    case Instruction::UDiv:
    case Instruction::URem: // Cast to unsigned first
      shouldCast = true;
      castIsSigned = false;
      break;
    case Instruction::GetElementPtr:
    case Instruction::AShr:
    case Instruction::SDiv:
    case Instruction::SRem: // Cast to signed first
      shouldCast = true;
      castIsSigned = true;
      break;
  }

  // Write out the casted operand if we should, otherwise just write the
  // operand.
  if (shouldCast) {
    Out << "((";
    printSimpleType(Out, OpTy, castIsSigned);
    Out << ")";
    writeOperand(Operand);
    Out << ")";
  } else {
    writeOperand(Operand);
  }
}

// Write the operand with a cast to another type based on the icmp predicate
// being used.
void CWriter::writeOperandWithCast(Value *Operand, const ICmpInst &Cmp) {
  // This has to do a cast to ensure the operand has the right signedness.
  // Also, if the operand is a pointer, we make sure to cast to an integer when
  // doing the comparison both for signedness and so that the C compiler doesn't
  // optimize things like "p < NULL" to false (p may contain an integer value
  // f.e.).
  bool shouldCast = Cmp.isRelational();

  // Write out the casted operand if we should, otherwise just write the
  // operand.
  if (!shouldCast) {
    writeOperand(Operand);
    return;
  }

  // Should this be a signed comparison?  If so, convert to signed.
  bool castIsSigned = Cmp.isSigned();

  // If the operand was a pointer, convert to a large integer type.
  Type *OpTy = Operand->getType();
  if (OpTy->isPointerTy()) {
    OpTy = TD->getIntPtrType(Operand->getContext());
  }

  Out << "((";
  printSimpleType(Out, OpTy, castIsSigned);
  Out << ")";
  writeOperand(Operand);
  Out << ")";
}

// generateCompilerSpecificCode - This is where we add conditional compilation
// directives to cater to specific compilers as need be.
//
static void generateCompilerSpecificCode(formatted_raw_ostream &Out,
    const DataLayout *TD) {
  // Alloca is hard to get, and we don't want to include stdlib.h here.
#if 0
  Out << "/* get a declaration for alloca */\n"
      << "#if defined(__CYGWIN__) || defined(__MINGW32__)\n"
      << "#define  alloca(x) __builtin_alloca((x))\n"
      << "#define _alloca(x) __builtin_alloca((x))\n"
      << "#elif defined(__APPLE__)\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#define longjmp _longjmp\n"
      << "#define setjmp _setjmp\n"
      << "#elif defined(__sun__)\n"
      << "#if defined(__sparcv9)\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#else\n"
      << "extern void *__builtin_alloca(unsigned int);\n"
      << "#endif\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__) || defined(__arm__)\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#elif defined(_MSC_VER)\n"
      << "#define inline _inline\n"
      << "#define alloca(x) _alloca(x)\n"
      << "#else\n"
      << "#include <alloca.h>\n"
      << "#endif\n\n";

  // We output GCC specific attributes to preserve 'linkonce'ness on globals.
  // If we aren't being compiled with GCC, just drop these attributes.
  Out << "#ifndef __GNUC__  /* Can only support \"linkonce\" vars with GCC */\n"
      << "#define __attribute__(X)\n"
      << "#endif\n\n";

  // On Mac OS X, "external weak" is spelled "__attribute__((weak_import))".
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak_import))\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __EXTERNAL_WEAK__\n"
      << "#endif\n\n";

  // For now, turn off the weak linkage attribute on Mac OS X. (See above.)
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __ATTRIBUTE_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#endif\n\n";

  // Add hidden visibility support. FIXME: APPLE_CC?
  Out << "#if defined(__GNUC__)\n"
      << "#define __HIDDEN__ __attribute__((visibility(\"hidden\")))\n"
      << "#endif\n\n";

  // Define NaN and Inf as GCC builtins if using GCC, as 0 otherwise
  // From the GCC documentation:
  //
  //   double __builtin_nan (const char *str)
  //
  // This is an implementation of the ISO C99 function nan.
  //
  // Since ISO C99 defines this function in terms of strtod, which we do
  // not implement, a description of the parsing is in order. The string is
  // parsed as by strtol; that is, the base is recognized by leading 0 or
  // 0x prefixes. The number parsed is placed in the significand such that
  // the least significant bit of the number is at the least significant
  // bit of the significand. The number is truncated to fit the significand
  // field provided. The significand is forced to be a quiet NaN.
  //
  // This function, if given a string literal, is evaluated early enough
  // that it is considered a compile-time constant.
  //
  //   float __builtin_nanf (const char *str)
  //
  // Similar to __builtin_nan, except the return type is float.
  //
  //   double __builtin_inf (void)
  //
  // Similar to __builtin_huge_val, except a warning is generated if the
  // target floating-point format does not support infinities. This
  // function is suitable for implementing the ISO C99 macro INFINITY.
  //
  //   float __builtin_inff (void)
  //
  // Similar to __builtin_inf, except the return type is float.
  Out << "#ifdef __GNUC__\n"
      << "#define LLVM_NAN(NanStr)   __builtin_nan(NanStr)   /* Double */\n"
      << "#define LLVM_NANF(NanStr)  __builtin_nanf(NanStr)  /* Float */\n"
      << "#define LLVM_NANS(NanStr)  __builtin_nans(NanStr)  /* Double */\n"
      << "#define LLVM_NANSF(NanStr) __builtin_nansf(NanStr) /* Float */\n"
      << "#define LLVM_INF           __builtin_inf()         /* Double */\n"
      << "#define LLVM_INFF          __builtin_inff()        /* Float */\n"
      << "#define LLVM_PREFETCH(addr,rw,locality) "
      "__builtin_prefetch(addr,rw,locality)\n"
      << "#define __ATTRIBUTE_CTOR__ __attribute__((constructor))\n"
      << "#define __ATTRIBUTE_DTOR__ __attribute__((destructor))\n"
      << "#define LLVM_ASM           __asm__\n"
      << "#else\n"
      << "#define LLVM_NAN(NanStr)   ((double)0.0)           /* Double */\n"
      << "#define LLVM_NANF(NanStr)  0.0F                    /* Float */\n"
      << "#define LLVM_NANS(NanStr)  ((double)0.0)           /* Double */\n"
      << "#define LLVM_NANSF(NanStr) 0.0F                    /* Float */\n"
      << "#define LLVM_INF           ((double)0.0)           /* Double */\n"
      << "#define LLVM_INFF          0.0F                    /* Float */\n"
      << "#define LLVM_PREFETCH(addr,rw,locality)            /* PREFETCH */\n"
      << "#define __ATTRIBUTE_CTOR__\n"
      << "#define __ATTRIBUTE_DTOR__\n"
      << "#define LLVM_ASM(X)\n"
      << "#endif\n\n";

  Out << "#if __GNUC__ < 4 /* Old GCC's, or compilers not GCC */ \n"
      << "#define __builtin_stack_save() 0   /* not implemented */\n"
      << "#define __builtin_stack_restore(X) /* noop */\n"
      << "#endif\n\n";

  // Output typedefs for 128-bit integers. If these are needed with a
  // 32-bit target or with a C compiler that doesn't support mode(TI),
  // more drastic measures will be needed.
  Out << "#if __GNUC__ && __LP64__ /* 128-bit integer types */\n"
      << "typedef int __attribute__((mode(TI))) llvmInt128;\n"
      << "typedef unsigned __attribute__((mode(TI))) llvmUInt128;\n"
      << "#endif\n\n";

  // Output target-specific code that should be inserted into main.
  Out << "#define CODE_FOR_MAIN() /* Any target-specific code for main()*/\n";
#endif
}

/// FindStaticTors - Given a static ctor/dtor list, unpack its contents into
/// the StaticTors set.
static void FindStaticTors(GlobalVariable *GV, std::set<Function *> &StaticTors) {
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!InitList) {
    return;
  }

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))) {
      if (CS->getNumOperands() != 2) {
        return;  // Not array of 2-element structs.
      }

      if (CS->getOperand(1)->isNullValue()) {
        return;  // Found a null terminator, exit printing.
      }
      Constant *FP = CS->getOperand(1);
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
        if (CE->isCast()) {
          FP = CE->getOperand(0);
        }
      if (Function *F = dyn_cast<Function>(FP)) {
        StaticTors.insert(F);
      }
    }
}

enum SpecialGlobalClass {
  NotSpecial = 0,
  GlobalCtors, GlobalDtors,
  NotPrinted
};

/// getGlobalVariableClass - If this is a global that is specially recognized
/// by LLVM, return a code that indicates how we should handle it.
static SpecialGlobalClass getGlobalVariableClass(const GlobalVariable *GV) {
  // If this is a global ctors/dtors list, handle it now.
  if (GV->hasAppendingLinkage() && GV->use_empty()) {
    if (GV->getName() == "llvm.global_ctors") {
      return GlobalCtors;
    } else if (GV->getName() == "llvm.global_dtors") {
      return GlobalDtors;
    }
  }

  // Otherwise, if it is other metadata, don't print it.  This catches things
  // like debug information.
  if (strcmp(GV->getSection(), "llvm.metadata") == 0) {
    return NotPrinted;
  }

  // __local variables are printed in the function that uses them
  if (GV->getType()->getAddressSpace() == 3) {
    return NotPrinted;
  }

  return NotSpecial;
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const char *Str, unsigned Length,
                               raw_ostream &Out) {
  for (unsigned i = 0; i != Length; ++i) {
    unsigned char C = Str[i];
    if (isprint(C) && C != '\\' && C != '"') {
      Out << C;
    } else if (C == '\\') {
      Out << "\\\\";
    } else if (C == '\"')
      Out << "\\\"";
    else if (C == '\t') {
      Out << "\\t";
    } else {
      Out << "\\x" << hexdigit(C >> 4) << hexdigit(C & 0x0F);
    }
  }
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const std::string &Str, raw_ostream &Out) {
  PrintEscapedString(Str.c_str(), Str.size(), Out);
}

void CWriter::printInitializer(GlobalVariable *I) {
  if (!I) {
    return;
  }

  // If the initializer is not null, emit the initializer.  If it is null,
  // we try to avoid emitting large amounts of zeros.  The problem with
  // this, however, occurs when the variable has weak linkage.  In this
  // case, the assembler will complain about the variable being both weak
  // and common, so we disable this optimization.
  // FIXME common linkage should avoid this problem.
  if (I->hasInitializer() && !I->getInitializer()->isNullValue()) {
    Out << " = " ;
    writeOperand(I->getInitializer(), true);
  } else if (I->hasWeakLinkage()) {
    // We have to specify an initializer, but it doesn't have to be
    // complete.  If the value is an aggregate, print out { 0 }, and let
    // the compiler figure out the rest of the zeros.
    Out << " = " ;
    if (I->getInitializer()->getType()->isStructTy() ||
        I->getInitializer()->getType()->isVectorTy()) {
      Out << "{ 0 }";
    } else if (I->getInitializer()->getType()->isArrayTy()) {
      // As with structs and vectors, but with an extra set of braces
      // because arrays are wrapped in structs.
      Out << "{ { 0 } }";
    } else {
      // Just print it out normally.
      writeOperand(I->getInitializer(), true);
    }
  }
}

bool CWriter::doInitialization(Module &M) {
  KernelFnCallees.clear();
  InlinedCallees.clear();
  NonInlinedCallees.clear();
  kernelList.clear();
  work_group_list.clear();
  arg_addr_space_list.clear();
  arg_access_qual_list.clear();
  arg_type_list.clear();
  arg_type_qual_list.clear();
  arg_name_list.clear();
  LaunchKernels.clear();

  FunctionPass::doInitialization(M);

  // Initialize
  TheModule = &M;
  StringRef filename = llvm::sys::path::filename(M.getModuleIdentifier());
  if ("<stdin>" == filename) {
    bcHash = "";
  } else {
    bcHash = filename.substr(0, filename.find("."));
  }

  TD = new DataLayout(&M);
  IL = new IntrinsicLowering(*TD);
  IL->AddPrototypes(M);

#if 0
  std::string Triple = TheModule->getTargetTriple();
  if (Triple.empty()) {
    Triple = llvm::sys::getDefaultTargetTriple();
  }

  std::string E;
  if (const Target *Match = TargetRegistry::lookupTarget(Triple, E)) {
    TAsm = Match->createMCAsmInfo(Triple);
  }
#endif
  TAsm = new CBEMCAsmInfo();
  MRI  = new MCRegisterInfo();
  TCtx = new MCContext(TAsm, MRI, NULL);
  Mang = new Mangler(TD);

  // Keep track of which functions are static ctors/dtors so they can have
  // an attribute added to their prototypes.
  std::set<Function *> StaticCtors, StaticDtors;
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    switch (getGlobalVariableClass(I)) {
      default:
        break;
      case GlobalCtors:
        FindStaticTors(I, StaticCtors);
        break;
      case GlobalDtors:
        FindStaticTors(I, StaticDtors);
        break;
    }
  }

  // get declaration for alloca
#if 0
  Out << "/* Provide Declarations */\n";
  Out << "#include <stdarg.h>\n";      // Varargs support
  Out << "#include <setjmp.h>\n";      // Unwind support
  Out << "#include <limits.h>\n";      // With overflow intrinsics support.
#endif
#ifdef MXPA_CODEGEN
  Out << "\n#ifdef QUERIES\n";
  Out << "#include <stdbool.h>\n";
  Out << "static inline int mxpa_smin(int a, int b) { return a>b?b:a; }\n"
      << "static inline int mxpa_smax(int a, int b) { return a>b?a:b; }\n"
      << "static inline int mxpa__Z5mad24iii(int a, int b, int c) { return a * b + c; }\n"
      << "#define get_global_id(x)   (__mxpa_local_id[(x)]+__mxpa_group_offset[(x)])\n"
      << "#define get_global_size(x) (__mxpa_global_size[(x)])\n"
      << "#define get_group_id(x) (__mxpa_group_id[(x)])\n"
      << "#define get_local_id(x) (__mxpa_local_id[(x)])\n"
      << "#define get_local_size(x) (__mxpa_local_size[(x)])\n"
      << "#define get_global_offset(x) (__mxpa_global_offset[(x)])\n"
      << "#define get_work_dim() (__mxpa_work_dim)\n"
      << "#define get_num_groups(x) (__mxpa_global_size[(x)] / __mxpa_local_size[(x)])\n"
      << "#define mxpa_get_num_groups(x) (global_size[(x)] / local_size[(x)])\n"
      << "#define mxpa_get_global_size(x) global_size[x]\n";
  Out << "#endif\n";
#endif
  generateCompilerSpecificCode(Out, TD);

  // Provide a definition for `bool' if not compiling with a C++ compiler.
  Out << "\n"
      << "\n\n/* Support for floating point constants */\n"
      << "typedef unsigned long ConstantDoubleTy;\n"
      << "typedef unsigned int ConstantFloatTy;\n"
      << "\n\n/* Global Declarations */\n";
  // First output all the declarations for the program, because C requires
  // Functions & globals to be declared before they are used.
  //
  if (!M.getModuleInlineAsm().empty()) {
    Out << "/* Module asm statements */\n"
        << "asm(";

    // Split the string into lines, to make it easier to read the .ll file.
    std::string Asm = M.getModuleInlineAsm();
    size_t CurPos = 0;
    size_t NewLine = Asm.find_first_of('\n', CurPos);
    while (NewLine != std::string::npos) {
      // We found a newline, print the portion of the asm string from the
      // last newline up to this newline.
      Out << "\"";
      PrintEscapedString(std::string(Asm.begin() + CurPos, Asm.begin() + NewLine),
                         Out);
      Out << "\\n\"\n";
      CurPos = NewLine + 1;
      NewLine = Asm.find_first_of('\n', CurPos);
    }
    Out << "\"";
    PrintEscapedString(std::string(Asm.begin() + CurPos, Asm.end()), Out);
    Out << "\");\n"
        << "/* End Module asm statements */\n";
  }

  // Loop over the symbol table, emitting all named constants.
  printModuleTypes();

#if 0 //there shouldn't be globals in OpenCL
  // Global variable declarations...
  if (!M.global_empty()) {
    Out << "\n/* External Global Variable Declarations */\n";
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {

      if (I->hasExternalLinkage() || I->hasExternalWeakLinkage() ||
          I->hasCommonLinkage()) {
        Out << "extern ";
        if (I->getInitializer()->getType()->isStructTy()) {
          Out << "struct ";
        }
      } else if (I->hasDLLImportLinkage()) {
        Out << "__declspec(dllimport) ";
      } else {
        continue;  // Internal Global
      }

      // Thread Local Storage
      if (I->isThreadLocal()) {
        Out << "__thread ";
      }

      printType(Out, I->getType()->getElementType(), false, GetValueName(I));

      if (I->hasExternalWeakLinkage()) {
        Out << " __EXTERNAL_WEAK__";
      }
      Out << ";\n";
    }
  }
#endif

  // Function declarations
#if 0
  Out << "\n/* Function Declarations */\n";
  Out << "double fmod(double, double);\n";   // Support for FP rem
  Out << "float fmodf(float, float);\n";
  Out << "long double fmodl(long double, long double);\n";
#endif
  // Store the intrinsics which will be declared/defined below.
  SmallVector<const Function *, 8> intrinsicsToDefine;

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    // Don't print declarations for intrinsic functions.
    // Store the used intrinsics, which need to be explicitly defined.
    if (I->isIntrinsic()) {
      switch (I->getIntrinsicID()) {
        default:
          break;
        case Intrinsic::uadd_with_overflow:
        case Intrinsic::sadd_with_overflow:
          intrinsicsToDefine.push_back(I);
          break;
      }
      continue;
    }

    if (I->getName() == "setjmp" ||
        I->getName() == "longjmp" || I->getName() == "_setjmp") {
      continue;
    }

    if (I->hasExternalWeakLinkage()) {
      Out << "extern ";
    }
    if (CompileOpenCLKernel) {
      printFunctionSignature(I, true);
    }
    //if (I->hasWeakLinkage() || I->hasLinkOnceLinkage())
    //  Out << " __ATTRIBUTE_WEAK__";
    if (I->hasExternalWeakLinkage()) {
      Out << " __EXTERNAL_WEAK__";
    }
    if (StaticCtors.count(I)) {
      Out << " __ATTRIBUTE_CTOR__";
    }
    if (StaticDtors.count(I)) {
      Out << " __ATTRIBUTE_DTOR__";
    }
    if (I->hasHiddenVisibility()) {
      Out << " __HIDDEN__";
    }

    if (I->hasName() && I->getName()[0] == 1) {
      Out << " LLVM_ASM(\"" << I->getName().substr(1) << "\")";
    }

    if (CompileOpenCLKernel) {
      Out << ";\n";
    }
  }
#if 0
  // Output the global variable declarations
  if (!M.global_empty()) {
    Out << "\n\n/* Global Variable Declarations */\n";
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I)
      if (!I->isDeclaration()) {
        // Ignore special globals, such as debug info.
        if (getGlobalVariableClass(I)) {
          continue;
        }

        if (I->hasLocalLinkage()) {
          Out << "static struct ";
        } else {
          Out << "extern ";
        }

        // Thread Local Storage
        if (I->isThreadLocal()) {
          Out << "__thread ";
        }

        printType(Out, I->getType()->getElementType(), false,
                  GetValueName(I));

        if (I->hasLinkOnceLinkage()) {
          Out << " __attribute__((common))";
        } else if (I->hasCommonLinkage()) { // FIXME is this right?
          Out << " __ATTRIBUTE_WEAK__";
        } else if (I->hasWeakLinkage()) {
          Out << " __ATTRIBUTE_WEAK__";
        } else if (I->hasExternalWeakLinkage()) {
          Out << " __EXTERNAL_WEAK__";
        }
        if (I->hasHiddenVisibility()) {
          Out << " __HIDDEN__";
        }
        Out << ";\n";
      }
  }
#endif
  // Output the global variable definitions and contents...
  if (!M.global_empty() && CompileOpenCLKernel) {
    Out << "\n\n/* Global Variable Definitions and Initialization */\n";
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I)
      if (!I->isDeclaration()) {
        // Ignore special globals, such as debug info.
        if (getGlobalVariableClass(I)) {
          continue;
        }
        // OpenCL allows only __constants in global scope
        if (!I->isConstant()) {
          continue;
        }
        if (isa<Constant>(I)) {
          Out << "__constant ";
          printType(Out, I->getType()->getElementType(), false,

                    GetValueName(I));
        } else {
          Out << "__constant struct ";
          printType(Out, I->getType()->getElementType(), false,
                    GetValueName(I));
        }
        // If the initializer is not null, emit the initializer.  If it is null,
        // we try to avoid emitting large amounts of zeros.  The problem with
        // this, however, occurs when the variable has weak linkage.  In this
        // case, the assembler will complain about the variable being both weak
        // and common, so we disable this optimization.
        // FIXME common linkage should avoid this problem.
        if (I->hasInitializer() && !I->getInitializer()->isNullValue()) {
          Out << " = " ;
          writeOperand(I->getInitializer(), true);
        } else if (I->hasWeakLinkage()) {
          // We have to specify an initializer, but it doesn't have to be
          // complete.  If the value is an aggregate, print out { 0 }, and let
          // the compiler figure out the rest of the zeros.
          Out << " = " ;
          if (I->getInitializer()->getType()->isStructTy() ||
              I->getInitializer()->getType()->isVectorTy()) {
            Out << "{ 0 }";
          } else if (I->getInitializer()->getType()->isArrayTy()) {
            // As with structs and vectors, but with an extra set of braces
            // because arrays are wrapped in structs.
            Out << "{ { 0 } }";
          } else {
            // Just print it out normally.
            writeOperand(I->getInitializer(), true);
          }
        }
        Out << ";\n";
      }
  }
  if (!M.empty()) {
    Out << "\n\n/* Function Bodies */\n";
  }

  // Emit some helper functions for dealing with FCMP instruction's
  // predicates
#if 0
  Out << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
#endif
  Out << "static inline int llvm_fcmp_ord(float X, float Y) { ";
  Out << "return X == X && Y == Y; }\n";
  Out << "static inline int llvm_fcmp_uno(float X, float Y) { ";
  Out << "return X != X || Y != Y; }\n";
  Out << "static inline int llvm_fcmp_ueq(float X, float Y) { ";
  Out << "return X == Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_une(float X, float Y) { ";
  Out << "return X != Y; }\n";
  Out << "static inline int llvm_fcmp_ult(float X, float Y) { ";
  Out << "return X <  Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_ugt(float X, float Y) { ";
  Out << "return X >  Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_ule(float X, float Y) { ";
  Out << "return X <= Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_uge(float X, float Y) { ";
  Out << "return X >= Y || llvm_fcmp_uno(X, Y); }\n";
  Out << "static inline int llvm_fcmp_oeq(float X, float Y) { ";
  Out << "return X == Y ; }\n";
  Out << "static inline int llvm_fcmp_one(float X, float Y) { ";
  Out << "return X != Y && llvm_fcmp_ord(X, Y); }\n";
  Out << "static inline int llvm_fcmp_olt(float X, float Y) { ";
  Out << "return X <  Y ; }\n";
  Out << "static inline int llvm_fcmp_ogt(float X, float Y) { ";
  Out << "return X >  Y ; }\n";
  Out << "static inline int llvm_fcmp_ole(float X, float Y) { ";
  Out << "return X <= Y ; }\n";
  Out << "static inline int llvm_fcmp_oge(float X, float Y) { ";
  Out << "return X >= Y ; }\n";

  // Emit definitions of the intrinsics.
  for (SmallVector<const Function *, 8>::const_iterator
       I = intrinsicsToDefine.begin(),
       E = intrinsicsToDefine.end(); I != E; ++I) {
    printIntrinsicDefinition(**I, Out);
  }

  return false;
}


/// Output all floating point constants that cannot be printed accurately...
void CWriter::printFloatingPointConstants(Function &F) {
  // Scan the module for floating point constants.  If any FP constant is used
  // in the function, we want to redirect it here so that we do not depend on
  // the precision of the printed form, unless the printed form preserves
  // precision.
  //
  for (constant_iterator I = constant_begin(&F), E = constant_end(&F);
       I != E; ++I) {
    if (const ConstantDataSequential *CDS = dyn_cast<ConstantDataSequential>(*I)) {
      printFloatingPointConstantDataSequentials(CDS);
    } else {
      printFloatingPointConstants(*I);
    }
  }

  //Out << '\n';
}

void CWriter::printFloatingPointConstants(const Constant *C) {
  // If this is a constant expression, recursively check for constant fp values.
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i) {
      printFloatingPointConstants(CE->getOperand(i));
    }
    return;
  }

  // Otherwise, check for a FP constant that we need to print.
  const ConstantFP *FPC = dyn_cast<ConstantFP>(C);
  if (FPC == 0 ||
      // Do not put in FPConstantMap if safe.
      isFPCSafeToPrint(FPC) ||
      // Already printed this constant?
      FPConstantMap.count(FPC)) {
    return;
  }

  FPConstantMap[FPC] = FPCounter;  // Number the FP constants

  if (FPC->getType() == Type::getDoubleTy(FPC->getContext())) {
    double Val = FPC->getValueAPF().convertToDouble();
    uint64_t i = FPC->getValueAPF().bitcastToAPInt().getZExtValue();
    Out << "#ifdef __OPENCL_VERSION__\n"
        << "static __constant ConstantDoubleTy FPConstant" << FPCounter
        << " = 0x" << utohexstr(i)
        << "ULL;    /* " << Val << " */\n"
        << "#else\n"
        << "static const ConstantDoubleTy FPConstant" << FPCounter
        << " = 0x" << utohexstr(i)
        << "ULL;    /* " << Val << " */\n"
        << "#endif";
    FPCounter += 1;
  } else if (FPC->getType() == Type::getFloatTy(FPC->getContext())) {
    float Val = FPC->getValueAPF().convertToFloat();
    uint32_t i = (uint32_t)FPC->getValueAPF().bitcastToAPInt().
                 getZExtValue();
    Out << "#ifdef __OPENCL_VERSION__\n"
        << "static __constant ConstantFloatTy FPConstant" << FPCounter
        << " = 0x" << utohexstr(i)
        << "U;    /* " << Val << " */\n"
        << "#else\n"
        << "static const ConstantFloatTy FPConstant" << FPCounter
        << " = 0x" << utohexstr(i)
        << "U;    /* " << Val << " */\n"
        << "#endif\n";
    FPCounter += 1;
  } else {
    llvm_unreachable("Unknown float type!");
  }
}

void CWriter::printFloatingPointConstantDataSequentials(
  const ConstantDataSequential *CDS) {
  for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
    printFloatingPointConstants(CDS->getElementAsConstant(i));
  }
}

/// printSymbolTable - Run through symbol table looking for type names.  If a
/// type name is found, emit its declaration...
///
void CWriter::printModuleTypes() {
#if 0
  Out << "/* Helper union for bitcasts */\n";
  Out << "typedef union {\n";
  Out << "  unsigned int Int32;\n";
  Out << "  unsigned long long Int64;\n";
  Out << "  float Float;\n";
  Out << "  double Double;\n";
  Out << "} llvmBitCastUnion;\n";
#endif
  // Get all of the struct types used in the module.
  run(*TheModule, false);

  if (StructTypes.empty()) {
    return;
  }

  Out << "/* Structure forward decls */\n";

  unsigned NextTypeID = 0;

  // If any of them are missing names, add a unique ID to UnnamedStructIDs.
  // Print out forward declarations for structure types.
  for (unsigned i = 0, e = StructTypes.size(); i != e; ++i) {
    StructType *ST = StructTypes[i];

    if (ST->isLiteral() || ST->getName().empty()) {
      UnnamedStructIDs[ST] = NextTypeID++;
    }

    std::string Name = getStructName(ST);
    Out << "struct " << Name << ";\n";
  }

  Out << '\n';

  // Now we can print out typedefs.  Above, we guaranteed that this can only be
  // for struct or opaque types.
  Out << "/* Typedefs */\n";
  for (unsigned i = 0, e = StructTypes.size(); i != e; ++i) {
    StructType *ST = StructTypes[i];
    std::string Name = getStructName(ST);
    Out << "typedef struct " << Name << ' ' << Name << ";\n";
  }

  Out << '\n';

  // Keep track of which structures have been printed so far.
  SmallPtrSet<Type *, 16> StructPrinted;

  // Loop over all structures then push them into the stack so they are
  // printed in the correct order.
  //
  Out << "/* Structure contents */\n";
  for (unsigned i = 0, e = StructTypes.size(); i != e; ++i)
    if (StructTypes[i]->isStructTy())
      // Only print out used types!
    {
      printContainedStructs(StructTypes[i], StructPrinted);
    }
}

// Push the struct onto the stack and recursively push all structs
// this one depends on.
//
// TODO:  Make this work properly with vector types
//
void CWriter::printContainedStructs(Type *Ty,
                                    SmallPtrSet<Type *, 16> &StructPrinted) {
  // Don't walk through pointers.
  if (Ty->isPointerTy() || (Ty->getTypeID() >= 0 && Ty->getTypeID() <= 9) || Ty->isIntegerTy()) {
    return;
  }

  // Walk through arrays, declaring their container structs.
  if (Ty->isArrayTy()) {
    ArrayType *ATy = cast<ArrayType>(Ty);
    std::vector<Type *> structMembers;
    structMembers.push_back(ATy);
    StructType *ST = StructType::get(TheModule->getContext(), structMembers);

    // Check to see if we have already printed this struct.
    if (!StructPrinted.insert(ST)) {
      return;
    }

    // Print all contained types first.
    for (Type::subtype_iterator I = Ty->subtype_begin(),
         E = Ty->subtype_end(); I != E; ++I) {
      printContainedStructs(*I, StructPrinted);
    }

    // Print structure type out.
    printType(Out, ST, false, getStructName(ST), true);
    Out << ";\n\n";
  }
  // Print all contained types first.
  for (Type::subtype_iterator I = Ty->subtype_begin(),
       E = Ty->subtype_end(); I != E; ++I) {
    printContainedStructs(*I, StructPrinted);
  }

  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    // Check to see if we have already printed this struct.
    if (!StructPrinted.insert(Ty)) {
      return;
    }

    // Print structure type out.
    printType(Out, ST, false, getStructName(ST), true);
    Out << ";\n\n";
  }
}

bool CWriter::collectKernelInfo() {
  NamedMDNode *openCLMetadata = TheModule->getNamedMetadata("opencl.kernels");
  if (!openCLMetadata) {
    return false;
  }
  unsigned kernelNum = openCLMetadata->getNumOperands();
  for (unsigned K = 0, E = kernelNum; K != E; ++K) {
    MDNode *kernelMD =  dyn_cast<MDNode>(openCLMetadata->getOperand(K));
    // FIXME: The kernel is removed in kernel_*.cl.c.mxpa.bc. This is a work
    //        around solution.
    if (!kernelMD->getOperand(0)) {
      continue;
    }
    Function *fun = dyn_cast<Function>(kernelMD->getOperand(0));
    unsigned num_md = kernelMD->getNumOperands();
    // Get number of metadata this kernel has
    std::vector<std::string> arg_addr_space;
    std::vector<std::string> arg_access_qual;
    std::vector<std::string> arg_type;
    std::vector<std::string> arg_type_qual;
    std::vector<std::string> arg_name;
    std::vector<unsigned int> workgroups(3, 0);
    for (unsigned A = 1; A < num_md; ++A) {
      MDNode *info_Nd = dyn_cast<MDNode>(kernelMD->getOperand(A));
      std::string infoName = info_Nd->getOperand(0)->getName().str();
      unsigned arg_num = info_Nd->getNumOperands();
      if (infoName.compare("kernel_arg_addr_space") == 0) {
        for (unsigned int i = 1; i < arg_num; ++i) {
          std::string info = "";
          ConstantInt *CI = dyn_cast<ConstantInt>(info_Nd->getOperand(i));
          if (CI->equalsInt(0)) {
            info = "0X119E";
          } else if (CI->equalsInt(1)) {
            info = "0X119B";
          } else if (CI->equalsInt(2)) {
            info = "0X119D";
          } else if (CI->equalsInt(3)) {
            info = "0X119C";
          }
          arg_addr_space.push_back(info);
        }
      } else if (infoName.compare("kernel_arg_access_qual") == 0) {
        for (unsigned int i = 1; i < arg_num; ++i) {
          std::string info = "";
          StringRef SR = info_Nd->getOperand(i)->getName();
          if (SR.equals("read only") || SR.equals("READ ONLY")) {
            info = "0X11A0";
          } else if (SR.equals("write only") || SR.equals("WRITE ONLY")) {
            info = "0X11A1";
          } else if (SR.equals("read write") || SR.equals("READ WRITE")) {
            info = "0X11A2";
          } else if (SR.equals("none") || SR.equals("NONE")) {
            info = "0X11A3";
          }
          arg_access_qual.push_back(info);
        }
      } else if (infoName.compare("kernel_arg_type") == 0) {
        for (unsigned int i = 1; i < arg_num; ++i) {
          arg_type.push_back(info_Nd->getOperand(i)->getName());
        }
      } else if (infoName.compare("kernel_arg_type_qual") == 0) {
        for (unsigned int i = 1; i < arg_num; ++i) {
          std::string type_qual = info_Nd->getOperand(i)->getName();
          unsigned int temp_val = 0;
          if (type_qual.find("const") != std::string::npos) {
            temp_val += 1;
          }
          if (type_qual.find("restrict") != std::string::npos) {
            temp_val += 2;
          }
          if (type_qual.find("volatile") != std::string::npos) {
            temp_val += 4;
          }
          std::stringstream ss;
          ss << temp_val;
          arg_type_qual.push_back(ss.str());
        }
      } else if (infoName.compare("kernel_arg_name") == 0) {
        for (unsigned int i = 1; i < arg_num; ++i) {
          arg_name.push_back(info_Nd->getOperand(i)->getName());
        }
      } else if (infoName.compare("reqd_work_group_size") == 0) {
        ConstantInt *constfirst  = dyn_cast<ConstantInt>(info_Nd->getOperand(1));
        ConstantInt *constsecond = dyn_cast<ConstantInt>(info_Nd->getOperand(2));
        ConstantInt *constthird  = dyn_cast<ConstantInt>(info_Nd->getOperand(3));
        // sext or zext?
        unsigned int first = (unsigned int)constfirst->getZExtValue();
        unsigned int second = (unsigned int)constsecond->getZExtValue();
        unsigned int third = (unsigned int)constthird->getZExtValue();
        workgroups.at(0) = first;
        workgroups.at(1) = second;
        workgroups.at(2) = third;
        llvm::errs() << "workgroup: " << first << " " << second << " " << third << "\n";
      }
    } // end of loop
    kernelList.push_back(fun->getName());
    work_group_list.push_back(workgroups);
    arg_addr_space_list.push_back(arg_addr_space);
    arg_access_qual_list.push_back(arg_access_qual);
    arg_type_list.push_back(arg_type);
    arg_type_qual_list.push_back(arg_type_qual);
    arg_name_list.push_back(arg_name);
  }
  return false;
}

bool CWriter::isKernelFunction(const Function *F) {
  if (!F) {
    return false;
  }
  llvm::NamedMDNode *openCLMetadata = TheModule->getNamedMetadata("opencl.kernels");
  if (!openCLMetadata) {
    return false;
  }

  for (unsigned K = 0, E = openCLMetadata->getNumOperands(); K != E; ++K) {
    llvm::MDNode &kernelMD = *openCLMetadata->getOperand(K);
    if (kernelMD.getOperand(0) == F) {
      return true;
    }
  }
  return false;
}

void CWriter::printFunctionSignature(const Function *F, bool Prototype) {
  /// isStructReturn - Should this function actually return a struct by-value?
  bool isStructReturn = F->hasStructRetAttr();

  if (F->hasLocalLinkage() && !isKernelFunction(F)) {
    Out << "static ";
  }
  // No dllimport and dllexport linkage in SPIR 1.2
  //if (F->hasDLLImportLinkage()) Out << "__declspec(dllimport) ";
  //if (F->hasDLLExportLinkage()) Out << "__declspec(dllexport) ";
  switch (F->getCallingConv()) {
    case CallingConv::X86_StdCall:
      Out << "__attribute__((stdcall)) ";
      break;
    case CallingConv::X86_FastCall:
      Out << "__attribute__((fastcall)) ";
      break;
    case CallingConv::X86_ThisCall:
      Out << "__attribute__((thiscall)) ";
      break;
    default:
      break;
  }

  // Loop over the arguments, printing them...
  FunctionType *FT = cast<FunctionType>(F->getFunctionType());
  const AttributeSet &PAL = F->getAttributes();

  std::string tstr;
  raw_string_ostream FunctionInnards(tstr);

  // Print out the name...
  // Split typen and its function signature. (e.g., char2vload2() ->
  // char2 vload2())
  FunctionInnards << " " << GetValueName(F) << '(';

  // Print the keyword of kernel
  if (isKernelFunction(F)) {
    Out << "__kernel ";
  }
  bool PrintedArg = false;
  if (!F->isDeclaration()) {
    if (!F->arg_empty()) {
      Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
      unsigned Idx = 1;

      // If this is a struct-return function, don't print the hidden
      // struct-return argument.
      if (isStructReturn) {
        assert(I != E && "Invalid struct return function!");
        ++I;
        ++Idx;
      }

      std::string ArgName;
      for (; I != E; ++I) {
        if (PrintedArg) {
          FunctionInnards << ", ";
        }
        if (I->hasName() || !Prototype) {
          ArgName = GetValueName(I);
        } else {
          ArgName = "";
        }
        Type *ArgTy = I->getType();
        if (PAL.hasAttribute(Idx, Attribute::ByVal)) {
          ArgTy = cast<PointerType>(ArgTy)->getElementType();
          ByValParams.insert(I);
        }
        printType(FunctionInnards, ArgTy,
                  /*isSigned=*/PAL.hasAttribute(Idx, Attribute::SExt),
                  ArgName);
        PrintedArg = true;
        ++Idx;
      }
    }
  } else {
    // Loop over the arguments, printing them.
    FunctionType::param_iterator I = FT->param_begin(), E = FT->param_end();
    unsigned Idx = 1;

    // If this is a struct-return function, don't print the hidden
    // struct-return argument.
    if (isStructReturn) {
      assert(I != E && "Invalid struct return function!");
      ++I;
      ++Idx;
    }

    for (; I != E; ++I) {
      if (PrintedArg) {
        FunctionInnards << ", ";
      }
      Type *ArgTy = *I;
      if (PAL.hasAttribute(Idx, Attribute::ByVal)) {
        assert(ArgTy->isPointerTy());
        ArgTy = cast<PointerType>(ArgTy)->getElementType();
      }
      printType(FunctionInnards, ArgTy,
                /*isSigned=*/PAL.hasAttribute(Idx, Attribute::SExt));
      PrintedArg = true;
      ++Idx;
    }
  }

  if (!PrintedArg && FT->isVarArg()) {
    FunctionInnards << "int vararg_dummy_arg";
    PrintedArg = true;
  }

  // Finish printing arguments... if this is a vararg function, print the ...,
  // unless there are no known types, in which case, we just emit ().
  //
  if (FT->isVarArg() && PrintedArg) {
    FunctionInnards << ",...";  // Output varargs portion of signature!
  } else if (!FT->isVarArg() && !PrintedArg) {
    FunctionInnards << "void"; // ret() -> ret(void) in C.
  }
  FunctionInnards << ')';

  // Get the return tpe for the function.
  Type *RetTy;
  if (!isStructReturn) {
    RetTy = F->getReturnType();
  } else {
    // If this is a struct-return function, print the struct-return type.
    RetTy = cast<PointerType>(FT->getParamType(0))->getElementType();
  }

  // Print out the return type and the signature built above.
  printType(Out, RetTy,
            /*isSigned=*/PAL.hasAttribute(0, Attribute::SExt),
            FunctionInnards.str());
}

static inline bool isFPIntBitCast(const Instruction &I) {
  if (!isa<BitCastInst>(I)) {
    return false;
  }
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DstTy = I.getType();
  return (SrcTy->isFloatingPointTy() && DstTy->isIntegerTy()) ||
         (DstTy->isFloatingPointTy() && SrcTy->isIntegerTy());
}

static void FindLocalName(Instruction *I, Value *&LocalValue, Type *&LocalTy) {
  if (!I) {
    return;
  }
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    if (GEP->getPointerAddressSpace() == 3) { // OpenCL __local?
      if (isa<GlobalValue>(GEP->getPointerOperand())) {
        PointerType  *PTy = cast<PointerType>(GEP->getPointerOperandType());
        LocalTy = PTy->getElementType();
        LocalValue = I->getOperand(0);
      }
    }
  } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (LI->getPointerAddressSpace() == 3) {
      Value *PO = LI->getPointerOperand();
      if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(PO)) {
        switch (CE->getOpcode()) {
          case Instruction::GetElementPtr: {
              PointerType  *PTy = cast<PointerType>(
                                    CE->getOperand(0)->getType());
              LocalTy = PTy->getElementType();
              LocalValue = CE->getOperand(0);
            }
            break;
          default:
            assert(0 && "Unhandled type of ConstantExpr in a load");
        };
      } else if (isa<GetElementPtrInst>(PO)) {
        FindLocalName(dyn_cast<Instruction>(PO), LocalValue, LocalTy);
      } else if (isa<PHINode>(PO)) {
        // The sources must have been processed. Do nothing.
      } else if (dyn_cast<LoadInst>(PO)) {
        // Sample:
        //  %21 = load i64 addrspace(3)* %20, align 4
        //  %20 = bitcast %struct.UDD addrspace(3)* %arrayidx.i to i64 addrspace(3)*
        FindLocalName(dyn_cast<Instruction>(PO), LocalValue, LocalTy);
      } else if (dyn_cast<StoreInst>(PO)) {
        FindLocalName(dyn_cast<Instruction>(PO), LocalValue, LocalTy);
      } else {
        if (PointerType *PTy = cast<PointerType>(PO->getType())) {
          LocalTy = PTy->getElementType();
          LocalValue = PO;
        } else {
          LI->dump();
          PO->dump();
          assert(0 && "LI: Unhandled type of reference to a local array");
        }
      }
    }
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (SI->getPointerAddressSpace() == 3) {
      if (isa<GlobalValue>(SI->getPointerOperand())) {
        Value *PO = SI->getPointerOperand();
        if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(PO)) {
          switch (CE->getOpcode()) {
            case Instruction::GetElementPtr: {
                PointerType  *PTy = cast<PointerType>(
                                      CE->getOperand(0)->getType());
                LocalTy = PTy->getElementType();
                LocalValue = CE->getOperand(0);
              }
              break;
            default:
              assert(0 && "Unhandled type of ConstantExpr in a store");
          };
        } else if (isa<GetElementPtrInst>(PO)) {
          FindLocalName(dyn_cast<Instruction>(PO), LocalValue, LocalTy);
        } else if (isa<PHINode>(PO)) {
          // The sources must have been processed. Do nothing.
        } else if (dyn_cast<LoadInst>(PO)) {
          FindLocalName(dyn_cast<Instruction>(PO), LocalValue, LocalTy);
        } else if (dyn_cast<StoreInst>(PO)) {
          FindLocalName(dyn_cast<Instruction>(PO), LocalValue, LocalTy);
        } else {
          PointerType  *PTy = cast<PointerType>(PO->getType());
          if (PTy) {
            LocalTy = PTy->getElementType();
            LocalValue = PO;
          } else {
            SI->dump();
            PO->dump();
            assert(0 && "SI: Unhandled type of reference to a local array");
          }
        }
      }
    }
  }
}

static bool usedInOneFunc(const User *U, Function const *&oneFunc) {
  if (const GlobalVariable *othergv = dyn_cast<GlobalVariable>(U)) {
    if (othergv->getName().str() == "llvm.used") {
      return true;
    }
  }

  if (const Instruction *instr = dyn_cast<Instruction>(U)) {
    if (instr->getParent() && instr->getParent()->getParent()) {
      const Function *curFunc = instr->getParent()->getParent();
      if (oneFunc && (curFunc != oneFunc)) {
        return false;
      }
      oneFunc = curFunc;
      return true;
    } else {
      return false;
    }
  }

  if (const MDNode *md = dyn_cast<MDNode>(U))
    if (md->hasName() && ((md->getName().str() == "llvm.dbg.gv") ||
                          (md->getName().str() == "llvm.dbg.sp"))) {
      return true;
    }

#if LLVM_VERSION_MAJOR == 3
#if LLVM_VERSION_MINOR == 3
  for (User::const_use_iterator ui = U->use_begin(), ue = U->use_end();
       ui != ue; ++ui) {
    if (usedInOneFunc(*ui, oneFunc) == false) {
      return false;
    }
  }
#elif LLVM_VERSION_MINOR == 5
  for (const User * u : U->users()) {
    if (usedInOneFunc(u, oneFunc) == false) {
      return false;
    }
  }
#else
#error Unsupported LLVM MINOR VERSION
#endif
#else
#error Unsupported LLVM MAJOR VERSION
#endif

  return true;
}

// Variables inside a __kernel function not declared with an address space qualifier,
// all variables inside non-kernel functions, and all function arguments are in
// the __private or private address space
void  CWriter::printFunctionPrivateVaraibles(Function &F) {
  // We only care Kernel Function
  if (!isKernelFunction(&F)) {
    return;
  }

  Module *M = F.getParent();
  if (M && !M->global_empty()) {
    // Find out if a static global variable can be demoted to Function's scope.
    // The conditions to check are,
    // 1. Is the global variable in private address space?
    // 2. Does it have internal linkage?
    // 3. Is the global variable referenced only in one function?
    // FIXME: private linkage should be handled as well
    for (Module::global_iterator I = M->global_begin(), E = M->global_end();
         I != E; ++I) {
      if (I->hasInternalLinkage() &&  I->getType()->getPointerAddressSpace() == 0 &&
          I->hasName()) {
        const Function *oneFunc = 0;
        bool flag = usedInOneFunc(I, oneFunc);
        if (flag == false || (flag && !oneFunc)) {
          continue;
        }
        // Not in this Funciton's scope
        if (F.getName().str() != oneFunc->getName().str()) {
          continue;
        }
        // Theoretically, an eligible parallel body, e.g OpenCL Kernel, does not have static global
        // non-constant variables inside. Even it has, we won't update them for the reason
        // that the occurrence of race condition can cause undefined behavior.
        // However for emitted OpenCL C Kernel, this happens. For example, Kernel body initiates
        // an instance of class A, and there is a static variable used to initialize A's members in A's ctors.
        // Note that it is not the same as tile_static variables which we have implemented correctly.
        // Take the following two steps to work around,
        // (1) Declare those variables in private or local address space inside its containing Kernel.
        //      This is to make Kernel compilation successful. Note that for performance-sensitive
        //      cases, local space is suggested.
        // (2) Keep in mind that Kerne's behavior is still undefined if we try to utilize any
        //      values from those variables outside Kernel.
        Out << " /* FIXME: Static global variables inside. Make private instead */\n";
        Out << " __private ";
        printType(Out, I->getType()->getElementType(), false, GetValueName(I));
        printInitializer(I);
        Out << ";\n";
      }
    }
  }
}

#ifdef MXPA_CODEGEN
void CWriter::printMXPAKernelRegister() {
  collectKernelInfo();

  if (kernelList.empty()) {
    return;
  }

  std::vector<std::string>::iterator const it_end = kernelList.end();

  Out << "#ifdef REGISTER_KERNEL\n";
  // add prefix for launch_kernel() on myriad2
  Out << "#ifdef REGISTER_KERNEL_SEPARATED\n"
      "# define MK_KERNEL_NAME(name) MXPA_##name\n"
      "#else\n"
      "# define MK_KERNEL_NAME(name) name\n"
      "#endif\n\n";

  /* declare kernels if we did not generate kernel functions here */
  for (std::vector<std::string>::iterator it = kernelList.begin();
       it != it_end; ++it) {
    Out << "/* " << *it << " */\n";
    Out << "extern void MK_KERNEL_NAME(launch_kernel_" << bcHash << "_" << *it << ")(\n"
        << "  unsigned *group_id, unsigned *global_size,\n"
        << "  unsigned *local_size, unsigned *global_offset, unsigned work_dim, void const **arguments);\n\n";

    Out << "extern int  MK_KERNEL_NAME(get_upper_" << bcHash << "_" << *it << ")(\n"
        << "  unsigned arg_no, "
        << "  unsigned *group_id, unsigned *global_size,\n"
        << "  unsigned *local_size, unsigned *global_offset, unsigned work_dim, void const **arguments);\n\n";

    Out << "extern int  MK_KERNEL_NAME(get_lower_" << bcHash << "_" << *it << ")(\n"
        << "  unsigned arg_no, "
        << "  unsigned *group_id, unsigned *global_size,\n"
        << "  unsigned *local_size, unsigned *global_offset, unsigned work_dim, void const **arguments);\n\n";

    Out << "extern int  MK_KERNEL_NAME(get_argument_dir_" << bcHash << "_" << *it << ")(unsigned arg_no);\n\n";

    Out << "extern int  MK_KERNEL_NAME(get_argument_nr_" << bcHash << "_" << *it << ")(void);\n\n";
  }

  // print arguments for each kernel
  for (size_t i = 0; i < kernelList.size(); i++) {
    const std::string &krnlName = kernelList[i];

    Out << "static const opencl_kernel_arg_info arg_info_" << krnlName << "[] = {\n";
    unsigned arg_number = arg_addr_space_list[i].size();
    for (unsigned arg_N = 0; arg_N < arg_number; ++arg_N) {
      Out << "{"
          << (arg_addr_space_list[i]).at(arg_N) << ", "
          << (arg_access_qual_list[i]).at(arg_N) << ", "
          << "\"" << (arg_type_list[i]).at(arg_N) << "\", "
          << (arg_type_qual_list[i]).at(arg_N) << ", "
          << "\"" << (arg_name_list[i]).at(arg_N) << "\""
          << "}";
      if (arg_N + 1 == arg_number) {
        Out << "\n";
      } else {
        Out << ",\n";
      }
    }
    Out << "};\n\n";
  }

  // print kernels of current program
  Out << "static const opencl_kernel_descriptor prog_" << bcHash << "[] = {\n";

  for (size_t i = 0; i < kernelList.size(); i++) {
    const std::string &krnlName = kernelList[i];
    Out << "{\n"
        << "\"" << bcHash << "\",\n"
        << "\"" << krnlName << "\",\n"
        << "MK_KERNEL_NAME(launch_kernel_" << bcHash << "_" << krnlName << "),\n"
        << "{ get_upper_" << bcHash << "_" << krnlName << ", "
        << "MK_KERNEL_NAME(get_upper_" << bcHash << "_" << krnlName << ") },\n"
        << "{ get_lower_" << bcHash << "_" << krnlName << ", "
        << "MK_KERNEL_NAME(get_lower_" << bcHash << "_" << krnlName << ") },\n"
        << "{ get_argument_dir_" << bcHash << "_" << krnlName << ", "
        << "MK_KERNEL_NAME(get_argument_dir_" << bcHash << "_" << krnlName << ") },\n"
        << "{ get_argument_nr_" << bcHash << "_" << krnlName << ", "
        << "MK_KERNEL_NAME(get_argument_nr_" << bcHash << "_" << krnlName << ") },\n"
        << "{" << work_group_list[i].at(0) << "," << work_group_list[i].at(1) << "," <<  work_group_list[i].at(2) << "},\n"
        << "arg_info_" << krnlName << "}";

    if (i + 1 == kernelList.size()) {
      Out << "\n";
    } else {
      Out << ",\n";
    }
  }

  Out << "};\n\n";

  Out << "#undef MK_KERNEL_NAME\n";

  // register program by constructor
  Out << "static void __attribute__((constructor)) __mxpa_register_me_"
      << bcHash << "() {\n";

  Out << "  reg_krldes_func(" << kernelList.size() << ", (opencl_kernel_descriptor *)prog_" << bcHash << ");\n";

  Out << "}\n"
      << "#endif\n\n";
}

void CWriter::printMXPAWrapper(Function &F) {
  assert(isKernelFunction(&F));
  assert(!F.hasStructRetAttr());
  Out << "#if defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__))\n";
  Out << "#define MXPA_NOEXIT 1\n";
  Out << "#endif\n";

  Out << "static void launch_kernel_" << GetValueName(&F) << '(';
  Out << "unsigned *group_id, unsigned *global_size,\n"
      "  unsigned *local_size, unsigned *global_offset, unsigned work_dim, void const **arguments) {\n";

  // Prepare the call to kernel
  std::string tstr;
  raw_string_ostream FunctionInnards(tstr);

  // Loop over the arguments, printing them...
  FunctionType *FT = cast<FunctionType>(F.getFunctionType());
  const AttributeSet &PAL = F.getAttributes();

  // Print out the kernel name and first few MxPA arguments
  Out << "  " << GetValueName(&F) << "(group_id, global_size, local_size, global_offset, work_dim, ";

  // Loop over the arguments, printing them.
  FunctionType::param_iterator I = FT->param_begin(), E = FT->param_end();
  unsigned Idx = 0;
  for (; I != E; ++I) {
    Type *ArgTy = *I;

    Out << ",\n    (";
    printType(Out, ArgTy, PAL.hasAttribute(Idx, Attribute::SExt));
    Out << ")";
    if (ArgTy->getTypeID() == Type::PointerTyID || ArgTy->getTypeID() == Type::ArrayTyID) {
      Out << " ((uintptr_t)arguments[" << Idx << "])";
    } else {
      Out << " *((";
      printType(Out, ArgTy, PAL.hasAttribute(Idx, Attribute::SExt));
      Out << "*)arguments[" << Idx << "])";
    }

    ++Idx;
  }
  Out << ");\n";
  Out << "#ifndef MXPA_NOEXIT\n  exit(0);\n#endif\n";
  Out << "}\n";
}

namespace {
static Value *GetSCEVBase(ScalarEvolution *SE, const SCEV *scev);
static bool isInnerArgument(Value *V);
}

/// NeedHndlMXPAForSlctInst - Detect if the Value '*V' has been stored in
/// MemoryAcesses. It also decide whether the value should be processed
/// separately or processed as a common case.
bool CWriter::NeedHndlMXPAForSlctInst(Value *V) {
  Argument *arg = dyn_cast<Argument>(V);
  if (!arg) {
    return false;
  }

  std::map<Value *, std::set<const SCEV *> >::iterator
  I = MemoryAccesses.find(V);

  if (I != MemoryAccesses.end()) {
    return false;
  }
  return true;
}

/// hndlMXPAArgForSlctInst - Handle one operand of a select instruction
/// separately. It regards the operand as a part of a *virtual* load
/// instruction. Consider the case :
///
///     "%z.y = select i1 %cmp, float* %z, float* %y"
///
/// It can be regarded as two *virtual* load instructions like :
///
///     "%z.y = load float* %z, align 4" and "%z.y = load float* %y, align 4"
///
/// The two *virtual* load instructions are only used for MXPA QUERY.
void CWriter::hndlMXPAArgForSlctInst(Value *V) {
  if (!NeedHndlMXPAForSlctInst(V)) {
    return;
  }
  printMXPAArgForSlctInst(V);
}

/// printMXPAArgForSlctInst - Print the MXPA argument query information for
/// a operand of a select instruction.
void CWriter::printMXPAArgForSlctInst(Value *V) {
  Argument *arg = dyn_cast<Argument>(V);
  if (!arg) {
    return;
  }
  Out << "    case " << arg->getArgNo() << ": { //";
  writeOperand(V);
  Out << "\n";

  const SCEV *scev = SE->getSCEV(V);
  Value *v = GetSCEVBase(SE, scev);

  if (v) {
    if (MemoryWrites.find(scev) != MemoryWrites.end()) {
      Out << "         return 1;\n";
    } else {
      Out << "         return 0;\n";
    }
    Out << "         }\n";
  }
}

/// printMXPAArgCommCase - Print the MXPA argument information for *common*
/// case.
void CWriter::printMXPAArgCommCase(
  std::map<Value *, std::set<const SCEV *> >::iterator &it) {
  Value *ptr = it->first;
  if (Argument *arg = dyn_cast<Argument>(ptr)) {
    Out << "    case " << arg->getArgNo() << ": { //";
    writeOperand(ptr);
    Out << "\n";
    std::set<const SCEV *> scevs = it->second;
    int rtVal = 3;
    for (std::set<const SCEV *>::iterator sit = scevs.begin(), se =
           scevs.end(); sit != se; sit++) {
      const SCEV *s = *sit;
      if (MemoryWrites.find(s) != MemoryWrites.end()) {
        if (rtVal == 0) {
          rtVal = 3;
          break;
        }
        rtVal = 1;
      } else {
        if (rtVal == 1) {
          rtVal = 3;
          break;
        }
        rtVal = 0;
      }
    }
    Out << "         return " << rtVal << ";\n";
    Out << "         }\n";
  }
}

void CWriter::printMXPAArgQueries(Function &F) {
  assert(isKernelFunction(&F));
  assert(!F.hasStructRetAttr());
  //const AttributeSet &PAL = F.getAttributes();

  // Generate argument # query
  Out << "\nint get_argument_nr_" << bcHash << "_" << GetValueName(&F) << "(void) {\n  return ";
  Out << F.arg_size() << ";\n}\n";

  // Generate argument direction queries
  Out << "\nint get_argument_dir_" << bcHash << "_" << GetValueName(&F) << "(unsigned arg_no) {\n";
  Out << "  switch(arg_no) {\n";
  typedef std::map<Value *, std::set<const SCEV *> > Scevs;
  for (Scevs::iterator it = MemoryAccesses.begin(), e = MemoryAccesses.end();
       it != e; it++) {
    Value *ptr = it->first;
    if (isa<Constant>(ptr)) {
      continue;  // Constant may be included
    }
    if (SelectInst *slc = dyn_cast<SelectInst>(ptr)) {
      Value *trueV = slc->getTrueValue();
      hndlMXPAArgForSlctInst(trueV);
      Value *falseV = slc->getFalseValue();
      hndlMXPAArgForSlctInst(falseV);
      continue;
    }
    if (PHINode *phi = dyn_cast<PHINode>(ptr)) {
      int numValues = phi->getNumIncomingValues();
      for (int i = 0; i < numValues; ++i) {
        Value *v = phi->getIncomingValue(i);
        hndlMXPAArgForSlctInst(v);
      }
      continue;
    }
    if (isInnerArgument(ptr)) {
      continue;
    }
    printMXPAArgCommCase(it);
    Out << "\n";
  }
  Out << "  };\nreturn 2;\n}\n\n";
}

namespace {
std::string corners(unsigned g1) {
  std::string s = "#undef mxpa_get_global_id\n#define mxpa_get_global_id(x)";
  if (g1) {
    s += " (group_id[x]*local_size[x]+local_size[x]-1)";
  } else {
    s += " (group_id[x]*local_size[x])";
  }
  s += "\n";

  s += "#undef mxpa_get_group_id\n#define mxpa_get_group_id(x)";
  s += " (group_id[x])";
  s += "\n";

  s += "#undef mxpa_get_local_size\n#define mxpa_get_local_size(x)";
  s += " (local_size[x])";
  s += "\n";

  s += "#undef mxpa_get_local_id\n#define mxpa_get_local_id(x)";
  if (g1) {
    s += " (local_size[x]-1)";
  } else {
    s += " (0)";
  }
  s += "\n";
  return s;
}
}

/// hndlMXPABndsForSlctInst - Handle one operand of a select instruction
/// separately. It regards the operand as a part of a *virtual* load
/// instruction.
void CWriter::hndlMXPABndsForSlctInst(Value *V, bool Upper) {
  if (!NeedHndlMXPAForSlctInst(V)) {
    return;
  }
  printMXPABndsForSlctInst(V, Upper);
}

/// printMXPABndsForSlctInst - Print the MXPA bound information for a operand
/// of a select instruction.
void CWriter::printMXPABndsForSlctInst(Value *V, bool Upper) {
  Argument *arg = dyn_cast<Argument>(V);
  if (!arg) {
    return;
  }
  Out << "    case " << arg->getArgNo() << ": { //";
  writeOperand(V);
  Out << "\n";
  if (Upper) {
    Out << "         int b = 0;\n";
  } else {
    Out << "         int b = INT_MAX;\n";
  }

  const SCEV *scev = SE->getSCEV(V);
  Value *vBase = GetSCEVBase(SE, scev);
  if (vBase) {
    for (Value::user_iterator ui = V->user_begin(),
         ue = V->user_end(); ui != ue; ui++) {
      Value *uv = *ui;
      if (Instruction *I = dyn_cast<Instruction>(uv)) {
        if (I->mayWriteToMemory()) {
          MemoryWrites.insert(scev);
        }
      }
    }
  } else {
    errs() << "CWriter: Cannot locate the base of a memory access. Upper/lower "
           "bound analysis may be inaccurate\n";
  }

  for (unsigned i = 0; i < 2; i++) {
    Out << corners(i);
    if (Upper) {
      Out << "         b = mxpa_smax(b, ";
    } else {
      Out << "         b = mxpa_smin(b, ";
    }
    writeBounds(scev, Upper, V);
    Out << ");\n";
  }

  // Upper bounds are exclusive
  if (Upper) {
    int elesize = TD->getTypeStoreSize(arg->getType()->getPointerElementType());
    if (MemoryElementSize[V] != NULL) {
      if (SCEVConstant *s = dyn_cast<SCEVConstant>((SCEV *)MemoryElementSize[V])) {
        elesize = std::max((int)((ConstantInt *)(s->getValue()))->getZExtValue(), elesize);
      }
    }

    Out << "         b+= "
        << elesize
        << ";\n";
  };
  Out << "         return b;\n";
  Out << "         }\n";
}

/// printMXPABndsCommCase - Print the MXPA bound information for *common* case.
void CWriter::printMXPABndsCommCase(
  std::map<Value *, std::set<const SCEV *> >::iterator &it,
  bool Upper) {
  Value *ptr = it->first;
  if (Argument *arg = dyn_cast<Argument>(ptr)) {
    Out << "    case " << arg->getArgNo() << ": { //";
    writeOperand(ptr);
    Out << "\n";
    std::set<const SCEV *> scevs = it->second;
    std::set<const SCEV *>::iterator sit = scevs.begin();
    std::set<const SCEV *>::iterator se = scevs.end();
    char ret = 0;
    for (; sit != se; sit++) {
      ret = 0;
      CheckBounds(*sit, Upper, ret);
      if (ret & 1) {
        Out << "         return -1;\n";
        Out << "         }\n";
        return;
      } else if ((ret & 1) == 0 && (ret & 2) == 2) {
        Out << "    #ifndef INDIRECTION\n";
        Out << "         return -1;\n";
        Out << "    #else\n";
        break;
      }
    }

    if (Upper) {
      Out << "         int b = 0;\n";
    } else {
      Out << "         int b = INT_MAX;\n";
    }
    for (sit = scevs.begin(), se = scevs.end(); sit != se; sit++) {
      for (unsigned i = 0; i < 2; i++) {
        Out << corners(i);
        if (Upper) {
          Out << "         b = mxpa_smax(b, ";
        } else {
          Out << "         b = mxpa_smin(b, ";
        }
        writeBounds(*sit, Upper, ptr);
        Out << ");\n";
      }
    }
    // Upper bounds are exclusive
    if (Upper) {
      int elesize = TD->getTypeStoreSize(arg->getType()->getPointerElementType());
      if (SCEVConstant *s = dyn_cast<SCEVConstant>((SCEV *)MemoryElementSize[ptr])) {
        elesize = std::max((int)((ConstantInt *)(s->getValue()))->getZExtValue(), elesize);
      }

      Out << "         b+= "
          << elesize
          << ";\n";
    };
    Out << "         return b;\n";

    if ((ret & 1) == 0 && (ret & 2) == 2) {
      Out << "    #endif\n";
    }

    Out << "         }\n";
  }
}


// Textually synthesize kernel bound query functions
void CWriter::printMXPABounds(Function &F, bool Upper) {
  assert(isKernelFunction(&F));
  assert(!F.hasStructRetAttr());
  // Upper/lower bound routines
  const AttributeSet &PAL = F.getAttributes();

  // Print out the kernel name and first few MxPA arguments
  Out << "\nint get_" << (Upper ? "upper" : "lower") << "_" << bcHash << "_" << GetValueName(&F) << "(";
  Out << "unsigned arg_no, unsigned *group_id, unsigned *global_size, unsigned *local_size, unsigned *global_offset, unsigned work_dim, "
      "void const **arguments) {\n";

  // Loop over the arguments, printing them.
  Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end();
  unsigned Idx = 0;

  std::string ArgName;
  for (; I != E; ++I) {
    ArgName = GetValueName(I);
    Type *ArgTy = I->getType();

    Out << "  ";
    printType(Out, ArgTy, PAL.hasAttribute(Idx, Attribute::SExt), ArgName);
    Out << "= (";
    printType(Out, ArgTy, PAL.hasAttribute(Idx, Attribute::SExt));
    Out << ")";
    if (ArgTy->getTypeID() == Type::PointerTyID || ArgTy->getTypeID() == Type::ArrayTyID) {
      Out << " ((uintptr_t)arguments[" << Idx << "]);\n";
    } else {
      Out << " *((";
      printType(Out, ArgTy, PAL.hasAttribute(Idx, Attribute::SExt));
      Out << "*)arguments[" << Idx << "]);\n";
    }
    ++Idx;
  }

  Out << "  switch(arg_no) {\n";
  typedef std::map<Value *, std::set<const SCEV *> > Scevs;

  for (Scevs::iterator it = MemoryAccesses.begin(), e = MemoryAccesses.end();
       it != e; it++) {
    Value *ptr = it->first;
    if (isa<Constant>(ptr)) {
      continue;  // Constant may be included
    }
    if (SelectInst *slc = dyn_cast<SelectInst>(ptr)) {
      Value *trueV = slc->getTrueValue();
      hndlMXPABndsForSlctInst(trueV, Upper);

      Value *falseV = slc->getFalseValue();
      hndlMXPABndsForSlctInst(falseV, Upper);
      continue;
    }
    if (PHINode *phi = dyn_cast<PHINode>(ptr)) {
      int numValues = phi->getNumIncomingValues();
      for (int i = 0; i < numValues; ++i) {
        Value *v = phi->getIncomingValue(i);
        hndlMXPABndsForSlctInst(v, Upper);
      }
      continue;
    }
    if (isInnerArgument(ptr)) {
      continue;
    }
    printMXPABndsCommCase(it, Upper);
    Out << "\n";
  }
  Out << "  };\nreturn 0;\n}";
}

/// isUselessFunction - Figure out the functions that needn't to generate code.
/// Two cases are considered: (1) The functions are inlined in the kernel
/// functions, and (2) The functions are called in kernel functions but are not
/// lineable.
bool CWriter::isUselessFunction(Function &F) {
  if (isKernelFunction(&F)) {
    return false;
  }

  if (isLaunchKernel(F)) {
    return false;
  }

  if (NonInlinedCallees.find(&F) != NonInlinedCallees.end()) {
    return true;
  }

  if (InlinedCallees.find(&F) != InlinedCallees.end()) {
    return true;
  }

  return false;
}

bool CWriter::isLaunchKernel(const Function &F) {
  return LaunchKernels.find(&F) != LaunchKernels.end();
}

#endif // MXPA_CODEGEN

void CWriter::printFunction(Function &F) {
  /// isStructReturn - Should this function actually return a struct by-value?
  bool isStructReturn = F.hasStructRetAttr();

  printFunctionSignature(&F, false);
  Out << " {\n";

  // If this is a struct return function, handle the result with magic.
  if (isStructReturn) {
    Type *StructTy =
      cast<PointerType>(F.arg_begin()->getType())->getElementType();
    Out << "  ";
    printType(Out, StructTy, false, "StructReturn");
    Out << ";  /* Struct return temporary */\n";

    Out << "  ";
    printType(Out, F.arg_begin()->getType(), false,
              GetValueName(F.arg_begin()));
    Out << " = &StructReturn;\n";
  }

  printFunctionPrivateVaraibles(F);

  bool PrintedVar = false;

  // print local variable information for the function
  std::set<std::string> PrintedLocal;
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    if (const AllocaInst *AI = isDirectAlloca(&*I)) {
      Out << "  ";
      printType(Out, AI->getAllocatedType(), false, GetValueName(AI));
      Out << ";    /* Address-exposed local */\n";
      PrintedVar = true;
    } else if (I->getType() != Type::getVoidTy(F.getContext()) &&
               !isInlinableInst(*I)) {
      Out << "  ";
      printType(Out, I->getType(), false, GetValueName(&*I));
      Out << ";\n";

      if (isa<PHINode>(*I)) {  // Print out PHI node temporaries as well...
        Out << "  ";
        printType(Out, I->getType(), false,
                  GetValueName(&*I) + "__PHI_TEMPORARY");
        Out << ";\n";
      }
      PrintedVar = true;
    }
    {
      Type *LocalTy = NULL;
      Value *LocalValue = NULL;
      FindLocalName(&*I, LocalValue, LocalTy);
      if (LocalTy) {
        std::string LocalName = GetValueName(LocalValue);
        if (PrintedLocal.find(LocalName) != PrintedLocal.end()) {
          continue;
        } else {
          PrintedLocal.insert(LocalName);
        }
        Out << "  __local ";
        printType(Out, LocalTy, false, LocalName);
        Out << ";\n";
        PrintedVar = true;
      }
    }
    // We need a temporary for the BitCast to use so it can pluck a value out
    // of a union to do the BitCast. This is separate from the need for a
    // variable to hold the result of the BitCast.
    if (isFPIntBitCast(*I)) {
      Out << "  llvmBitCastUnion " << GetValueName(&*I)
          << "__BITCAST_TEMPORARY;\n";
      PrintedVar = true;
    }
  }

  if (PrintedVar) {
    Out << '\n';
  }

  if (F.hasExternalLinkage() && F.getName() == "main") {
    Out << "  CODE_FOR_MAIN();\n";
  }

  // print the basic blocks
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (Loop *L = LI->getLoopFor(BB)) {
      if (L->getHeader() == BB && L->getParentLoop() == 0) {
        printLoop(L);
      }
    } else {
      printBasicBlock(BB);
    }
  }

  Out << "}\n\n";
}

void CWriter::printLoop(Loop *L) {
  Out << "  do {     /* Syntactic loop '" << L->getHeader()->getName()
      << "' to make GCC happy */\n";
  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    Loop *BBLoop = LI->getLoopFor(BB);
    if (BBLoop == L) {
      printBasicBlock(BB);
    } else if (BB == BBLoop->getHeader() && BBLoop->getParentLoop() == L) {
      printLoop(BBLoop);
    }
  }
  Out << "  } while (1); /* end of syntactic loop '"
      << L->getHeader()->getName() << "' */\n";
}

void CWriter::printBasicBlock(BasicBlock *BB) {

  // Don't print the label for the basic block if there are no uses, or if
  // the only terminator use is the predecessor basic block's terminator.
  // We have to scan the use list because PHI nodes use basic blocks too but
  // do not require a label to be generated.
  //
  bool NeedsLabel = false;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    if (isGotoCodeNecessary(*PI, BB)) {
      NeedsLabel = true;
      break;
    }

  if (NeedsLabel) {
    Out << GetValueName(BB) << ":\n";
  }

  // Output all of the instructions in the basic block...
  for (BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E;
       ++II) {
    if (!isInlinableInst(*II) && !isDirectAlloca(II)) {
      if (II->getType() != Type::getVoidTy(BB->getContext()) &&
          !isInlineAsm(*II)) {
        outputLValue(II);
      } else {
        Out << "  ";
      }
      writeInstComputationInline(*II);
      Out << ";\n";
    }
  }

  // Don't emit prefix or suffix for the terminator.
  visit(*BB->getTerminator());
}


// Specific Instruction type classes... note that all of the casts are
// necessary because we use the instruction classes as opaque types...
//
void CWriter::visitReturnInst(ReturnInst &I) {
  // If this is a struct return function, return the temporary struct.
  bool isStructReturn = I.getParent()->getParent()->hasStructRetAttr();

  if (isStructReturn) {
    Out << "  return StructReturn;\n";
    return;
  }

  // Don't output a void return if this is the last basic block in the function
  if (I.getNumOperands() == 0 &&
      &*--I.getParent()->getParent()->end() == I.getParent() &&
      !I.getParent()->size() == 1) {
    return;
  }

  Out << "  return";
  if (I.getNumOperands()) {
    Out << ' ';
    writeOperand(I.getOperand(0));
  }
  Out << ";\n";
}

void CWriter::visitSwitchInst(SwitchInst &SI) {

  Value *Cond = SI.getCondition();

  Out << "  switch (";
  writeOperand(Cond);
  Out << ") {\n  default:\n";
  printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
  printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
  Out << ";\n";

  // Skip the first item since that's the default case.
  for (SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end(); i != e; ++i) {
    ConstantInt *CaseVal = i.getCaseValue();
    BasicBlock *Succ = i.getCaseSuccessor();
    Out << "  case ";
    writeOperand(CaseVal);
    Out << ":\n";
    printPHICopiesForSuccessor(SI.getParent(), Succ, 2);
    printBranchToBlock(SI.getParent(), Succ, 2);
    if (Function::iterator(Succ) ==
        std::next(Function::iterator(SI.getParent()))) {
      Out << "    break;\n";
    }
  }

  Out << "  }\n";
}

void CWriter::visitIndirectBrInst(IndirectBrInst &IBI) {
  Out << "  goto *(void*)(";
  writeOperand(IBI.getOperand(0));
  Out << ");\n";
}

void CWriter::visitUnreachableInst(UnreachableInst &I) {
  Out << "  /*UNREACHABLE*/;\n";
}

bool CWriter::isGotoCodeNecessary(BasicBlock *From, BasicBlock *To) {
  /// FIXME: This should be reenabled, but loop reordering safe!!
  return true;

  if (std::next(Function::iterator(From)) != Function::iterator(To)) {
    return true;  // Not the direct successor, we need a goto.
  }

  //isa<SwitchInst>(From->getTerminator())

  if (LI->getLoopFor(From) != LI->getLoopFor(To)) {
    return true;
  }
  return false;
}

void CWriter::printPHICopiesForSuccessor(BasicBlock *CurBlock,
    BasicBlock *Successor,
    unsigned Indent) {
  for (BasicBlock::iterator I = Successor->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    // Now we have to do the printing.
    Value *IV = PN->getIncomingValueForBlock(CurBlock);
    if (!isa<UndefValue>(IV)) {
      Out << std::string(Indent, ' ');
      Out << "  " << GetValueName(I) << "__PHI_TEMPORARY = ";
      writeOperand(IV);
      Out << ";   /* for PHI node */\n";
    }
  }
}

void CWriter::printBranchToBlock(BasicBlock *CurBB, BasicBlock *Succ,
                                 unsigned Indent) {
  if (isGotoCodeNecessary(CurBB, Succ)) {
    Out << std::string(Indent, ' ') << "  goto ";
    writeOperand(Succ);
    Out << ";\n";
  }
}

// Branch instruction printing - Avoid printing out a branch to a basic block
// that immediately succeeds the current one.
//
void CWriter::visitBranchInst(BranchInst &I) {

  if (I.isConditional()) {
    if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(0))) {
      Out << "  if (";
      writeOperand(I.getCondition());
      Out << ") {\n";

      printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(0), 2);
      printBranchToBlock(I.getParent(), I.getSuccessor(0), 2);

      if (isGotoCodeNecessary(I.getParent(), I.getSuccessor(1))) {
        Out << "  } else {\n";
        printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(1), 2);
        printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
      }
    } else {
      // First goto not necessary, assume second one is...
      Out << "  if (!";
      writeOperand(I.getCondition());
      Out << ") {\n";

      printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(1), 2);
      printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
    }

    Out << "  }\n";
  } else {
    printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(0), 0);
    printBranchToBlock(I.getParent(), I.getSuccessor(0), 0);
  }
  Out << "\n";
}

// PHI nodes get copied into temporary values at the end of predecessor basic
// blocks.  We now need to copy these temporary values into the REAL value for
// the PHI.
void CWriter::visitPHINode(PHINode &I) {
  writeOperand(&I);
  Out << "__PHI_TEMPORARY";
}


void CWriter::visitBinaryOperator(Instruction &I) {
  // binary instructions, shift instructions, setCond instructions.
  assert(!I.getType()->isPointerTy());

  // We must cast the results of binary operations which might be promoted.
  bool needsCast = false;
  if ((I.getType() == Type::getInt8Ty(I.getContext())) ||
      (I.getType() == Type::getInt16Ty(I.getContext()))
      || (I.getType() == Type::getFloatTy(I.getContext()))) {
    needsCast = true;
    Out << "((";
    printType(Out, I.getType(), false);
    Out << ")(";
  }

  // If this is a negation operation, print it out as such.  For FP, we don't
  // want to print "-0.0 - X".
  if (BinaryOperator::isNeg(&I)) {
    Out << "-(";
    writeOperand(BinaryOperator::getNegArgument(cast<BinaryOperator>(&I)));
    Out << ")";
  } else if (BinaryOperator::isFNeg(&I)) {
    Out << "-(";
    writeOperand(BinaryOperator::getFNegArgument(cast<BinaryOperator>(&I)));
    Out << ")";
  } else if (I.getOpcode() == Instruction::FRem) {
    // Output a call to fmod/fmodf instead of emitting a%b
    if (I.getType() == Type::getFloatTy(I.getContext())) {
      Out << "fmodf(";
    } else if (I.getType() == Type::getDoubleTy(I.getContext())) {
      Out << "fmod(";
    } else { // all 3 flavors of long double
      Out << "fmodl(";
    }
    writeOperand(I.getOperand(0));
    Out << ", ";
    writeOperand(I.getOperand(1));
    Out << ")";
  } else {

    // Write out the cast of the instruction's value back to the proper type
    // if necessary.
    bool NeedsClosingParens = writeInstructionCast(I);

    // Certain instructions require the operand to be forced to a specific type
    // so we use writeOperandWithCast here instead of writeOperand. Similarly
    // below for operand 1
    writeOperandWithCast(I.getOperand(0), I.getOpcode());

    switch (I.getOpcode()) {
      case Instruction::Add:
      case Instruction::FAdd:
        Out << " + ";
        break;
      case Instruction::Sub:
      case Instruction::FSub:
        Out << " - ";
        break;
      case Instruction::Mul:
      case Instruction::FMul:
        Out << " * ";
        break;
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::FRem:
        Out << " % ";
        break;
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::FDiv:
        Out << " / ";
        break;
      case Instruction::And:
        Out << " & ";
        break;
      case Instruction::Or:
        Out << " | ";
        break;
      case Instruction::Xor:
        Out << " ^ ";
        break;
      case Instruction::Shl :
        Out << " << ";
        break;
      case Instruction::LShr:
      case Instruction::AShr:
        Out << " >> ";
        break;
      default:
#ifndef NDEBUG
        errs() << "Invalid operator type!" << I;
#endif
        llvm_unreachable(0);
    }

    writeOperandWithCast(I.getOperand(1), I.getOpcode());
    if (NeedsClosingParens) {
      Out << "))";
    }
  }

  if (needsCast) {
    Out << "))";
  }
}

void CWriter::visitICmpInst(ICmpInst &I) {
  // We must cast the results of icmp which might be promoted.
  bool needsCast = false;

  // Write out the cast of the instruction's value back to the proper type
  // if necessary.
  bool NeedsClosingParens = writeInstructionCast(I);

  // Certain icmp predicate require the operand to be forced to a specific type
  // so we use writeOperandWithCast here instead of writeOperand. Similarly
  // below for operand 1
  writeOperandWithCast(I.getOperand(0), I);

  switch (I.getPredicate()) {
    case ICmpInst::ICMP_EQ:
      Out << " == ";
      break;
    case ICmpInst::ICMP_NE:
      Out << " != ";
      break;
    case ICmpInst::ICMP_ULE:
    case ICmpInst::ICMP_SLE:
      Out << " <= ";
      break;
    case ICmpInst::ICMP_UGE:
    case ICmpInst::ICMP_SGE:
      Out << " >= ";
      break;
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_SLT:
      Out << " < ";
      break;
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_SGT:
      Out << " > ";
      break;
    default:
#ifndef NDEBUG
      errs() << "Invalid icmp predicate!" << I;
#endif
      llvm_unreachable(0);
  }

  writeOperandWithCast(I.getOperand(1), I);
  if (NeedsClosingParens) {
    Out << "))";
  }

  if (needsCast) {
    Out << "))";
  }
}

void CWriter::visitFCmpInst(FCmpInst &I) {
  if (I.getPredicate() == FCmpInst::FCMP_FALSE) {
    Out << "0";
    return;
  }
  if (I.getPredicate() == FCmpInst::FCMP_TRUE) {
    Out << "1";
    return;
  }

  const char *op = 0;
  switch (I.getPredicate()) {
    default:
      llvm_unreachable("Illegal FCmp predicate");
    case FCmpInst::FCMP_ORD:
      op = "ord";
      break;
    case FCmpInst::FCMP_UNO:
      op = "uno";
      break;
    case FCmpInst::FCMP_UEQ:
      op = "ueq";
      break;
    case FCmpInst::FCMP_UNE:
      op = "une";
      break;
    case FCmpInst::FCMP_ULT:
      op = "ult";
      break;
    case FCmpInst::FCMP_ULE:
      op = "ule";
      break;
    case FCmpInst::FCMP_UGT:
      op = "ugt";
      break;
    case FCmpInst::FCMP_UGE:
      op = "uge";
      break;
    case FCmpInst::FCMP_OEQ:
      op = "oeq";
      break;
    case FCmpInst::FCMP_ONE:
      op = "one";
      break;
    case FCmpInst::FCMP_OLT:
      op = "olt";
      break;
    case FCmpInst::FCMP_OLE:
      op = "ole";
      break;
    case FCmpInst::FCMP_OGT:
      op = "ogt";
      break;
    case FCmpInst::FCMP_OGE:
      op = "oge";
      break;
  }

  Out << "llvm_fcmp_" << op << "(";
  // Write the first operand
  writeOperand(I.getOperand(0));
  Out << ", ";
  // Write the second operand
  writeOperand(I.getOperand(1));
  Out << ")";
}

static const char *getFloatBitCastField(Type *Ty) {
  switch (Ty->getTypeID()) {
    default:
      llvm_unreachable("Invalid Type");
    case Type::FloatTyID:
      return "Float";
    case Type::DoubleTyID:
      return "Double";
    case Type::IntegerTyID: {
        unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
        if (NumBits <= 32) {
          return "Int32";
        } else {
          return "Int64";
        }
      }
  }
}

void CWriter::visitCastInst(CastInst &I) {
  Type *DstTy = I.getType();
  Type *SrcTy = I.getOperand(0)->getType();
  if (isFPIntBitCast(I)) {
    Out << '(';
    // These int<->float and long<->double casts need to be handled specially
    Out << GetValueName(&I) << "__BITCAST_TEMPORARY."
        << getFloatBitCastField(I.getOperand(0)->getType()) << " = ";
    writeOperand(I.getOperand(0));
    Out << ", " << GetValueName(&I) << "__BITCAST_TEMPORARY."
        << getFloatBitCastField(I.getType());
    Out << ')';
    return;
  }
  if (SrcTy->getTypeID() == Type::IntegerTyID) {
    unsigned NumBits = cast<IntegerType>(SrcTy)->getBitWidth();
    if (NumBits == 128) {
      Out << '*';
    }
  }

  bool ShouldCast = true;
  if (isa<PointerType>(DstTy)) {
    PointerType *PTy = dyn_cast<PointerType>(DstTy);
    if (PTy && (PTy->getAddressSpace() != 0)) {
      ShouldCast = false;
    }
  }
  if (ShouldCast) {
    Out << '(';
    printCast(I.getOpcode(), SrcTy, DstTy);
  }

  // Make a sext from i1 work by subtracting the i1 from 0 (an int).
  if (SrcTy == Type::getInt1Ty(I.getContext()) &&
      I.getOpcode() == Instruction::SExt) {
    Out << "0-";
  }

  writeOperand(I.getOperand(0));

  if (DstTy == Type::getInt1Ty(I.getContext()) &&
      (I.getOpcode() == Instruction::Trunc ||
       I.getOpcode() == Instruction::FPToUI ||
       I.getOpcode() == Instruction::FPToSI ||
       I.getOpcode() == Instruction::PtrToInt)) {
    // Make sure we really get a trunc to bool by anding the operand with 1
    Out << "&1u";
  }
  if (ShouldCast) {
    Out << ')';
  }
}

void CWriter::visitSelectInst(SelectInst &I) {
  Out << "((";
  writeOperand(I.getCondition());
  Out << ") ? (";
  writeOperand(I.getTrueValue());
  Out << ") : (";
  writeOperand(I.getFalseValue());
  Out << "))";
}

// Returns the macro name or value of the max or min of an integer type
// (as defined in limits.h).
static void printLimitValue(IntegerType &Ty, bool isSigned, bool isMax,
                            raw_ostream &Out) {
  const char *type;
  const char *sprefix = "";

  unsigned NumBits = Ty.getBitWidth();
  if (NumBits <= 8) {
    type = "CHAR";
    sprefix = "S";
  } else if (NumBits <= 16) {
    type = "SHRT";
  } else if (NumBits <= 32) {
    type = "INT";
  } else if (NumBits <= 64) {
    type = "LLONG";
  } else {
    llvm_unreachable("Bit widths > 64 not implemented yet");
  }

  if (isSigned) {
    Out << sprefix << type << (isMax ? "_MAX" : "_MIN");
  } else {
    Out << "U" << type << (isMax ? "_MAX" : "0");
  }
}

#ifndef NDEBUG
static bool isSupportedIntegerSize(IntegerType &T) {
  return T.getBitWidth() == 8 || T.getBitWidth() == 16 ||
         T.getBitWidth() == 32 || T.getBitWidth() == 64;
}
#endif

void CWriter::printIntrinsicDefinition(const Function &F, raw_ostream &Out) {
  FunctionType *funT = F.getFunctionType();
  Type *retT = F.getReturnType();
  IntegerType *elemT = cast<IntegerType>(funT->getParamType(1));

  assert(isSupportedIntegerSize(*elemT) &&
         "CBackend does not support arbitrary size integers.");
  assert(cast<StructType>(retT)->getElementType(0) == elemT &&
         elemT == funT->getParamType(0) && funT->getNumParams() == 2);

  switch (F.getIntrinsicID()) {
    default:
      llvm_unreachable("Unsupported Intrinsic.");
    case Intrinsic::uadd_with_overflow:
      // static inline Rty uadd_ixx(unsigned ixx a, unsigned ixx b) {
      //   Rty r;
      //   r.field0 = a + b;
      //   r.field1 = (r.field0 < a);
      //   return r;
      // }
      Out << "static inline ";
      printType(Out, retT);
      Out << GetValueName(&F);
      Out << "(";
      printSimpleType(Out, elemT, false);
      Out << "a,";
      printSimpleType(Out, elemT, false);
      Out << "b) {\n  ";
      printType(Out, retT);
      Out << "r;\n";
      Out << "  r.field0 = a + b;\n";
      Out << "  r.field1 = (r.field0 < a);\n";
      Out << "  return r;\n}\n";
      break;

    case Intrinsic::sadd_with_overflow:
      // static inline Rty sadd_ixx(ixx a, ixx b) {
      //   Rty r;
      //   r.field1 = (b > 0 && a > XX_MAX - b) ||
      //              (b < 0 && a < XX_MIN - b);
      //   r.field0 = r.field1 ? 0 : a + b;
      //   return r;
      // }
      Out << "static ";
      printType(Out, retT);
      Out << GetValueName(&F);
      Out << "(";
      printSimpleType(Out, elemT, true);
      Out << "a,";
      printSimpleType(Out, elemT, true);
      Out << "b) {\n  ";
      printType(Out, retT);
      Out << "r;\n";
      Out << "  r.field1 = (b > 0 && a > ";
      printLimitValue(*elemT, true, true, Out);
      Out << " - b) || (b < 0 && a < ";
      printLimitValue(*elemT, true, false, Out);
      Out << " - b);\n";
      Out << "  r.field0 = r.field1 ? 0 : a + b;\n";
      Out << "  return r;\n}\n";
      break;
  }
}

void CWriter::lowerIntrinsics(Function &F) {
  // This is used to keep track of intrinsics that get generated to a lowered
  // function. We must generate the prototypes before the function body which
  // will only be expanded on first use (by the loop below).
  std::vector<Function *> prototypesToGen;

  // Examine all the instructions in this function to find the intrinsics that
  // need to be lowered.
  for (Function::iterator BB = F.begin(), EE = F.end(); BB != EE; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;)
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
            case Intrinsic::not_intrinsic:
            case Intrinsic::vastart:
            case Intrinsic::vacopy:
            case Intrinsic::vaend:
            case Intrinsic::returnaddress:
            case Intrinsic::frameaddress:
            case Intrinsic::setjmp:
            case Intrinsic::fmuladd:
            case Intrinsic::longjmp:
            case Intrinsic::prefetch:
            case Intrinsic::powi:
            case Intrinsic::x86_sse_cmp_ss:
            case Intrinsic::x86_sse_cmp_ps:
            case Intrinsic::x86_sse2_cmp_sd:
            case Intrinsic::x86_sse2_cmp_pd:
            case Intrinsic::ppc_altivec_lvsl:
            case Intrinsic::uadd_with_overflow:
            case Intrinsic::sadd_with_overflow:
              // We directly implement these intrinsics
              break;
            default:
              // If this is an intrinsic that directly corresponds to a GCC
              // builtin, we handle it.
              const char *BuiltinName = "";
              // If we handle it, don't lower it.
              if (BuiltinName[0]) {
                break;
              }

              // All other intrinsic calls we must lower.
              Instruction *Before = 0;
              if (CI != &BB->front()) {
                Before = std::prev(BasicBlock::iterator(CI));
              }

              IL->LowerIntrinsicCall(CI);
              if (Before) {        // Move iterator to instruction after call
                I = Before;
                ++I;
              } else {
                I = BB->begin();
              }
              // If the intrinsic got lowered to another call, and that call has
              // a definition then we need to make sure its prototype is emitted
              // before any calls to it.
              if (CallInst *Call = dyn_cast<CallInst>(I))
                if (Function *NewF = Call->getCalledFunction())
                  if (!NewF->isDeclaration()) {
                    prototypesToGen.push_back(NewF);
                  }

              break;
          }

  // We may have collected some prototypes to emit in the loop above.
  // Emit them now, before the function that uses them is emitted. But,
  // be careful not to emit them twice.
  std::vector<Function *>::iterator I = prototypesToGen.begin();
  std::vector<Function *>::iterator E = prototypesToGen.end();
  for (; I != E; ++I) {
    if (intrinsicPrototypesAlreadyGenerated.insert(*I).second) {
      Out << '\n';
      printFunctionSignature(*I, true);
      Out << ";\n";
    }
  }
}

void CWriter::visitCallInst(CallInst &I) {
  if (isa<InlineAsm>(I.getCalledValue())) {
    return visitInlineAsm(I);
  }

  bool WroteCallee = false;

  // Handle intrinsic function calls first...
  if (Function *F = I.getCalledFunction())
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID())
      if (visitBuiltinCall(I, ID, WroteCallee)) {
        return;
      }

  Value *Callee = I.getCalledValue();

  PointerType  *PTy   = cast<PointerType>(Callee->getType());
  FunctionType *FTy   = cast<FunctionType>(PTy->getElementType());

  // If this is a call to a struct-return function, assign to the first
  // parameter instead of passing it to the call.
  const AttributeSet &PAL = I.getAttributes();
  bool hasByVal = I.hasByValArgument();
  bool isStructRet = I.hasStructRetAttr();
  if (isStructRet) {
    writeOperandDeref(I.getArgOperand(0));
    Out << " = ";
  }

  if (I.isTailCall()) {
    Out << " /*tail*/ ";
  }

  if (!WroteCallee) {
    // If this is an indirect call to a struct return function, we need to cast
    // the pointer. Ditto for indirect calls with byval arguments.
    bool NeedsCast = (hasByVal || isStructRet) && !isa<Function>(Callee);

    // GCC is a real PITA.  It does not permit codegening casts of functions to
    // function pointers if they are in a call (it generates a trap instruction
    // instead!).  We work around this by inserting a cast to void* in between
    // the function and the function pointer cast.  Unfortunately, we can't just
    // form the constant expression here, because the folder will immediately
    // nuke it.
    //
    // Note finally, that this is completely unsafe.  ANSI C does not guarantee
    // that void* and function pointers have the same size. :( To deal with this
    // in the common case, we handle casts where the number of arguments passed
    // match exactly.
    //
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Callee))
      if (CE->isCast())
        if (Function *RF = dyn_cast<Function>(CE->getOperand(0))) {
          if (RF->getName() != "memcpy") { // skip casting for memcpy function
            NeedsCast = true;
          }
          Callee = RF;
        }

    if (NeedsCast) {
      // Ok, just cast the pointer type.
      Out << "((";
      if (isStructRet)
        printStructReturnPointerFunctionType(Out, PAL,
                                             cast<PointerType>(I.getCalledValue()->getType()));
      else if (hasByVal) {
        printType(Out, I.getCalledValue()->getType(), false, "", true, PAL);
      } else {
        printType(Out, I.getCalledValue()->getType());
      }
      Out << ")(void*)";
    }
    writeOperand(Callee);
    if (NeedsCast) {
      Out << ')';
    }
  }

  Out << '(';

  bool PrintedArg = false;
  if (FTy->isVarArg() && !FTy->getNumParams()) {
    Out << "0 /*dummy arg*/";
    PrintedArg = true;
  }

  unsigned NumDeclaredParams = FTy->getNumParams();
  CallSite CS(&I);
  CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
  unsigned ArgNo = 0;
  if (isStructRet) {   // Skip struct return argument.
    ++AI;
    ++ArgNo;
  }

  Function *F = I.getCalledFunction();
  if (F && F->getName() == "barrier") {
    // Special processing for OpenCL barrier() function:
    // Numerical values of CLK_LOCAL_MEM_FENCE and CLK_GLOBAL_MEM_FENCE can't be
    // determined before clBuildProgram() so we have to convert from integers
    // back to strings at this point.
    assert(CS.arg_size() == 1 && "Incorrect number of arguments to barrier()!");
    assert((*AI)->getType()->isIntegerTy() && "Illegal argument passed to barrier()!");
    ConstantInt *arg_value = dyn_cast_or_null<ConstantInt>(AI->get());
    assert(arg_value && "Can't extract integer value!");
    APInt v = arg_value->getValue();
    if (v == 1) {
      Out << "CLK_LOCAL_MEM_FENCE";
    } else if (v == 2) {
      Out << "CLK_GLOBAL_MEM_FENCE";
    } else if (v == 3) {
      Out << "CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE";
    } else {
      assert(0 && "Illegal argument passed to barrier()!");
    }
  } else {
    for (; AI != AE; ++AI, ++ArgNo) {
      if (PrintedArg) {
        Out << ", ";
      }
      if (ArgNo < NumDeclaredParams &&
          (*AI)->getType() != FTy->getParamType(ArgNo)) {
        Out << '(';
        printType(Out, FTy->getParamType(ArgNo),
                  /*isSigned=*/PAL.hasAttribute(ArgNo + 1, Attribute::SExt));
        Out << ')';
      }
      // Check if the argument is expected to be passed by value.
      if (I.paramHasAttr(ArgNo + 1, Attribute::ByVal)) {
        writeOperandDeref(*AI);
      } else {
        writeOperand(*AI);
      }
      PrintedArg = true;
    }
  }
  Out << ')';
}

/// visitBuiltinCall - Handle the call to the specified builtin.  Returns true
/// if the entire call is handled, return false if it wasn't handled, and
/// optionally set 'WroteCallee' if the callee has already been printed out.
bool CWriter::visitBuiltinCall(CallInst &I, Intrinsic::ID ID,
                               bool &WroteCallee) {
  switch (ID) {
    default: {
        // If this is an intrinsic that directly corresponds to a GCC
        // builtin, we emit it here.
        const char *BuiltinName = "";
        assert(BuiltinName[0] && "Unknown LLVM intrinsic!");

        Out << BuiltinName;
        WroteCallee = true;
        return false;
      }
    case Intrinsic::vastart:
      Out << "0; ";

      Out << "va_start(*(va_list*)";
      writeOperand(I.getArgOperand(0));
      Out << ", ";
      // Output the last argument to the enclosing function.
      if (I.getParent()->getParent()->arg_empty()) {
        Out << "vararg_dummy_arg";
      } else {
        writeOperand(--I.getParent()->getParent()->arg_end());
      }
      Out << ')';
      return true;
    case Intrinsic::vaend:
      if (!isa<ConstantPointerNull>(I.getArgOperand(0))) {
        Out << "0; va_end(*(va_list*)";
        writeOperand(I.getArgOperand(0));
        Out << ')';
      } else {
        Out << "va_end(*(va_list*)0)";
      }
      return true;
    case Intrinsic::fmuladd:
      Out << "(";
      writeOperand(I.getArgOperand(0));
      Out << " * ";
      writeOperand(I.getArgOperand(1));
      Out << " + ";
      writeOperand(I.getArgOperand(2));
      Out << ')';
      return true;
    case Intrinsic::vacopy:
      Out << "0; ";
      Out << "va_copy(*(va_list*)";
      writeOperand(I.getArgOperand(0));
      Out << ", *(va_list*)";
      writeOperand(I.getArgOperand(1));
      Out << ')';
      return true;
    case Intrinsic::returnaddress:
      Out << "__builtin_return_address(";
      writeOperand(I.getArgOperand(0));
      Out << ')';
      return true;
    case Intrinsic::frameaddress:
      Out << "__builtin_frame_address(";
      writeOperand(I.getArgOperand(0));
      Out << ')';
      return true;
    case Intrinsic::powi:
      Out << "__builtin_powi(";
      writeOperand(I.getArgOperand(0));
      Out << ", ";
      writeOperand(I.getArgOperand(1));
      Out << ')';
      return true;
    case Intrinsic::setjmp:
      Out << "setjmp(*(jmp_buf*)";
      writeOperand(I.getArgOperand(0));
      Out << ')';
      return true;
    case Intrinsic::longjmp:
      Out << "longjmp(*(jmp_buf*)";
      writeOperand(I.getArgOperand(0));
      Out << ", ";
      writeOperand(I.getArgOperand(1));
      Out << ')';
      return true;
    case Intrinsic::prefetch:
      Out << "LLVM_PREFETCH((const void *)";
      writeOperand(I.getArgOperand(0));
      Out << ", ";
      writeOperand(I.getArgOperand(1));
      Out << ", ";
      writeOperand(I.getArgOperand(2));
      Out << ")";
      return true;
    case Intrinsic::stacksave:
      // Emit this as: Val = 0; *((void**)&Val) = __builtin_stack_save()
      // to work around GCC bugs (see PR1809).
      Out << "0; *((void**)&" << GetValueName(&I)
          << ") = __builtin_stack_save()";
      return true;
    case Intrinsic::x86_sse_cmp_ss:
    case Intrinsic::x86_sse_cmp_ps:
    case Intrinsic::x86_sse2_cmp_sd:
    case Intrinsic::x86_sse2_cmp_pd:
      Out << '(';
      printType(Out, I.getType());
      Out << ')';
      // Multiple GCC builtins multiplex onto this intrinsic.
      switch (cast<ConstantInt>(I.getArgOperand(2))->getZExtValue()) {
        default:
          llvm_unreachable("Invalid llvm.x86.sse.cmp!");
        case 0:
          Out << "__builtin_ia32_cmpeq";
          break;
        case 1:
          Out << "__builtin_ia32_cmplt";
          break;
        case 2:
          Out << "__builtin_ia32_cmple";
          break;
        case 3:
          Out << "__builtin_ia32_cmpunord";
          break;
        case 4:
          Out << "__builtin_ia32_cmpneq";
          break;
        case 5:
          Out << "__builtin_ia32_cmpnlt";
          break;
        case 6:
          Out << "__builtin_ia32_cmpnle";
          break;
        case 7:
          Out << "__builtin_ia32_cmpord";
          break;
      }
      if (ID == Intrinsic::x86_sse_cmp_ps || ID == Intrinsic::x86_sse2_cmp_pd) {
        Out << 'p';
      } else {
        Out << 's';
      }
      if (ID == Intrinsic::x86_sse_cmp_ss || ID == Intrinsic::x86_sse_cmp_ps) {
        Out << 's';
      } else {
        Out << 'd';
      }

      Out << "(";
      writeOperand(I.getArgOperand(0));
      Out << ", ";
      writeOperand(I.getArgOperand(1));
      Out << ")";
      return true;
    case Intrinsic::ppc_altivec_lvsl:
      Out << '(';
      printType(Out, I.getType());
      Out << ')';
      Out << "__builtin_altivec_lvsl(0, (void*)";
      writeOperand(I.getArgOperand(0));
      Out << ")";
      return true;
    case Intrinsic::uadd_with_overflow:
    case Intrinsic::sadd_with_overflow:
      Out << GetValueName(I.getCalledFunction()) << "(";
      writeOperand(I.getArgOperand(0));
      Out << ", ";
      writeOperand(I.getArgOperand(1));
      Out << ")";
      return true;
  }
}

//This converts the llvm constraint string to something gcc is expecting.
//TODO: work out platform independent constraints and factor those out
//      of the per target tables
//      handle multiple constraint codes
std::string CWriter::InterpretASMConstraint(InlineAsm::ConstraintInfo &c) {
  assert(0 && "FIXME: Execute CWriter::InterpretASMConstraint\n");
  assert(c.Codes.size() == 1 && "Too many asm constraint codes to handle");

#if 0
  // Grab the translation table from MCAsmInfo if it exists.
  const MCAsmInfo *TargetAsm;
  std::string Triple = TheModule->getTargetTriple();
  if (Triple.empty()) {
    Triple = llvm::sys::getDefaultTargetTriple();
  }

  std::string E;
  if (const Target *Match = TargetRegistry::lookupTarget(Triple, E)) {
    TargetAsm = Match->createMCAsmInfo(Triple);
  } else {
    return c.Codes[0];
  }
#if 0
  const char *const *table = TargetAsm->getAsmCBE();

  // Search the translation table if it exists.
  for (int i = 0; table && table[i]; i += 2)
    if (c.Codes[0] == table[i]) {
      delete TargetAsm;
      return table[i + 1];
    }
#endif
  // Default is identity.
  delete TargetAsm;
#endif
  return c.Codes[0];
}

//TODO: import logic from AsmPrinter.cpp
static std::string gccifyAsm(std::string asmstr) {
  for (std::string::size_type i = 0; i != asmstr.size(); ++i)
    if (asmstr[i] == '\n') {
      asmstr.replace(i, 1, "\\n");
    } else if (asmstr[i] == '\t') {
      asmstr.replace(i, 1, "\\t");
    } else if (asmstr[i] == '$') {
      if (asmstr[i + 1] == '{') {
        std::string::size_type a = asmstr.find_first_of(':', i + 1);
        std::string::size_type b = asmstr.find_first_of('}', i + 1);
        std::string n = "%" +
                        asmstr.substr(a + 1, b - a - 1) +
                        asmstr.substr(i + 2, a - i - 2);
        asmstr.replace(i, b - i + 1, n);
        i += n.size() - 1;
      } else {
        asmstr.replace(i, 1, "%");
      }
    } else if (asmstr[i] == '%') { //grr
      asmstr.replace(i, 1, "%%");
      ++i;
    }

  return asmstr;
}

//TODO: assumptions about what consume arguments from the call are likely wrong
//      handle communitivity
void CWriter::visitInlineAsm(CallInst &CI) {
  InlineAsm *as = cast<InlineAsm>(CI.getCalledValue());
  InlineAsm::ConstraintInfoVector Constraints = as->ParseConstraints();

  std::vector<std::pair<Value *, int> > ResultVals;
  if (CI.getType() == Type::getVoidTy(CI.getContext()))
    ;
  else if (StructType *ST = dyn_cast<StructType>(CI.getType())) {
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      ResultVals.push_back(std::make_pair(&CI, (int)i));
    }
  } else {
    ResultVals.push_back(std::make_pair(&CI, -1));
  }

  // Fix up the asm string for gcc and emit it.
  Out << "__asm__ volatile (\"" << gccifyAsm(as->getAsmString()) << "\"\n";
  Out << "        :";

  unsigned ValueCount = 0;
  bool IsFirst = true;

  // Convert over all the output constraints.
  for (InlineAsm::ConstraintInfoVector::iterator I = Constraints.begin(),
       E = Constraints.end(); I != E; ++I) {

    if (I->Type != InlineAsm::isOutput) {
      ++ValueCount;
      continue;  // Ignore non-output constraints.
    }

    assert(I->Codes.size() == 1 && "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty()) {
      continue;
    }

    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }

    // Unpack the dest.
    Value *DestVal;
    int DestValNo = -1;

    if (ValueCount < ResultVals.size()) {
      DestVal = ResultVals[ValueCount].first;
      DestValNo = ResultVals[ValueCount].second;
    } else {
      DestVal = CI.getArgOperand(ValueCount - ResultVals.size());
    }

    if (I->isEarlyClobber) {
      C = "&" + C;
    }

    Out << "\"=" << C << "\"(" << GetValueName(DestVal);
    if (DestValNo != -1) {
      Out << ".field" << DestValNo;  // Multiple retvals.
    }
    Out << ")";
    ++ValueCount;
  }


  // Convert over all the input constraints.
  Out << "\n        :";
  IsFirst = true;
  ValueCount = 0;
  for (InlineAsm::ConstraintInfoVector::iterator I = Constraints.begin(),
       E = Constraints.end(); I != E; ++I) {
    if (I->Type != InlineAsm::isInput) {
      ++ValueCount;
      continue;  // Ignore non-input constraints.
    }

    assert(I->Codes.size() == 1 && "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty()) {
      continue;
    }

    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }

    assert(ValueCount >= ResultVals.size() && "Input can't refer to result");
    Value *SrcVal = CI.getArgOperand(ValueCount - ResultVals.size());

    Out << "\"" << C << "\"(";
    if (!I->isIndirect) {
      writeOperand(SrcVal);
    } else {
      writeOperandDeref(SrcVal);
    }
    Out << ")";
  }

  // Convert over the clobber constraints.
  IsFirst = true;
  for (InlineAsm::ConstraintInfoVector::iterator I = Constraints.begin(),
       E = Constraints.end(); I != E; ++I) {
    if (I->Type != InlineAsm::isClobber) {
      continue;  // Ignore non-input constraints.
    }

    assert(I->Codes.size() == 1 && "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty()) {
      continue;
    }

    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }

    Out << '\"' << C << '"';
  }

  Out << ")";
}

void CWriter::visitAllocaInst(AllocaInst &I) {
  Out << '(';
  printType(Out, I.getType());
  Out << ") alloca(sizeof(";
  printType(Out, I.getType()->getElementType());
  Out << ')';
  if (I.isArrayAllocation()) {
    Out << " * " ;
    writeOperand(I.getOperand(0));
  }
  Out << ')';
}

void CWriter::printGEPExpression(Value *Ptr, gep_type_iterator I,
                                 gep_type_iterator E, bool Static) {

  // If there are no indices, just print out the pointer.
  if (I == E) {
    writeOperand(Ptr);
    return;
  }

  // Find out if the last index is into a vector.  If so, we have to print this
  // specially.  Since vectors can't have elements of indexable type, only the
  // last index could possibly be of a vector element.
  VectorType *LastIndexIsVector = 0;
  {
    for (gep_type_iterator TmpI = I; TmpI != E; ++TmpI) {
      LastIndexIsVector = dyn_cast<VectorType>(*TmpI);
    }
  }

  Out << "(";

  // If the last index is into a vector, we can't print it as &a[i][j] because
  // we can't index into a vector with j in GCC.  Instead, emit this as
  // (((float*)&a[i])+j)
  if (LastIndexIsVector) {
    Out << "((";
    printType(Out, PointerType::getUnqual(LastIndexIsVector->getElementType()));
    Out << ")(";
  }

  Out << '&';

  bool IsUserDefinedStructOfArray = false;
  // If the first index is 0 (very typical) we can do a number of
  // simplifications to clean up the code.
  Value *FirstOp = I.getOperand();
  if (!isa<Constant>(FirstOp) || !cast<Constant>(FirstOp)->isNullValue()) {
    // First index isn't simple, print it the hard way.
    writeOperand(Ptr);
  } else {
    ++I;  // Skip the zero index.

    // Okay, emit the first operand. If Ptr is something that is already address
    // exposed, like a global, avoid emitting (&foo)[0], just emit foo instead.
    if (isAddressExposed(Ptr)) {
      writeOperandInternal(Ptr, Static);
    } else if (I != E && (*I)->isStructTy()) {
      // If we didn't already emit the first operand, see if we can print it as
      // P->f instead of "P[0].f"
      writeOperand(Ptr);
      Out << "->field" << cast<ConstantInt>(I.getOperand())->getZExtValue();
      ++I;  // eat the struct index as well.
      if ((*I)->isArrayTy()) {
        IsUserDefinedStructOfArray = true;
      }

    } else {
      // Instead of emitting P[0][1], emit (*P)[1], which is more idiomatic.
      Out << "(*";
      writeOperand(Ptr);
      Out << ")";
    }
  }

  for (; I != E; ++I) {
    if ((*I)->isStructTy()) {
      // For unnamed arrays that are declared during code gen,
      // there won't be a GEP that has struct.array access
      // but for arrays that are enclosed in a user-defined structure,
      // we will see struct.array in GEP
      gep_type_iterator J = I;
      if (++J != E && ((*J)->isArrayTy())) {
        IsUserDefinedStructOfArray = true;
      }
      Out << ".field" << cast<ConstantInt>(I.getOperand())->getZExtValue();
    } else if ((*I)->isArrayTy()) {
      if (!IsUserDefinedStructOfArray) {
        Out << ".field0";
      }
      Out << ".array[";
      writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
      Out << ']';
      IsUserDefinedStructOfArray = false;
    } else if (!(*I)->isVectorTy()) {
      Out << '[';
      writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
      Out << ']';
    } else {
      // If the last index is into a vector, then print it out as "+j)".  This
      // works with the 'LastIndexIsVector' code above.
      if (isa<Constant>(I.getOperand()) &&
          cast<Constant>(I.getOperand())->isNullValue()) {
        Out << "))";  // avoid "+0".
      } else {
        Out << ")+(";
        writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
        Out << "))";
      }
    }
  }
  Out << ")";
}
#ifdef MXPA_CODEGEN

void CWriter::writeLoadInst(llvm::Value *Operand, llvm::Type *OperandType,
                            bool IsVolatile, unsigned Alignment, bool isUpperBound, Value *arg) {
  bool IsUnaligned = Alignment &&
                     Alignment < TD->getABITypeAlignment(OperandType);

  if (!IsUnaligned) {
    Out << '*';
    Out << '(';
    printType(Out, OperandType);
    Out << "*)";
  }
  if (IsVolatile || IsUnaligned) {
    Out << "((";
    if (IsUnaligned) {
      Out << "struct __attribute__ ((packed, aligned(" << Alignment << "))) {";
    }
    printType(Out, OperandType, false, IsUnaligned ? "data" : "volatile*");
    if (IsUnaligned) {
      Out << "; } ";
      if (IsVolatile) {
        Out << "volatile ";
      }
      Out << "*";
    }
    Out << ")";
  }

  bool isAddressImplicit = isAddressExposed(Operand);
  if (isAddressImplicit) {
    Out << "(&";  // Global variables are referenced as their addresses by llvm
  }


  {
    bool flag = true;
    if (Instruction *I = dyn_cast<Instruction>(Operand)) {
      // Should we inline this instruction to build a tree?
      if (isInlinableInst(*I) && !isDirectAlloca(I)) {
        Out << '(';
        //      writeInstComputationInline(*I);
        writeBounds(SE->getSCEV(Operand), isUpperBound, arg);
        Out << ')';
        flag = false;
      }
    }

    if (flag) {
      Constant *CPV = dyn_cast<Constant>(Operand);

      if (CPV && !isa<GlobalValue>(CPV)) {
        printConstant(CPV, false);
      } else {
        Out << GetValueName(Operand);
      }
    }
  }

  if (isAddressImplicit) {
    Out << ')';
  }

  if (IsVolatile || IsUnaligned) {
    Out << ')';
    if (IsUnaligned) {
      Out << "->data";
    }
  }
}


// In bound extraction, there are only two kinds of unknowns allowed:
// 1) calls to get_{global/local}_ids
void CWriter::writeBoundsUnknown(Value *v, bool isUpperBound, Value *arg) {
  if (CallInst *CI = dyn_cast<CallInst>(v)) {
    Out << "mxpa_";
    writeOperand(CI->getCalledValue());
    Out << '(';
    CallSite CS(CI);
    CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
    unsigned ArgNo = 0;
    bool PrintedArg = false;
    for (; AI != AE; ++AI, ++ArgNo) {
      if (PrintedArg) {
        Out << ", ";
      }
      const SCEV *scev = SE->getSCEV(*AI);
      writeBounds(scev, isUpperBound, arg);
      PrintedArg = true;
    }
    Out << ')';
  } else if (LoadInst *I = dyn_cast<LoadInst>(v)) {
    writeLoadInst(I->getOperand(0), I->getType(), I->isVolatile(),
                  I->getAlignment(), isUpperBound, arg);
  } else if (SDivOperator *CI = dyn_cast<SDivOperator>(v)) {
    Out << "(";
    writeBounds(SE->getSCEV(CI->getOperand(0)), isUpperBound, arg);
    Out << " / ";
    writeBounds(SE->getSCEV(CI->getOperand(1)), isUpperBound, arg);
    Out << ")";
  } else if (Instruction *I = dyn_cast<Instruction>(v)) {
    visit(*I);
  } else {
    if (v->getName()  == arg->getName()) {
      Out << "0U";
    } else if (v->getType()->isPointerTy()) {
      Out << "(unsigned char *)(";
      writeOperand(v);
      Out << ")";
    } else {
      writeOperand(v);
    }
  }
}

void CWriter::writeBounds(const SCEV *scev, bool isUpperBound, Value *arg) {
  switch (scev->getSCEVType()) {
    case scConstant: {
        scev->print(Out);
        break;
      }
    case scTruncate: {
        Out << "(";
        const SCEVTruncateExpr *Trunc = cast<SCEVTruncateExpr>(scev);
        const SCEV *op = Trunc->getOperand();
        writeBounds(op, isUpperBound, arg);
        Out << ")";
        break;
      }
    case scZeroExtend: {
        Out << "(";
        const SCEVZeroExtendExpr *A = cast<SCEVZeroExtendExpr>(scev);
        const SCEV *op = A->getOperand();
        writeBounds(op, isUpperBound, arg);
        Out << ")";
        break;
      }
    case scSignExtend: {
        const SCEVSignExtendExpr *SExt = cast<SCEVSignExtendExpr>(scev);
        const SCEV *op = SExt->getOperand();
        writeBounds(op, isUpperBound, arg);
        break;
      }
    case scAddExpr: {
        const SCEVAddExpr *A = cast<SCEVAddExpr>(scev);
        Out << "(";
        for (size_t i = 0; i < A->getNumOperands(); i++) {
          if (i) {
            Out << " + ";
          }
          const SCEV *op = A->getOperand(i);
          writeBounds(op, isUpperBound, arg);
        }
        Out << ")";
        break;
      }
    case scMulExpr: {
        const SCEVMulExpr *M = cast<SCEVMulExpr>(scev);
        Out << "(";
        for (size_t i = 0; i < M->getNumOperands(); i++) {
          if (i) {
            Out << " * ";
          }
          const SCEV *op = M->getOperand(i);
          writeBounds(op, isUpperBound, arg);
        }
        Out << ")";
        break;
      }
    case scUDivExpr: {
        const SCEVUDivExpr *UD = cast<SCEVUDivExpr>(scev);
        Out << "(";
        const SCEV *op = UD->getLHS();
        writeBounds(op, isUpperBound, arg);
        Out << " / ";
        op = UD->getRHS();
        writeBounds(op, isUpperBound, arg);
        Out << ")";
        break;
      }
    case scAddRecExpr: {
        const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(scev);
        const SCEV *start = AR->getStart();
        Out << "(";
        writeBounds(start, isUpperBound, arg);
        if (isUpperBound) {
          Out << ") + ";
          writeBounds(AR->getOperand(1), isUpperBound, arg);
          Out << "* (";
          writeBounds(SE->getBackedgeTakenCount(AR->getLoop()), isUpperBound, arg);
        }
        Out << ")";
        break;
      }
    case scUMaxExpr: {
        const SCEVUMaxExpr *M = cast<SCEVUMaxExpr>(scev);
        Out << "mxpa_smax(";
        assert(M->getNumOperands() == 2);
        const SCEV *op = M->getOperand(0);
        writeBounds(op, isUpperBound, arg);
        Out << ", ";
        op = M->getOperand(1);
        writeBounds(op, isUpperBound, arg);
        Out << ")";
        break;
      }
    case scSMaxExpr: {
        const SCEVSMaxExpr *M = cast<SCEVSMaxExpr>(scev);
        Out << "mxpa_smax(";
        assert(M->getNumOperands() == 2);
        const SCEV *op = M->getOperand(0);
        writeBounds(op, isUpperBound, arg);
        Out << ", ";
        op = M->getOperand(1);
        writeBounds(op, isUpperBound, arg);
        Out << ")";
        break;
      }
    case scUnknown: {
        const SCEVUnknown *U = cast<SCEVUnknown>(scev);
        Value *v = U->getValue();
        writeBoundsUnknown(v, isUpperBound, arg);
        break;
      }
    case scCouldNotCompute:
      Out << "***COULDNOTCOMPUTE***";
      return;
    default:
      // This condition may happen because some SCEVType may be lost in the
      // former switch cases.
      assert(false && "A SCEVType is lost!!!");
      break;
  };
#if 0
  llvm::errs() << "Inner loop trip count: ";
  SE->getBackedgeTakenCount(AR->getLoop())->dump();
#endif
}

static bool checkValidFun(std::string funname) {
  std::string funList[] = {
    "get_global_id",
    "get_local_id",
    "get_global_size",
    "get_local_size",
    "get_global_offset",
    "get_work_dim",
    "get_group_id",
    "get_num_groups",
    "_Z5mad24iii"
  };
  unsigned i = 0;
  for (; i < sizeof(funList) / sizeof(std::string); ++i) {
    if (funname == funList[i]) {
      break;
    }
  }
  if (i == sizeof(funList) / sizeof(std::string)) {
    return 1;
  }
  return 0;
}

void CWriter::CheckBounds(const SCEV *scev, bool isUpperBound, char &ret) {
  switch (scev->getSCEVType()) {
    case scTruncate: {
        const SCEVTruncateExpr *Trunc = cast<SCEVTruncateExpr>(scev);
        const SCEV *op = Trunc->getOperand();
        CheckBounds(op, isUpperBound, ret);
        break;
      }
    case scZeroExtend: {
        const SCEVZeroExtendExpr *A = cast<SCEVZeroExtendExpr>(scev);
        const SCEV *op = A->getOperand();
        CheckBounds(op, isUpperBound, ret);
        break;
      }
    case scSignExtend: {
        const SCEVSignExtendExpr *SExt = cast<SCEVSignExtendExpr>(scev);
        const SCEV *op = SExt->getOperand();
        CheckBounds(op, isUpperBound, ret);
        break;
      }
    case scAddExpr: {
        const SCEVAddExpr *A = cast<SCEVAddExpr>(scev);
        for (size_t i = 0; i < A->getNumOperands(); i++) {
          CheckBounds(A->getOperand(i), isUpperBound, ret);
        }
        break;
      }
    case scMulExpr: {
        const SCEVMulExpr *M = cast<SCEVMulExpr>(scev);
        for (size_t i = 0; i < M->getNumOperands(); i++) {
          CheckBounds(M->getOperand(i), isUpperBound, ret);
        }
        break;
      }
    case scUDivExpr: {
        const SCEVUDivExpr *UD = cast<SCEVUDivExpr>(scev);
        const SCEV *op = UD->getLHS();
        CheckBounds(op, isUpperBound, ret);
        op = UD->getRHS();
        CheckBounds(op, isUpperBound, ret);
        break;
      }
    case scAddRecExpr: {
        const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(scev);
        const SCEV *start = AR->getStart();
        CheckBounds(start, isUpperBound, ret);
        if (isUpperBound) {
          CheckBounds(AR->getOperand(1), isUpperBound, ret);
          CheckBounds(SE->getBackedgeTakenCount(AR->getLoop()), isUpperBound,
                      ret);
        }
        break;
      }
    case scSMaxExpr: {
        const SCEVSMaxExpr *M = cast<SCEVSMaxExpr>(scev);
        assert(M->getNumOperands() == 2);
        CheckBounds(M->getOperand(0), isUpperBound, ret);
        CheckBounds(M->getOperand(1), isUpperBound, ret);
        break;
      }
    case scUnknown: {
        const SCEVUnknown *U = cast<SCEVUnknown>(scev);
        Value *v = U->getValue();
        if (CallInst *CI = dyn_cast<CallInst>(v)) {
          if (checkValidFun(CI->getCalledFunction()->getName())) {
            ret = (ret & 0xFE) | 0x1;
            break;
          }
          CallSite CS(CI);
          CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
          unsigned ArgNo = 0;
          for (; AI != AE; ++AI, ++ArgNo) {
            CheckBounds(SE->getSCEV(*AI), isUpperBound, ret);
          }
        } else if (LoadInst *I = dyn_cast<LoadInst>(v)) {
          CheckBounds(SE->getSCEV(I->getOperand(0)), isUpperBound, ret);
          ret = (ret & 0xFD) | 0x2;
          break;
        } else if (SDivOperator *DI = dyn_cast<SDivOperator>(v)) {
          CheckBounds(SE->getSCEV(DI->getOperand(0)), isUpperBound, ret);
          CheckBounds(SE->getSCEV(DI->getOperand(1)), isUpperBound, ret);
          break;
        } else if (dyn_cast<PHINode>(v)) {
          ret = (ret & 0xFE) | 0x1;
          break;
        } else if (!isa<Argument>(v)) {
          ret = (ret & 0xFE) | 0x1;
        }
        break;
      }
    case scCouldNotCompute: {
        ret = (ret & 0xFE) | 0x1;
        break;
      }
  };
}

namespace {
static Value *GetSCEVBase(ScalarEvolution *SE, const SCEV *scev) {
  switch (scev->getSCEVType()) {
    case scAddRecExpr: {
        const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(scev);
        const SCEV *start = AR->getStart();
        return GetSCEVBase(SE, start);
        break;
      }
    case scAddExpr: {
        const SCEVAddExpr *A = cast<SCEVAddExpr>(scev);
        for (size_t i = 0; i < A->getNumOperands(); i++) {
          const SCEV *op = A->getOperand(i);
          Value *v = GetSCEVBase(SE, op);
          if (v) {
            return v;
          }
        }
        break;
      }
    case scUnknown: {
        const SCEVUnknown *U = cast<SCEVUnknown>(scev);
        Value *v = U->getValue();
        // pointer arguments are disregarded
        if (v->getType()->isPointerTy()) {
          return v;
        }
        break;
      }
  };
  return NULL;
}

static bool isInnerArgument(Value *V) {
  bool isInnerArg = false;
  for (Value::user_iterator i = V->user_begin(), ie = V->user_end();
       i != ie; ++i) {
    if (isa<CallInst>(*i)) {
      isInnerArg = isInnerArg || true;
    }
  }
  return isInnerArg;
}
}
#endif
void CWriter::writeMemoryAccess(Value *Operand, Type *OperandType,
                                bool IsVolatile, unsigned Alignment) {
#if MXPA_CODEGEN
  const SCEV *scev = SE->getSCEV(Operand);
  Value *v = GetSCEVBase(SE, scev);
  if (v) {
    std::set<const SCEV *> &scevs = MemoryAccesses[v];
    scevs.insert(scev);
    for (Value::user_iterator ui = Operand->user_begin(), ue = Operand->user_end();
         ui != ue; ui++) {
      Value *uv = *ui;
      if (Instruction *I = dyn_cast<Instruction>(uv)) {
        if (I->mayWriteToMemory()) {
          MemoryWrites.insert(scev);
        }
      }
    }
  } else {
    errs() << "CWriter: Cannot locate the base of a memory access. Upper/lower "
           "bound analysis may be inaccurate\n";
  }
#endif

#if 0
  bool IsUnaligned = Alignment &&
                     Alignment < TD->getABITypeAlignment(OperandType);

  if (!IsUnaligned)
#endif
    Out << '*';
#if 0
  if (IsVolatile || IsUnaligned) {
    Out << "((";
    if (IsUnaligned) {
      Out << "struct __attribute__ ((packed, aligned(" << Alignment << "))) {";
    }
    printType(Out, OperandType, false, IsUnaligned ? "data" : "volatile*");
    if (IsUnaligned) {
      Out << "; } ";
      if (IsVolatile) {
        Out << "volatile ";
      }
      Out << "*";
    }
    Out << ")";
  }
#endif

  writeOperand(Operand);
#if 0
  if (IsVolatile || IsUnaligned) {
    Out << ')';
    if (IsUnaligned) {
      Out << "->data";
    }
  }
#endif
}

void CWriter::visitLoadInst(LoadInst &I) {
  const SCEV *scev = SE->getSCEV(I.getOperand(0));
  Value *v = GetSCEVBase(SE, scev);
  MemoryElementSize[v] = SE->getElementSize(&I);
  writeMemoryAccess(I.getOperand(0), I.getType(), I.isVolatile(),
                    I.getAlignment());

}

void CWriter::visitStoreInst(StoreInst &I) {
  ConstantArray *CA = dyn_cast<ConstantArray>(I.getOperand(0));
  if (I.getOperand(0)->getType()->isArrayTy() && CA) {
    Out << "memcpy(";
    writeOperand(I.getPointerOperand());
    Out << ", ";
    printConstantArray(CA, true);
    Out << ", sizeof(";
    printType(Out, I.getOperand(0)->getType());
    Out << "))";
  } else {
    const SCEV *scev = SE->getSCEV(I.getPointerOperand());
    Value *v = GetSCEVBase(SE, scev);
    MemoryElementSize[v] = SE->getElementSize(&I);
    writeMemoryAccess(I.getPointerOperand(), I.getOperand(0)->getType(),
                      I.isVolatile(), I.getAlignment());
    Out << " = ";
    Value *Operand = I.getOperand(0);
    Constant *BitMask = 0;
    if (IntegerType *ITy = dyn_cast<IntegerType>(Operand->getType()))
      if (!ITy->isPowerOf2ByteWidth())
        // We have a bit width that doesn't match an even power-of-2 byte
        // size. Consequently we must & the value with the type's bit mask
      {
        BitMask = ConstantInt::get(ITy, ITy->getBitMask());
      }
    if (BitMask) {
      Out << "((";
    }
    writeOperand(Operand);
    if (BitMask) {
      Out << ") & ";
      printConstant(BitMask, false);
      Out << ")";
    }
  }
}

void CWriter::visitGetElementPtrInst(GetElementPtrInst &I) {
  printGEPExpression(I.getPointerOperand(), gep_type_begin(I),
                     gep_type_end(I), false);
}

void CWriter::visitVAArgInst(VAArgInst &I) {
  Out << "va_arg(*(va_list*)";
  writeOperand(I.getOperand(0));
  Out << ", ";
  printType(Out, I.getType());
  Out << ");\n ";
}

void CWriter::visitInsertElementInst(InsertElementInst &I) {
  Type *EltTy = I.getType()->getElementType();
  writeOperand(I.getOperand(0));
  Out << ";\n  ";
  Out << "((";
  printType(Out, PointerType::getUnqual(EltTy));
  Out << ")(&" << GetValueName(&I) << "))[";
  writeOperand(I.getOperand(2));
  Out << "] = (";
  writeOperand(I.getOperand(1));
  Out << ")";
}

void CWriter::visitExtractElementInst(ExtractElementInst &I) {
  // We know that our operand is not inlined.
  Out << "((";
  Type *EltTy =
    cast<VectorType>(I.getOperand(0)->getType())->getElementType();
  printType(Out, PointerType::getUnqual(EltTy));
  Out << ")(&" << GetValueName(I.getOperand(0)) << "))[";
  writeOperand(I.getOperand(1));
  Out << "]";
}

// <result> = shufflevector <n x <ty>> <v1>, <n x <ty>> <v2>, <m x i32> <mask>
// ; yields <m x <ty>>
void CWriter::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  Out << "(";
  printType(Out, SVI.getType());
  Out << "){ ";
  VectorType *VT = SVI.getType();
  unsigned NumElts = VT->getNumElements();
  Type *EltTy = VT->getElementType();

  for (unsigned i = 0; i != NumElts; ++i) {
    if (i) {
      Out << ", ";
    }
    int SrcVal = SVI.getMaskValue(i);
    if ((unsigned)SrcVal >= NumElts * 2) {
      Out << " 0/*undef*/ ";
    } else {
      Value *Op = SVI.getOperand((unsigned)SrcVal >= NumElts);
      if (isa<Instruction>(Op)) {
        // Do an extractelement of this value from the appropriate input.
        Out << "((";
        printType(Out, PointerType::getUnqual(EltTy));
        Out << ")(&" << GetValueName(Op)
            << "))[" << (SrcVal & (NumElts - 1)) << "]";
      } else if (isa<ConstantAggregateZero>(Op) || isa<UndefValue>(Op)) {
        Out << "0";
      } else {
        printConstant(cast<ConstantVector>(Op)->getOperand(SrcVal &
                      (NumElts - 1)),
                      false);
      }
    }
  }
  Out << "}";
}

void CWriter::visitInsertValueInst(InsertValueInst &IVI) {
  // Start by copying the entire aggregate value into the result variable.
  writeOperand(IVI.getOperand(0));
  Out << ";\n  ";

  // Then do the insert to update the field.
  Out << GetValueName(&IVI);
  for (const unsigned *b = IVI.idx_begin(), *i = b, *e = IVI.idx_end();
       i != e; ++i) {
    Type *IndexedTy =
      ExtractValueInst::getIndexedType(IVI.getOperand(0)->getType(),
                                       makeArrayRef(b, i + 1));
    if (IndexedTy->isArrayTy()) {
      Out << ".array[" << *i << "]";
    } else {
      Out << ".field" << *i;
    }
  }
  Out << " = ";
  writeOperand(IVI.getOperand(1));
}

void CWriter::visitExtractValueInst(ExtractValueInst &EVI) {
  Out << "(";
  if (isa<UndefValue>(EVI.getOperand(0))) {
    Out << "(";
    printType(Out, EVI.getType());
    Out << ") 0/*UNDEF*/";
  } else {
    Out << GetValueName(EVI.getOperand(0));
    for (const unsigned *b = EVI.idx_begin(), *i = b, *e = EVI.idx_end();
         i != e; ++i) {
      Type *IndexedTy =
        ExtractValueInst::getIndexedType(EVI.getOperand(0)->getType(),
                                         makeArrayRef(b, i + 1));
      if (IndexedTy->isArrayTy()) {
        Out << ".array[" << *i << "]";
      } else {
        Out << ".field" << *i;
      }
    }
  }
  Out << ")";
}

/// getKernelFnCallees - Get the callees of a kernel functions if the
/// function attributes of the callees are the same as the kernel function's.
void CWriter::getKernelFnCallees(const Module &M) {
  for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (!isKernelFunction(I)) {
      continue;
    }

    KernelFunctionAS = I->getAttributes().getFnAttributes();

    for (Function::const_iterator b = I->begin(), be = I->end();
         b != be; ++b) {
      for (BasicBlock::const_iterator i = b->begin(), ie = b->end();
           i != ie; ++i) {
        const CallInst *callInst = dyn_cast<CallInst>(&*i);
        if (!callInst) {
          continue;
        }
        const Function *kcallee = callInst->getCalledFunction();
        if (!kcallee) {
          continue;
        }

        const AttributeSet &kcalleeAS =
          kcallee->getAttributes().getFnAttributes();
        if (kcalleeAS != KernelFunctionAS) {
          continue;
        }

        std::set<const Function *>::iterator IC = KernelFnCallees.find(kcallee);
        if (IC != KernelFnCallees.end()) {
          continue;
        }
        KernelFnCallees.insert(kcallee);
      }
    }
  }
}

/// getInlinedCallees - Get the inlineable non-kernel functions on conditions
/// that (1) Their function attributes are the same as the kernel's, and
//  (2) They are not in the set KernelFnCallees.
void CWriter::getInlinedCallees(const Module &M) {
  for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I) {
    const AttributeSet &AS = I->getAttributes().getFnAttributes();
    if (AS != KernelFunctionAS) {
      continue;
    }

    if (isKernelFunction(I)) {
      continue;
    }

    std::set<const Function *>::iterator it = KernelFnCallees.find(I);
    if (it != KernelFnCallees.end()) {
      if (NonInlinedCallees.find(I) != NonInlinedCallees.end()) {
        continue;
      }
      NonInlinedCallees.insert(I);
    }
    InlinedCallees.insert(I);
  }
}

/// getLaunchKernels - Get the list of launch kernes annotated by metadata
/// launch.kernels
void CWriter::getLaunchKernels(const Module &M) {
  NamedMDNode *launchKernelMetadata =
    TheModule->getNamedMetadata("launch.kernels");

  if (!launchKernelMetadata) {
    return;
  }

  unsigned launchKernelNum = launchKernelMetadata->getNumOperands();
  for (unsigned K = 0, E = launchKernelNum; K != E; ++K) {
    MDNode *kernelMD =  dyn_cast<MDNode>(launchKernelMetadata->getOperand(K));
    if (!kernelMD->getOperand(0)) {
      continue;
    }
    Function *fun = dyn_cast<Function>(kernelMD->getOperand(0));
    LaunchKernels.insert(fun);
  }
}

void CWriter::run(const Module &M, bool onlyNamed) {
  OnlyNamed = onlyNamed;

  getKernelFnCallees(M);
  getInlinedCallees(M);
  getLaunchKernels(M);

  // Get types from global variables.
  for (Module::const_global_iterator I = M.global_begin(),
       E = M.global_end(); I != E; ++I) {
    incorporateType(M, I->getType());
    if (I->hasInitializer()) {
      incorporateValue(M, I->getInitializer());
    }
  }

  // Get types from aliases.
  for (Module::const_alias_iterator I = M.alias_begin(),
       E = M.alias_end(); I != E; ++I) {
    incorporateType(M, I->getType());
    if (const Value *Aliasee = I->getAliasee()) {
      incorporateValue(M, Aliasee);
    }
  }

  // Get types from functions.
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDForInst;
  for (Module::const_iterator FI = M.begin(), E = M.end(); FI != E; ++FI) {
    incorporateType(M, FI->getType());

    // First incorporate the arguments.
    for (Function::const_arg_iterator AI = FI->arg_begin(),
         AE = FI->arg_end(); AI != AE; ++AI) {
      incorporateValue(M, AI);
    }

    for (Function::const_iterator BB = FI->begin(), E = FI->end();
         BB != E; ++BB)
      for (BasicBlock::const_iterator II = BB->begin(),
           E = BB->end(); II != E; ++II) {
        const Instruction &I = *II;

        // Incorporate the type of the instruction.
        incorporateType(M, I.getType());

        // Incorporate non-instruction operand types. (We are incorporating all
        // instructions with this loop.)
        for (User::const_op_iterator OI = I.op_begin(), OE = I.op_end();
             OI != OE; ++OI)
          if (!isa<Instruction>(OI)) {
            incorporateValue(M, *OI);
          }

        // Incorporate types hiding in metadata.
        I.getAllMetadataOtherThanDebugLoc(MDForInst);
        for (unsigned i = 0, e = MDForInst.size(); i != e; ++i) {
          incorporateMDNode(M, MDForInst[i].second);
        }

        MDForInst.clear();
      }
  }

  for (Module::const_named_metadata_iterator I = M.named_metadata_begin(),
       E = M.named_metadata_end(); I != E; ++I) {
    const NamedMDNode *NMD = I;
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
      incorporateMDNode(M, NMD->getOperand(i));
    }
  }
}
/*
void TypeFinder::clear() {
  VisitedConstants.clear();
  VisitedTypes.clear();
  StructTypes.clear();
}
*/
/// incorporateType - This method adds the type to the list of used structures
/// if it's not in there already.
void CWriter::incorporateType(const Module &M, Type *Ty) {
  // Check to see if we're already visited this type.
  if (!VisitedTypes.insert(Ty).second) {
    return;
  }

  // If this is a structure or opaque type, add a name for the type.
  if (StructType *STy = dyn_cast<StructType>(Ty))
    if (!OnlyNamed || STy->hasName()) {
      StructTypes.push_back(STy);
    }

  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    std::vector<Type *> structMembers;
    structMembers.push_back(ATy);
    StructTypes.push_back(StructType::get(M.getContext(), structMembers));
  }

  // Recursively walk all contained types.
  for (Type::subtype_iterator I = Ty->subtype_begin(),
       E = Ty->subtype_end(); I != E; ++I) {
    incorporateType(M, *I);
  }
}

/// incorporateValue - This method is used to walk operand lists finding types
/// hiding in constant expressions and other operands that won't be walked in
/// other ways.  GlobalValues, basic blocks, instructions, and inst operands are
/// all explicitly enumerated.
void CWriter::incorporateValue(const Module &M, const Value *V) {
  if (const MDNode *MNe = dyn_cast<MDNode>(V)) {
    return incorporateMDNode(M, MNe);
  }

  if (!isa<Constant>(V) || isa<GlobalValue>(V)) {
    return;
  }

  // Already visited?
  if (!VisitedConstants.insert(V).second) {
    return;
  }

  // Check this type.
  incorporateType(M, V->getType());

  // If this is an instruction, we incorporate it separately.
  if (isa<Instruction>(V)) {
    return;
  }

  // Look in operands for types.
  const User *U = cast<User>(V);
  for (Constant::const_op_iterator I = U->op_begin(),
       E = U->op_end(); I != E; ++I) {
    incorporateValue(M, *I);
  }
}

/// incorporateMDNode - This method is used to walk the operands of an MDNode to
/// find types hiding within.
void CWriter::incorporateMDNode(const Module &M, const MDNode *V) {
  // Already visited?
  if (!VisitedConstants.insert(V).second) {
    return;
  }

  // Look in operands for types.
  for (unsigned i = 0, e = V->getNumOperands(); i != e; ++i)
    if (Value *Op = V->getOperand(i)) {
      incorporateValue(M, Op);
    }
}

//===----------------------------------------------------------------------===//
//                       External Interface declaration
//===----------------------------------------------------------------------===//

bool CTargetMachine::addPassesToEmitFile(PassManagerBase &PM,
    formatted_raw_ostream &o,
    CodeGenFileType FileType,
    bool DisableVerify,
    AnalysisID StartAfter,
    AnalysisID StopAfter) {
  if (FileType != TargetMachine::CGFT_AssemblyFile) {
    return true;
  }

  PM.add(createGCLoweringPass());
  PM.add(createLowerInvokePass());
  PM.add(createCFGSimplificationPass());   // clean up after lower invoke.
  PM.add(new CWriter(o));
  return false;
}
