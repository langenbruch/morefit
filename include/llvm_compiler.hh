#ifndef LLVM_COMPILER_HH
#define LLVM_COMPILER_HH

// code based on tutorials at
// https://blog.memzero.de/llvm-orc-jit/
// https://www.llvm.org/docs/tutorial/BuildingAJIT1.html

#include <clang/Basic/DiagnosticOptions.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>

#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Host.h>

#include "compute.hh"

namespace cc {

  using clang::CompilerInstance;
  using clang::CompilerInvocation;
  using clang::DiagnosticConsumer;
  using clang::DiagnosticOptions;
  using clang::DiagnosticsEngine;
  using clang::EmitLLVMOnlyAction;
  using clang::TextDiagnosticPrinter;

  using llvm::Expected;
  using llvm::IntrusiveRefCntPtr;
  using llvm::LLVMContext;
  using llvm::MemoryBuffer;
  using llvm::Module;
  using llvm::StringError;

  class CCompiler {
  public:
    CCompiler() {
      //Setup custom diagnostic options.
      auto DO = IntrusiveRefCntPtr<DiagnosticOptions>(new DiagnosticOptions());
      DO->ShowColors = 1;

      //Setup stderr custom diagnostic consumer.
      DC = std::make_unique<TextDiagnosticPrinter>(llvm::errs(), DO.get());

      //Create custom diagnostics engine.
      //The engine will NOT take ownership of the DiagnosticConsumer object.
      DE = std::make_unique<DiagnosticsEngine>(
					       nullptr /* DiagnosticIDs */, std::move(DO), DC.get(),
					       false /* own DiagnosticConsumer */);
    }

    struct CompileResult {
      std::unique_ptr<LLVMContext> C;
      std::unique_ptr<Module> M;
    };

    Expected<CompileResult> compile(const char* code, morefit::compute_options* opts) const {
      using std::errc;
      const auto err = [](errc ec) { return std::make_error_code(ec); };
      const char code_fname[] = "jit.c";

      //Create compiler instance.
      CompilerInstance CC;
      //Setup compiler invocation.
      std::vector<const char*> args;
      std::string triple = "-triple=" + opts->llvm_triple;
      if (opts->llvm_triple != "")
	args.push_back(triple.c_str());
      if (opts->llvm_optimization != "")
	args.push_back(opts->llvm_optimization.c_str());
      else
	  args.push_back("-O3");
      if (opts->llvm_vectorization)
	args.push_back("-fveclib=libmvec");
      //possibly configure via cmake?
      args.push_back("-I/usr/include");
      args.push_back("-I/usr/include/x86_64-linux-gnu");
      
      args.push_back(code_fname);
      bool ok = CompilerInvocation::CreateFromArgs(CC.getInvocation(), args, *DE);
      assert(ok);

      //Setup custom diagnostic printer.
      CC.createDiagnostics(DC.get(), false /* own DiagnosticConsumer */);

      //Configure remapping from pseudo file name to in-memory code buffer
      //PreprocessorOptions take ownership of MemoryBuffer.
      CC.getPreprocessorOpts().addRemappedFile(code_fname, MemoryBuffer::getMemBuffer(code).release());

      //Configure codegen options.
      auto& CG = CC.getCodeGenOpts();
      
      if (opts->print_level > 1)
	{
	  std::cout << "Command line arguments ";
	  for (unsigned int i=0; i<CG.CommandLineArgs.size(); i++)
	    std::cout << CG.CommandLineArgs.at(i) << " ";
	  std::cout << std::endl;
	}
      //valid target CPU values are: nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont, nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge, core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512, skx, cascadelake, cooperlake, cannonlake, icelake-client, rocketlake, icelake-server, tigerlake, sapphirerapids, alderlake, raptorlake, meteorlake, sierraforest, grandridge, graniterapids, graniterapids-d, emeraldrapids, knl, knm, k8, athlon64, athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10, barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3, znver4, x86-64, x86-64-v2, x86-64-v3, x86-64-v4      
      if (opts->llvm_cpu != "")
	CC.getTargetOpts().CPU = opts->llvm_cpu;
      //valid target CPU values are: i386, i486, winchip-c6, winchip2, c3, i586, pentium, pentium-mmx, pentiumpro, i686, pentium2, pentium3, pentium3m, pentium-m, c3-2, yonah, pentium4, pentium4m, prescott, nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont, nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge, core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512, skx, cascadelake, cooperlake, cannonlake, icelake-client, rocketlake, icelake-server, tigerlake, sapphirerapids, alderlake, raptorlake, meteorlake, sierraforest, grandridge, graniterapids, graniterapids-d, emeraldrapids, knl, knm, lakemont, k6, k6-2, k6-3, athlon, athlon-tbird, athlon-xp, athlon-mp, athlon-4, k8, athlon64, athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10, barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3, znver4, x86-64, geode
      if (opts->llvm_tunecpu != "")
	CC.getTargetOpts().TuneCPU = opts->llvm_tunecpu;

      if (opts->print_level > 1)
	{
	  std::cout << "Target CPU " << CC.getTargetOpts().CPU << std::endl;
	  std::cout << "Target TuneCPU " << CC.getTargetOpts().TuneCPU << std::endl;
	  std::cout << "Target Triple " << CC.getTargetOpts().Triple << std::endl;
	  std::cout << "Target Features ";
	  for (unsigned int i=0; i<CC.getTargetOpts().Features.size(); i++)
	    std::cout << CC.getTargetOpts().Features.at(i) << " ";
	  std::cout << std::endl;
	}
      CG.setInlining(clang::CodeGenOptions::NormalInlining);

      auto t_before_llvm_ir = std::chrono::high_resolution_clock::now();
      //Generate LLVM IR.
      EmitLLVMOnlyAction A;
      if (!CC.ExecuteAction(A)) {
	return llvm::make_error<StringError>(
					     "Failed to generate LLVM IR from C code!",
					     err(errc::invalid_argument));
      }
      auto t_after_llvm_ir = std::chrono::high_resolution_clock::now();
      
      if (opts->print_level > 1)
	std::cout << "compiling LLVM IR takes " << std::chrono::duration<double, std::milli>(t_after_llvm_ir-t_before_llvm_ir).count() << " ms in total" << std::endl;

      //Take generated LLVM IR module and the LLVMContext.
      auto M = A.takeModule();
      auto C = std::unique_ptr<LLVMContext>(A.takeLLVMContext());
      assert(M);
      return CompileResult{std::move(C), std::move(M)};
    }

  private:
    std::unique_ptr<DiagnosticConsumer> DC;
    std::unique_ptr<DiagnosticsEngine> DE;
  };

}  //namespace cc

#endif
