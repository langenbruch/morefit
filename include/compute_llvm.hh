/**
 * @file compute_llvm.hh
 * @author Christoph Langenbruch
 * @date 2024-11-25
 *
 */

#ifndef COMPUTELLVM_H
#define COMPUTELLVM_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include <array>
#include <string>
#include <regex>
#include <thread>

#include "eventvector.hh"
#include "parametervector.hh"
#include "graph.hh"
#include "compute.hh"
#include "utils.hh"

#include "llvm_compiler.hh"
#include "llvm_jit.hh"

namespace morefit {

  class LLVMBackend: public ComputeBackend {
  public:
    LLVMBackend(compute_options* opts):
      ComputeBackend(opts)
    {}
  };

  //we need to use a simple form of thread pooling, ie we need to avoid repeated spawning and joining of threads
  //to this end we use a spinning worker
  //inspired by https://lemire.me/blog/2020/06/10/reusing-a-thread-in-c-for-better-performance/
  template<typename kernelT, typename evalT, typename seedT=uint32_t>
  class worker {
  public:
    enum worker_type { generate, compute, compute_param, compute_kahan, compute_param_kahan};
    worker_type wt_;    
  private:
    void* fcn_ptr_;
    bool has_work_{false};
    bool exiting_{false};
    std::mutex locking_mutex_{};
    std::condition_variable cond_var_{};
    std::thread thread_{};
    //arguments
    int nevents_ = 0;
    int nevents_padded_ = 0;
    int from_ = 0;
    int to_ = 0;
    int thread_idx_ = 0;
    int nthreads_ = 0;
    seedT* seed_ = nullptr;
    kernelT* input_ = nullptr;
    kernelT* output_ = nullptr;
    kernelT* params_ = nullptr;
    kernelT* kahan_ = nullptr;
  public:
    worker(worker_type wt, void* fcn_ptr):
      wt_(wt),
      fcn_ptr_(fcn_ptr),
      thread_(std::thread([this] {
	while (!exiting_)
	  {
	    std::unique_lock<std::mutex> lock(locking_mutex_);
	    cond_var_.wait(lock, [this] {
	      return has_work_ || exiting_;
	    });
	    if (exiting_)
	      break;
	    //work here
	    switch (wt_)
	      {
	      case worker_type::generate:
		{
		  void (*mykernel)(int, int, int, int, int, int, uint32_t*, kernelT*); 
		  mykernel = (void (*)(int, int, int, int, int, int, uint32_t* ,kernelT*))(fcn_ptr_);
		  mykernel(nevents_, nevents_padded_, from_, to_, thread_idx_, nthreads_, seed_, output_);
		  break;
		}
	      case worker_type::compute:
		{
		  void (*mykernel)(int, int, int, int, kernelT*, kernelT*); 
		  mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*))(fcn_ptr_);
		  mykernel(nevents_, nevents_padded_, from_, to_, input_, output_);
		  break;
		}
	      case worker_type::compute_param:
		{
		  void (*mykernel)(int, int, int, int, kernelT*, kernelT*, kernelT*); 
		  mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*, kernelT*))(fcn_ptr_);
		  mykernel(nevents_, nevents_padded_, from_, to_, input_, output_, params_);
		  break;
		}
	      case worker_type::compute_kahan:
		{
		  void (*mykernel)(int, int, int, int, kernelT*, kernelT*, evalT*); 
		  mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*, evalT*))(fcn_ptr_);
		  mykernel(nevents_, nevents_padded_, from_, to_, input_, output_, kahan_);
		  break;
		}
	      case worker_type::compute_param_kahan:
		{
		  void (*mykernel)(int, int, int, int, kernelT*, kernelT*, kernelT*, evalT*); 
		  mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*, kernelT*, evalT*))(fcn_ptr_);
		  mykernel(nevents_, nevents_padded_, from_, to_, input_, output_, params_, kahan_);
		  break;
		}
	      }
	    //end work here
	    has_work_ = false;
	    lock.unlock();
	    cond_var_.notify_all();
	  }
      }))
    {      
    }
    inline ~worker()
    {
      stop_thread();
    }
    inline void stop_thread()
    {
      std::unique_lock<std::mutex> lock(locking_mutex_);
      has_work_ = false;
      exiting_ = true;
      lock.unlock();
      cond_var_.notify_all();
      if (thread_.joinable())
	thread_.join();
    }
    inline void work()
    {
      std::unique_lock<std::mutex> lock(locking_mutex_);
      has_work_ = true;
      //could set arguments here      
      //how to propagate function signature?
      lock.unlock();
      cond_var_.notify_one();
    }
    inline void finish()
    {
      std::unique_lock<std::mutex> lock(locking_mutex_);
      cond_var_.wait(lock, [this] {
	return has_work_ == false;
      });
    }
    void set_args_generate(unsigned int nevents, unsigned int nevents_padded, unsigned int from, unsigned int to, int thread_idx, int nthreads, seedT* seed, kernelT* output)
    {
      nevents_ = nevents;
      nevents_padded_ = nevents_padded;
      from_ = from;
      to_ = to;
      thread_idx_ = thread_idx;
      nthreads_ = nthreads;
      seed_ = seed;
      output_ = output;
    } 
    void set_args_compute(unsigned int nevents, unsigned int nevents_padded, unsigned int from, unsigned int to, kernelT* input, kernelT* output)
    {
      nevents_ = nevents;
      nevents_padded_ = nevents_padded;
      from_ = from;
      to_ = to;
      input_ = input;
      output_ = output;
    }
    void set_args_compute_params(unsigned int nevents, unsigned int nevents_padded, unsigned int from, unsigned int to, kernelT* input, kernelT* output, kernelT* params)
    {
      nevents_ = nevents;
      nevents_padded_ = nevents_padded;
      from_ = from;
      to_ = to;
      input_ = input;
      output_ = output;
      params_ = params;
    }
    void set_args_compute_kahan(unsigned int nevents, unsigned int nevents_padded, unsigned int from, unsigned int to, kernelT* input, kernelT* output, kernelT* kahan)
    {
      nevents_ = nevents;
      nevents_padded_ = nevents_padded;
      from_ = from;
      to_ = to;
      input_ = input;
      output_ = output;
      kahan_ = kahan;
    }
    void set_args_compute_params_kahan(unsigned int nevents, unsigned int nevents_padded, unsigned int from, unsigned int to, kernelT* input, kernelT* output, kernelT* params, kernelT* kahan)
    {
      nevents_ = nevents;
      nevents_padded_ = nevents_padded;
      from_ = from;
      to_ = to;
      input_ = input;
      output_ = output;
      params_ = params;
      kahan_ = kahan;
    }
  };
  

  template<typename kernelT, typename evalT, typename seedT=uint32_t> 
  class LLVMBlock: public ComputeBlock<kernelT, evalT, seedT> {
  private:
    LLVMBackend* backend_;
    std::string name_;
    unsigned int nevents_;
    unsigned int nevents_padded_;
    std::vector<dimension<evalT>> input_signature_;
    std::vector<dimension<evalT>> output_signature_;
    //pointers
    void* fcn_ptr_;
    kernelT* input_;
    seedT* seed_;
    kernelT* output_;
    evalT* kahan_;
    kernelT* params_;
    //owns the above pointers
    bool holds_input_;
    bool holds_seed_;
    bool holds_output_;
    bool holds_kahan_;
    bool holds_params_;
    //kernel information
    bool kahan_summation_;
    bool generate_kernel_;
    std::unique_ptr<jit::Jit> jit_ptr_;
    //worker thread pool
    std::vector<std::unique_ptr<worker<kernelT, evalT, seedT>>> workers;
  public:
    LLVMBlock(LLVMBackend* backend):
      backend_(backend),
      name_("kernel"),
      nevents_(0),
      nevents_padded_(0),
      fcn_ptr_(NULL),
      input_(NULL),
      seed_(NULL),
      output_(NULL),
      kahan_(NULL),
      params_(NULL),
      holds_input_(false),
      holds_seed_(false),
      holds_output_(false),
      holds_kahan_(false),
      holds_params_(false),
      kahan_summation_(false),
      generate_kernel_(false)
    {
    }
    ~LLVMBlock()
    {      
      if (input_ && holds_input_)
	delete[] input_;
      if (seed_ && holds_seed_)
	delete[] seed_;
      if (output_ && holds_output_)
	delete[] output_;
      if (kahan_ && holds_kahan_)
	delete[] kahan_;
      if (params_ && holds_params_)
	delete[] params_;
    }
    virtual bool SetupInputBuffer(unsigned long int nbytes) override
    {
      if (input_ && holds_input_)
	delete[] input_;
      input_ = new kernelT[nbytes/sizeof(kernelT)];
      holds_input_ = true;
      return true;
    }
    virtual bool SetupSeedBuffer(unsigned long int nbytes) override
    {
      if (seed_ && holds_seed_)
	delete[] seed_;
      seed_ = new seedT[nbytes/sizeof(seedT)];
      holds_seed_ = true;
      return true;
    }
    virtual bool SetupInputBuffer(ComputeBlock<kernelT, evalT>* input, bool use_data_in=false) override
    {
      LLVMBlock<kernelT, evalT>* inputblock = dynamic_cast<LLVMBlock<kernelT, evalT>*>(input);      
      if (input_ && holds_input_)
	delete [] input_; 
      if (use_data_in)
	input_ = inputblock->input_;
      else
	input_ = inputblock->output_;
      holds_input_ = false;
      return true;
    }
    virtual bool SetupParameterBuffer(unsigned long int nbytes) override
    {
      if (params_ && holds_params_)
	delete[] params_;
      params_ = new kernelT[nbytes/sizeof(kernelT)];
      holds_params_ = true;
      return true;
    }
    virtual bool SetupOutputBuffer(unsigned long int nbytes) override
    {
      if (output_ && holds_output_)
	delete[] output_;
      output_ = new kernelT[nbytes/sizeof(kernelT)];
      holds_output_ = true;
      return true;
    }
    virtual bool SetupKahanBuffer(unsigned long int nbytes) override
    {
      if (kahan_ && holds_kahan_)
	delete[] kahan_;
      kahan_ = new evalT[nbytes/sizeof(evalT)];
      holds_kahan_ = true;
      return true;
    }
    virtual bool SetNevents(int nevents, int nevents_padded)
    {
      nevents_ = nevents;
      if (nevents_padded<0)
	nevents_padded_ = nevents;
      else
	nevents_padded_ = nevents_padded;
      return true;
    }
    virtual bool CopyToInputBuffer(const EventVector<kernelT, evalT>& data) override
    {
      //TODO does not need to actually copy as the thread has access to host memory
      std::memcpy(input_, data.get_data(), data.buffer_size());
      return true;
    }
    virtual bool CopyToSeedBuffer(const EventVector<seedT, evalT>& data) override
    {
      std::memcpy(seed_, data.get_data(), data.buffer_size());
      return true;
    }
    virtual bool CopyToParameterBuffer(const std::vector<kernelT>& params) override
    {
      std::memcpy(params_, &params[0], params.size()*sizeof(kernelT));
      return true;
    }
    virtual bool CopyFromOutputBuffer(EventVector<kernelT, evalT>& data) override
    {
      std::memcpy(data.get_data(), output_, data.buffer_size());      
      return true;
    }
    virtual bool CopyFromKahanBuffer(std::vector<evalT>& kahan_sums) override
    {
      std::memcpy(&kahan_sums[0], kahan_, kahan_sums.size()*sizeof(evalT));      
      return true;
    }
    //compilation of generation kernel with specific input and output signature
    virtual bool MakeGenerateKernel(std::string name, unsigned int nevents, std::vector<dimension<evalT>> input_signature,  std::vector<dimension<evalT>> output_signature,
				    const std::vector<std::string>& params, const std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& graphs, evalT maxprob=1.0) override
    {
      name_ = name;
      nevents_ = nevents;
      nevents_padded_ = nevents; //no padding necessary for CPU
      input_signature_ = input_signature;
      output_signature_ = output_signature;
      generate_kernel_ = true; //let the SubmitKernel method know we are submitting a generate kernel, as we need to use the proper signature
      kahan_summation_ = false;      
      
      //safeties
      assert(input_signature.size() == 4);
      assert(graphs.size() == 1);

      //write kernel code
      std::string kernelT_str = graphs.at(0)->kernelT_str();
      std::string evalT_str = graphs.at(0)->evalT_str();
      std::string kernel_code;
      kernel_code += "#include <math.h>\n";
      kernel_code += "#include <stdint.h>\n";
      //prng code
      if constexpr (std::is_same_v<uint32_t, seedT>)
	{
	  //xoshiro128++
	  kernel_code += "inline uint32_t rol32(uint32_t x, int k)\n";
	  kernel_code += "{\n";
	  kernel_code += "  return (x << k) | (x >> (32 - k));\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";      
	  kernel_code += kernelT_str+" xoshiro(uint32_t* state)\n";
	  kernel_code += "{\n";      
	  kernel_code += "  uint32_t const result = rol32(state[0] + state[3], 7) + state[0];\n";
	  //kernel_code += "  uint32_t const result = state[0] + state[3];\n";//xoshiro128+
	  kernel_code += "  uint32_t const t = state[1] << 9;\n";
	  kernel_code += "  state[2] ^= state[0];\n";
	  kernel_code += "  state[3] ^= state[1];\n";
	  kernel_code += "  state[1] ^= state[2];\n";
	  kernel_code += "  state[0] ^= state[3];\n";
	  kernel_code += "  state[2] ^= t;\n";
	  kernel_code += "  state[3] = rol32(state[3], 11);\n";
	  kernel_code += "  return result * 2.3283064365386963e-10;\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";
	  kernel_code += "void "+name_+"(const unsigned int nevents, const unsigned int nevents_padded, const unsigned int from, const unsigned int to, const int threadidx, const int nthreads, const uint32_t * data, "+kernelT_str+"* output)\n";
	}
      if constexpr (std::is_same_v<uint64_t, seedT>)
	{
	  //xoshiro256++
	  kernel_code += "inline uint64_t rol64(uint64_t x, int k)\n";
	  kernel_code += "{\n";
	  kernel_code += "  return (x << k) | (x >> (64 - k));\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";      
	  kernel_code += kernelT_str+" xoshiro(uint64_t* state)\n";
	  kernel_code += "{\n";      
	  kernel_code += "  uint64_t const result = rol64(state[0] + state[3], 23) + state[0];\n";
	  kernel_code += "  uint64_t const t = state[1] << 17;\n";
	  kernel_code += "  state[2] ^= state[0];\n";
	  kernel_code += "  state[3] ^= state[1];\n";
	  kernel_code += "  state[1] ^= state[2];\n";
	  kernel_code += "  state[0] ^= state[3];\n";
	  kernel_code += "  state[2] ^= t;\n";
	  kernel_code += "  state[3] = rol64(state[3], 45);\n";
	  kernel_code += "  return (result >> 11) * 0x1.0p-53;\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";
	  kernel_code += "void "+name_+"(const unsigned int nevents, const unsigned int nevents_padded, const unsigned int from, const unsigned int to, const int threadidx, const int nthreads, const uint64_t * data, "+kernelT_str+"* output)\n";
	}
      
      kernel_code += "{\n";
      //initialise prng
      if constexpr (std::is_same_v<uint32_t, seedT>)
	kernel_code += "  uint32_t rnd_state[4];\n";
      if constexpr (std::is_same_v<uint64_t, seedT>)
	kernel_code += "  uint64_t rnd_state[4];\n";
      kernel_code += "  rnd_state[0] = data[threadidx+0*nthreads];\n";
      kernel_code += "  rnd_state[1] = data[threadidx+1*nthreads];\n";
      kernel_code += "  rnd_state[2] = data[threadidx+2*nthreads];\n";
      kernel_code += "  rnd_state[3] = data[threadidx+3*nthreads];\n";
      //adding parameter definitions
      for (unsigned int i=0; i<params.size(); i++)
	{
	  kernel_code += "const " + kernelT_str + " ";
	  kernel_code += params.at(i);
	  kernel_code += " = parameters[";
	  kernel_code += std::to_string(i);
	  kernel_code += "];\n";
	}

      //set max probability
      std::ostringstream maxprob_str;
      maxprob_str.precision(15);
      maxprob_str << std::scientific << maxprob;
      kernel_code += "  const "+kernelT_str+" maxprob = " + maxprob_str.str() + ";\n";
      kernel_code += "\n";
      kernel_code += "  "+kernelT_str+" prob;\n";
      for (unsigned int i=0; i<output_signature.size(); i++)
	kernel_code += "  "+kernelT_str + " " + output_signature.at(i).get_name()+";\n";
      
      //adding parameter definitions, should not actually matter, maybe should contain maxprob?
      for (unsigned int i=0; i<params.size(); i++)
	{
	  kernel_code += "  const "+kernelT_str+" ";
	  kernel_code += params.at(i);
	  kernel_code += " = parameters[";
	  kernel_code += std::to_string(i);
	  kernel_code += "];\n";
	}      
      //loop over events from defined start to finish
      kernel_code += "for (unsigned int i=from; i<to; i++)\n";
      kernel_code += "{\n";
      kernel_code += "  _Bool finished = 0;\n";
      kernel_code += "  do {\n";
      //throw random numbers
      for (unsigned int i=0; i<output_signature.size(); i++)
	{
	  std::ostringstream min;
	  min.precision(15);
	  min << std::scientific << output_signature.at(i).get_min();
	  std::ostringstream delta;
	  delta.precision(15);
	  delta << std::scientific << (output_signature.at(i).get_max()-output_signature.at(i).get_min());
	  kernel_code += "    "+output_signature.at(i).get_name() + " = " + min.str() + " + " + delta.str() + "*xoshiro(rnd_state);\n";
	}
      //evaluate pdf
      kernel_code += "    prob = " + graphs.at(0)->get_kernel() + ";\n"; 
      //accept/reject
      kernel_code += "    if (maxprob*(xoshiro(rnd_state)) < prob)\n";
      kernel_code += "    {\n";
      //save accepted event to output
      for (unsigned int i=0; i<output_signature.size(); i++)
	{
	  kernel_code += "      output[" + (i==0 ? std::string("i") : ("nevents_padded*" + std::to_string(i) + std::string("+i"))) + "] = ";
	  kernel_code +=  output_signature.at(i).get_name();
	  kernel_code += ";\n";	  
	}
      kernel_code += "      finished = 1;\n";
      kernel_code += "    }\n";      
      kernel_code += "} while (!finished);\n";
      kernel_code += "}\n";
      kernel_code += "}\n";
      //print kernel if wanted
      if (backend_->opts_->print_kernel)
	std::cout << std::endl << "GENERATION KERNEL SOURCECODE:" << std::endl << kernel_code << std::endl << "END KERNEL SOURCECODE" << std::endl;
      auto t_before_compilation = std::chrono::high_resolution_clock::now();      
      auto R = cc::CCompiler().compile(kernel_code.c_str(), backend_->opts_);
      //abort if compilation failed.
      auto [C, M] = cantFail(std::move(R));      
      auto t_after_compilation = std::chrono::high_resolution_clock::now();     
      std::cout << "compiling generate kernel takes " << std::chrono::duration<double, std::milli>(t_after_compilation-t_before_compilation).count() << " ms in total" << std::endl;
      if (backend_->opts_->llvm_print_intermediate)
	{
	  std::cout << std::endl << "LLVM INTERMEDIATE REPRESENTATION:" << std::endl;	  	  
	  M->print(llvm::errs(), nullptr);
	  std::cout << "END LLVM INTERMEDIATE REPRESENTATION" << std::endl;
	}
      //JIT compile the IR module.
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      
      jit_ptr_ = std::move(jit::Jit::Create(name_, backend_->opts_));
      auto TSM = llvm::orc::ThreadSafeModule(std::move(M), std::move(C));
      auto RT = jit_ptr_->addModule(std::move(TSM));
      if (auto E = RT.takeError()) {
	llvm::errs() << llvm::toString(std::move(E)) << '\n';
	return 1;
      }
      if (auto ADDR = jit_ptr_->lookup(name_.c_str()))
	fcn_ptr_ = (void*)(*ADDR).getAddress().toPtr<void(*)()>();
      return true;
    }
    //compilation of compute kernel with specific input and output signature
    virtual bool MakeComputeKernel(std::string name, unsigned int nevents, std::vector<dimension<evalT>> input_signature,  std::vector<dimension<evalT>> output_signature,
				   const std::vector<std::string>& params, const std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& graphs, bool kahan_summation=false) override
    {

      kahan_summation_ = kahan_summation;
      name_ = name;
      nevents_ = nevents;
      nevents_padded_ = nevents;

      input_signature_ = input_signature;
      output_signature_ = output_signature;
      assert(output_signature.size() == graphs.size());
      
      //write kernel code
      std::string kernelT_str = graphs.at(0)->kernelT_str();
      std::string evalT_str = graphs.at(0)->evalT_str();
      const bool vectorization = backend_->opts_->llvm_vectorization;
      const unsigned int vectorization_width = backend_->opts_->llvm_vectorization_width;
      std::string kernel_code;
      kernel_code += "#include <math.h>\n";
      kernel_code += "\n";
      if (kahan_summation && vectorization)
	{
	  //helper function to allow for vectorized kahan summation
	  kernel_code += "inline void kahan_sum(const unsigned int width, const unsigned int start_idx, "+kernelT_str+"* sum, "+kernelT_str+"* carry, const unsigned int i, const "+kernelT_str+" value)\n";
	  kernel_code += "{\n";
	  kernel_code += "#pragma clang fp reassociate(off)\n"; //do not optimize away kahan summation (ie. use associativity to set carry to 0)
	  kernel_code += "  const unsigned int idx = i % width + start_idx;\n";
	  kernel_code += "  const "+kernelT_str+" y = value - carry[idx];\n";
	  kernel_code += "  const "+kernelT_str+" t = sum[idx] + y;\n";
	  kernel_code += "  carry[idx] = (t - sum[idx]) - y;\n";
	  kernel_code += "  sum[idx] = t;\n";
	  kernel_code += "}\n\n";
	}

      if (kahan_summation)
	{
	  if (params.size() > 0)
	    kernel_code += ("void "+name_+"(const unsigned int nevents, const unsigned int nevents_padded, const unsigned int from, const unsigned int to, const "+kernelT_str+"* data, "+kernelT_str+"* output, const "+kernelT_str+"* parameters, "+evalT_str+"* kahan)\n");
	  else
	    kernel_code += ("void "+name_+"(const unsigned int nevents, const unsigned int nevents_padded, const unsigned int from, const unsigned int to, const "+kernelT_str+"* data, "+kernelT_str+"* output, "+evalT_str+" *kahan)\n");
	}
      else
	{
	  if (params.size() > 0)
	    kernel_code += ("void "+name_+"(const unsigned int nevents, const unsigned int nevents_padded, const unsigned int from, const unsigned int to, const "+kernelT_str+"* data, "+kernelT_str+"* output, const "+kernelT_str+"* parameters)\n");
	  else
	    kernel_code += ("void "+name_+"(const unsigned int nevents, const unsigned int nevents_padded, const unsigned int from, const unsigned int to, const "+kernelT_str+"* data, "+kernelT_str+"* output)\n");
	}
      kernel_code += "{\n";
      //online kahan summation
      if (kahan_summation)
	{
	  if (vectorization)
	    {
	      kernel_code += evalT_str+" morefit_sum["+std::to_string(graphs.size()*vectorization_width)+"];\n";
	      kernel_code += evalT_str+" morefit_c["+std::to_string(graphs.size()*vectorization_width)+"];\n";
	      kernel_code += "for (unsigned int i=0;i<"+std::to_string(graphs.size()*vectorization_width)+";i++)\n";
	      kernel_code += "  {\n";
	      kernel_code += "  morefit_sum[i]=0.0;\n";
	      kernel_code += "  morefit_c[i]=0.0;\n";
	      kernel_code += "  }\n";
	    }
	  else
	    {
	      kernel_code += evalT_str+" morefit_sum["+std::to_string(graphs.size())+"];\n";
	      kernel_code += evalT_str+" morefit_c["+std::to_string(graphs.size())+"];\n";
	      kernel_code += evalT_str+" morefit_y["+std::to_string(graphs.size())+"];\n";
	      kernel_code += evalT_str+" morefit_t["+std::to_string(graphs.size())+"];\n";
	      
	      kernel_code += "for (unsigned int i=0;i<"+std::to_string(graphs.size())+";i++)\n";
	      kernel_code += "  {\n";
	      kernel_code += "  morefit_sum[i]=0.0;\n";
	      kernel_code += "  morefit_c[i]=0.0;\n";
	      kernel_code += "  }\n";
	    }
	}      
      //adding parameter definitions
      for (unsigned int i=0; i<params.size(); i++)
	{
	  kernel_code += "const " + kernelT_str+" ";
	  kernel_code += params.at(i);
	  kernel_code += " = parameters[";
	  kernel_code += std::to_string(i);
	  kernel_code += "];\n";
	}
      
      if (vectorization)//explicitly ask for vectorization of event loop
	{
	  kernel_code += "#pragma clang loop vectorize(enable)\n";//ask for vectorization
	  kernel_code += "#pragma clang loop vectorize_width("+std::to_string(vectorization_width)+")\n";
	}
      
      kernel_code += "for (unsigned int i=from; i<to; i++)\n";
      kernel_code += "{\n";      
      if (kahan_summation && !vectorization)
	kernel_code += "#pragma clang fp reassociate(off)\n";//do not optimize away kahan summation

      //set up kernel inputs
      for (unsigned int i=0; i<input_signature.size(); i++)
	{
	  kernel_code += "  const " + kernelT_str + " ";
	  kernel_code += input_signature.at(i).get_name();
	  kernel_code += " = data[" + (i==0 ? std::string("i") : ("nevents*" + std::to_string(i) + "+i")) + "];\n";  //TODO I guess we do not need padding as argument for llvm, any issue with vectorization?
	}

      //actual kernel calculating the prob
      for (unsigned int i=0; i<graphs.size(); i++)
	{
	  if (kahan_summation)
	    {
	      if (vectorization && graphs.size() == 1) //for one graph it is more efficient to store result in the output and then do a second vectorised loop for kahan summation
		kernel_code += "  output[" + (i==0 ? std::string("i") : ("nevents*" + std::to_string(i) + "+i")) + "] = ";
	      else //for multiple graphs (or not vectorized) use local variable
		kernel_code += "  const " + kernelT_str + " result_" + std::to_string(i) + " = ";
	    }
	  else
	    kernel_code += "  output[" + (i==0 ? std::string("i") : ("nevents*" + std::to_string(i) + "+i")) + "] = ";
	  kernel_code +=  graphs.at(i)->get_kernel(); 
	  kernel_code += ";\n";
	}
      for (unsigned int i=0; i<graphs.size(); i++)
	{
	  if (kahan_summation)
	    {
	      //online kahan summation
	      if (vectorization && graphs.size() > 1) //for multiple graphs perform vectorized kahan summation in place
		kernel_code += "  kahan_sum("+std::to_string(vectorization_width)+", "+std::to_string(i*vectorization_width)+", morefit_sum, morefit_c, i, result_"+ std::to_string(i) + ");\n";		      
	      else if (!vectorization) //not vectorized kahan summation
		{
		  std::string  graph_idx = std::to_string(i);
		  kernel_code +=  "  morefit_y[" + graph_idx + "] = result_" + graph_idx + " - morefit_c[" + graph_idx + "];\n";
		  kernel_code +=  "  morefit_t[" + graph_idx + "] = morefit_sum[" + graph_idx + "] + morefit_y[" + graph_idx + "];\n";
		  kernel_code +=  "  morefit_c[" + graph_idx + "] = (morefit_t[" + graph_idx + "] - morefit_sum[" + graph_idx + "]) - morefit_y[" + graph_idx + "];\n";
		  kernel_code +=  "  morefit_sum[" + graph_idx + "] = morefit_t[" + graph_idx + "];\n";
		}
	    }	  
	}      
      kernel_code += "}\n";
      if (kahan_summation)
	{
	  if (vectorization && graphs.size() == 1) //for a single graph a separate loop for kahan summation is found more efficient
	    {
	      kernel_code += kernelT_str+" kahan_carry["+std::to_string(graphs.size())+"];\n";
	      kernel_code += "for (unsigned int i=0;i<"+std::to_string(graphs.size())+";i++)\n";
	      kernel_code += "  kahan_carry[i] = 0.0;\n";
	      kernel_code += "for (unsigned int j=0; j<"+std::to_string(graphs.size())+"; j++)\n";
	      kernel_code += "{\n";
	      kernel_code += "#pragma clang loop vectorize(enable)\n";
	      kernel_code += "#pragma clang loop vectorize_width(4)\n";
	      kernel_code += "  for (unsigned int i=from; i<to; i++)\n";
	      kernel_code += "    kahan_sum("+std::to_string(vectorization_width)+", j*"+std::to_string(vectorization_width)+", morefit_sum, morefit_c, i, output[nevents*j+i]);\n";	      
	      kernel_code += "  kahan[j] = 0.0;\n";
	      for (unsigned int j=0; j<vectorization_width; j++)
		kernel_code += "  kahan_sum("+std::to_string(graphs.size())+", 0, kahan, kahan_carry, j, morefit_sum[j*"+std::to_string(vectorization_width) + "+" + std::to_string(j)+"]);\n";	   
	      kernel_code += "}\n";
	    }	  
	  else if (vectorization && graphs.size() > 1)//for multiple graphs we only need to do final summation over the vectorization width results
	    {
	      kernel_code += kernelT_str+" kahan_carry["+std::to_string(graphs.size())+"];\n";
	      kernel_code += "for (unsigned int i=0; i<"+std::to_string(graphs.size())+"; i++)\n";
	      kernel_code += "  kahan_carry[i] = 0.0;\n";
	      kernel_code += "for (unsigned int i=0; i<" + std::to_string(graphs.size()) + "; i++)\n";
	      kernel_code += "{\n";
	      kernel_code += "  kahan[i] = 0.0;\n";
	      for (unsigned int j=0; j<vectorization_width; j++)
		kernel_code += "  kahan_sum("+std::to_string(graphs.size())+", 0, kahan, kahan_carry, i, morefit_sum[i*"+std::to_string(vectorization_width) + "+" + std::to_string(j)+"]);\n";	      
	      kernel_code += "}\n";
	    }	  
	  else
	    {
	      for (unsigned int i=0; i<graphs.size(); i++)
		kernel_code += "kahan["+std::to_string(i)+"] = morefit_sum["+std::to_string(i)+"];\n";
	    }
	}
      kernel_code += "}\n";
      //print kernel if wanted
      if (backend_->opts_->print_kernel)
	std::cout << std::endl << "COMPUTE KERNEL SOURCECODE:" << std::endl << kernel_code << std::endl << "END KERNEL SOURCECODE" << std::endl;


      auto t_before_compilation = std::chrono::high_resolution_clock::now();      
      auto R = cc::CCompiler().compile(kernel_code.c_str(), backend_->opts_);
      //abort if compilation failed.
      auto [C, M] = cantFail(std::move(R));
      auto t_after_compilation = std::chrono::high_resolution_clock::now();     
      std::cout << "compiling compute kernel takes " << std::chrono::duration<double, std::milli>(t_after_compilation-t_before_compilation).count() << " ms in total" << std::endl;

      if (backend_->opts_->llvm_print_intermediate)
	{
	  std::cout << std::endl << "LLVM INTERMEDIATE REPRESENTATION:" << std::endl;	  	  
	  M->print(llvm::errs(), nullptr);
	  std::cout << "END LLVM INTERMEDIATE REPRESENTATION" << std::endl;
	}
      
      //JIT compile the IR module.
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();

      jit_ptr_ = std::move(jit::Jit::Create(name_, backend_->opts_));            
      auto TSM = llvm::orc::ThreadSafeModule(std::move(M), std::move(C));
      auto RT = jit_ptr_->addModule(std::move(TSM));
      if (auto E = RT.takeError()) {
	llvm::errs() << llvm::toString(std::move(E)) << '\n';
	return 1;
      }
      auto t_before_making = std::chrono::high_resolution_clock::now();     

      if (auto ADDR = jit_ptr_->lookup(name_.c_str()))
	fcn_ptr_ = (void*)(*ADDR).getAddress().toPtr<void(*)()>();
      auto t_after_making = std::chrono::high_resolution_clock::now();     

      std::cout << "making lookup takes " << std::chrono::duration<double, std::milli>(t_after_making-t_before_making).count() << " ms in total" << std::endl;
      
      return true;
    }
    virtual bool SubmitKernel()
    {
      if (generate_kernel_) //submit generate kernel
	{
	  if (backend_->opts_->llvm_nthreads > 1) //only start separate threads for nthread>1 to avoid overhead
	    {	  
	      if (workers.size() == 0) //create threads on first call
		{
		  for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)			  
		    workers.emplace_back(std::make_unique<worker<kernelT, evalT, seedT>>(worker<kernelT, evalT, seedT>::worker_type::generate, fcn_ptr_));
		}
	      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
		{
		  int size = nevents_/backend_->opts_->llvm_nthreads;
		  int from = i*size;
		  int to = i*size + size;
		  if (i==backend_->opts_->llvm_nthreads-1)
		    to = nevents_;
		  workers.at(i)->set_args_generate(nevents_, nevents_padded_, from, to, i, backend_->opts_->llvm_nthreads, seed_, output_);
		  workers.at(i)->work();
		}
	      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
		workers.at(i)->finish();		      
	    }
	  else //single-threaded
	    {
	      void (*mykernel)(int, int, int, int, int, int, uint32_t*, kernelT*); 
	      mykernel = (void (*)(int, int, int, int, int, int, uint32_t* ,kernelT*))(fcn_ptr_);
	      mykernel(nevents_, nevents_padded_, 0, nevents_, 0, backend_->opts_->llvm_nthreads, seed_, output_);
	    }
	}
      else //submit standard compute kernel
	{
	  //running method, signature depends on kernel options
	  if (backend_->opts_->llvm_nthreads > 1)//only start separate threads for nthread>1 to avoid overhead
	    {
	      assert(nevents_ >= backend_->opts_->llvm_nthreads);
	      if (kahan_summation_)
		{
		  unsigned int kahan_size = output_signature_.size();		  
		  std::vector<std::vector<evalT>> kahan_sums(backend_->opts_->llvm_nthreads, std::vector<evalT>(kahan_size,0.0));
		  if (params_)
		    {
		      if (workers.size() == 0)//create threads on first call
			{
			  for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)			  
			    workers.emplace_back(std::make_unique<worker<kernelT, evalT, seedT>>(worker<kernelT, evalT, seedT>::worker_type::compute_param_kahan, fcn_ptr_));
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			{
			  int size = nevents_/backend_->opts_->llvm_nthreads;
			  int from = i*size;
			  int to = i*size + size;
			  if (i==backend_->opts_->llvm_nthreads-1)
			    to = nevents_;
			  workers.at(i)->set_args_compute_params_kahan(nevents_, nevents_padded_, from, to, input_, output_, params_, &kahan_sums.at(i)[0]);
			  workers.at(i)->work();
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			workers.at(i)->finish();		      
		      //add up kahan sums
		      for (unsigned int i=0; i<kahan_size; i++)
			{
			  //doubtful that kahan summation here is necessary
			  std::vector<evalT> results;
			  results.reserve(backend_->opts_->llvm_nthreads);
			  for (unsigned int j=0; j<backend_->opts_->llvm_nthreads; j++)
			    results.push_back(kahan_sums.at(j).at(i));
			  kahan_[i] = kahan_summation<evalT,evalT>(results);
			}
		    }
		  else
		    {
		      if (workers.size() == 0)//create threads on first call
			{
			  for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)			  
			    workers.emplace_back(std::make_unique<worker<kernelT, evalT, seedT>>(worker<kernelT, evalT, seedT>::worker_type::compute_kahan, fcn_ptr_));
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			{
			  int size = nevents_/backend_->opts_->llvm_nthreads;
			  int from = i*size;
			  int to = i*size + size;
			  if (i==backend_->opts_->llvm_nthreads-1)
			    to = nevents_;
			  workers.at(i)->set_args_compute_kahan(nevents_, nevents_padded_, from, to, input_, output_, &kahan_sums.at(i)[0]);
			  workers.at(i)->work();
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			workers.at(i)->finish();
		      //add up kahan sums
		      for (unsigned int i=0; i<kahan_size; i++)
			{
			  std::vector<evalT> results;
			  results.reserve(backend_->opts_->llvm_nthreads);
			  for (unsigned int j=0; j<backend_->opts_->llvm_nthreads; j++)
			    results.push_back(kahan_sums.at(j).at(i));
			  kahan_[i] = kahan_summation<evalT,evalT>(results);
			}
		    }
		}
	      else
		{
		  if (params_)
		    {
		      if (workers.size() == 0)//create threads on first call
			{
			  for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)			  
			    workers.emplace_back(std::make_unique<worker<kernelT, evalT, seedT>>(worker<kernelT, evalT, seedT>::worker_type::compute_param, fcn_ptr_));
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			{
			  int size = nevents_/backend_->opts_->llvm_nthreads;
			  int from = i*size;
			  int to = i*size + size;
			  if (i==backend_->opts_->llvm_nthreads-1)
			    to = nevents_;
			  workers.at(i)->set_args_compute_params(nevents_, nevents_padded_, from, to, input_, output_, params_);
			  workers.at(i)->work();
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			workers.at(i)->finish();
		    }
		  else
		    {
		      if (workers.size() == 0)//create threads on first call
			{
			  for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			    workers.emplace_back(std::make_unique<worker<kernelT, evalT, seedT>>(worker<kernelT, evalT, seedT>::worker_type::compute, fcn_ptr_));
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			{
			  int size = nevents_/backend_->opts_->llvm_nthreads;
			  int from = i*size;
			  int to = i*size + size;
			  if (i==backend_->opts_->llvm_nthreads-1)
			    to = nevents_;
			  workers.at(i)->set_args_compute(nevents_, nevents_padded_, from, to, input_, output_);
			  workers.at(i)->work();
			}
		      for (unsigned int i=0; i<backend_->opts_->llvm_nthreads; i++)
			workers.at(i)->finish();
		    }
		}
	    }
	  else //single thread direct execution
	    {
	      assert(!generate_kernel_);
	      if (kahan_summation_)
		{
		  if (params_)
		    {
		      void (*mykernel)(int, int, int, int, kernelT*, kernelT*, kernelT*, evalT*); 
		      mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*, kernelT*, evalT*))(fcn_ptr_);
		      mykernel(nevents_, nevents_padded_, 0, nevents_, input_, output_, params_, kahan_);
		    }
		  else
		    {
		      void (*mykernel)(int, int, int, int, kernelT*, kernelT*, evalT*); 
		      mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*, evalT*))(fcn_ptr_);
		      mykernel(nevents_, nevents_padded_, 0, nevents_, input_, output_, kahan_);
		    }
		}
	      else
		{
		  if (params_)
		    {
		      void (*mykernel)(int, int, int, int, kernelT*, kernelT*, kernelT*); 
		      mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*, kernelT*))(fcn_ptr_);
		      mykernel(nevents_, nevents_padded_, 0, nevents_, input_, output_, params_);
		    }
		  else
		    {
		      void (*mykernel)(int, int, int, int, kernelT*, kernelT*); 
		      mykernel = (void (*)(int, int, int, int, kernelT*,kernelT*))(fcn_ptr_);
		      mykernel(nevents_, nevents_padded_, 0, nevents_, input_, output_);
		    }
		}
	    }
	}
      return true;      
    }
    virtual bool Finish() override
    {
      return true;
    }
    virtual bool ProvidesKahan() override
    {
      return true;
    }
  };

}

#endif
