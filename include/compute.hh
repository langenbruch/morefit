/**
 * @file compute.hh
 * @author Christoph Langenbruch
 * @date 2024-10-06
 *
 */

#ifndef COMPUTE_H
#define COMPUTE_H

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

#include "eventvector.hh"
#include "parametervector.hh"
#include "graph.hh"


namespace morefit {

  //compute options class
  struct compute_options {
    //general options
    bool print_kernel;
    int print_level;
    //opencl options
    unsigned int opencl_platform;
    unsigned int opencl_device;
    unsigned int opencl_workgroup_size;
    int opencl_max_generator_workitems;
    int opencl_max_compute_workitems;
    unsigned int opencl_kahan_reduction_factor;
    //llvm options
    unsigned int llvm_nthreads;
    bool llvm_print_intermediate;
    bool llvm_vectorization;
    int llvm_vectorization_width;
    std::string llvm_triple;
    std::string llvm_cpu;
    std::string llvm_tunecpu;
    std::string llvm_optimization;
    compute_options():
      //general options
      print_kernel(true),
      print_level(0),
      //opencl options
      opencl_platform(0),
      opencl_device(0),
      opencl_workgroup_size(256),
      opencl_max_generator_workitems(8192),
      opencl_max_compute_workitems(-1),
      opencl_kahan_reduction_factor(32),
      //llvm options
      llvm_nthreads(1),
      llvm_print_intermediate(true),
      llvm_vectorization(true),
      llvm_vectorization_width(4),
      llvm_triple("x86_64-pc-linux-gnu"),
      llvm_cpu("x86-64-v3"),
      llvm_tunecpu("x86-64"),
      llvm_optimization("-O3")
    {}
    void print()
    {
      const unsigned int width = 40;
      std::cout << "COMPUTE OPTIONS" << std::endl;
      std::cout << "General:" << std::endl;
      std::cout << std::setw(width) << std::left << "  print kernel " << (print_kernel ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  print level " << print_level << std::endl;
      std::cout << "OpenCL:" << std::endl;
      std::cout << std::setw(width) << std::left << "  platform id " << opencl_platform << std::endl;
      std::cout << std::setw(width) << std::left << "  device id " << opencl_device << std::endl;
      std::cout << std::setw(width) << std::left << "  work group size " << opencl_workgroup_size << std::endl;
      std::cout << std::setw(width) << std::left << "  max number of generator workitems " << opencl_max_generator_workitems << std::endl;
      std::cout << std::setw(width) << std::left << "  max number of compute workitems " << opencl_max_compute_workitems << std::endl;
      std::cout << std::setw(width) << std::left << "  kahan reduction factor " << opencl_kahan_reduction_factor << std::endl;
      std::cout << "LLVM:" << std::endl;
      std::cout << std::setw(width) << std::left << "  print intermediate representation " << (llvm_print_intermediate ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  number of threads " << llvm_nthreads << std::endl;      
      std::cout << std::setw(width) << std::left << "  vectorization " << (llvm_vectorization ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  vectorization width " << llvm_vectorization_width << std::endl;
      std::cout << std::setw(width) << std::left << "  triple " << llvm_triple << std::endl;
      std::cout << std::setw(width) << std::left << "  cpu " << llvm_cpu << std::endl;
      std::cout << std::setw(width) << std::left << "  tunecpu " << llvm_tunecpu << std::endl;
      std::cout << std::setw(width) << std::left << "  optimization " << llvm_optimization << std::endl;
    }
  };

  //base class for compute backend
  class ComputeBackend {
  public:
    compute_options* opts_;
    //do events require padding? This depends on the backend, eg for opencl events need to be padded to workgroup size
    virtual int required_padding() const {return 0;}
  protected:
    ComputeBackend(compute_options* opts):
      opts_(opts)
    {}
  };

  //abstract baseclass for compute backend (opencl, llvm, etc.)
  template<typename kernelT, typename evalT, typename seedT=uint32_t> 
  class ComputeBlock {
  public:
    //allocate input buffer on accelerator
    virtual bool SetupInputBuffer(unsigned long int nbytes) = 0;
    //allocate buffer for PRNG seeds on accelerator
    virtual bool SetupSeedBuffer(unsigned long int nbytes) = 0;
    //allocate input buffer on accelerator TODO
    virtual bool SetupInputBuffer(ComputeBlock<kernelT, evalT>* input, bool use_data_in=false) = 0;
    //allocate parameter buffer on accelerator
    virtual bool SetupParameterBuffer(unsigned long int nbytes) = 0;
    //allocate output buffer on accelerator
    virtual bool SetupOutputBuffer(unsigned long int nbytes) = 0;
    //allocate kahan buffer on accelerator
    virtual bool SetupKahanBuffer(unsigned long int nbytes) {return false;}
    //(re)setting the number of events
    virtual bool SetNevents(int nevents, int nevents_padded=-1) { return false;}
    //copy data from host to input buffer of accelerator
    virtual bool CopyToInputBuffer(const EventVector<kernelT, evalT>& data) = 0;
    //copy data from host to input buffer of accelerator
    virtual bool CopyToSeedBuffer(const EventVector<seedT, evalT>& data) = 0;
    //copy parameters from host to parameter buffer of accelerator
    virtual bool CopyToParameterBuffer(const std::vector<kernelT>& params) = 0;
    //copy output from accelerator to host
    virtual bool CopyFromOutputBuffer(EventVector<kernelT, evalT>& data) = 0;
    //copy output from kahan summation from accelerator to host (if provided)
    virtual bool CopyFromKahanBuffer(std::vector<evalT>& kahan_sums) {return false;}
    //compilation of generation kernel with specific input and output signature
    virtual bool MakeGenerateKernel(std::string name, unsigned int nevents, std::vector<dimension<evalT>> input_signature,  std::vector<dimension<evalT>> output_signature, 
				    const std::vector<std::string>& params, const std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& graphs, evalT maxprob=1.0) {return true;}
    //compilation of compute kernel with specific input and output signature
    virtual bool MakeComputeKernel(std::string name, unsigned int nevents, std::vector<dimension<evalT>> input_signature,  std::vector<dimension<evalT>> output_signature, 
				   const std::vector<std::string>& params, const std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& graphs, bool kahan_summation=false) = 0;
    //submit kernel
    virtual bool SubmitKernel() = 0;
    //make sure computation is finished
    virtual bool Finish() = 0;
    //state if kahan summation is provided on the accelerator or needs to be done on host
    virtual bool ProvidesKahan() {return false;};
  };
 
}

#endif
