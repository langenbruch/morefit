/**
 * @file compute_opencl.hh
 * @author Christoph Langenbruch
 * @date 2024-11-25
 *
 */

#ifndef COMPUTEOPENCL_H
#define COMPUTEOPENCL_H

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
#include "compute.hh"

namespace morefit {


  class OpenCLBackend: public ComputeBackend {
  private:
    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;
    cl_platform_id platform_;
  public:
    OpenCLBackend(compute_options* opts):
      ComputeBackend(opts)
    {
      cl_int CL_err = CL_SUCCESS;
      cl_uint numPlatforms = 0;
      CL_err = clGetPlatformIDs(0, NULL, &numPlatforms);
      if (CL_err == CL_SUCCESS)
	{
	  bool found_platform_and_device = false;
	  printf("%u platform(s) found\n", numPlatforms);
	  cl_platform_id platforms[numPlatforms];
	  clGetPlatformIDs(numPlatforms, platforms, NULL);
	  for (unsigned int i=0; i<numPlatforms; i++)
	    {
	      // print platform name
	      size_t nameSize = 0;
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &nameSize);
	      char platformname[nameSize];
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, nameSize, platformname, NULL);
	      printf("  platform name: %s\n", platformname);

	      size_t versionSize = 0;
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 0, NULL, &versionSize);
	      char version[versionSize];
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, versionSize, version, NULL);
	      printf("  platform version: %s\n", version);

	      size_t vendorSize = 0;
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &vendorSize);
	      char vendor[vendorSize];
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, vendorSize, vendor, NULL);
	      printf("  platform vendor: %s\n", vendor);

	      size_t extensionsSize = 0;
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &extensionsSize);
	      char extensions[extensionsSize];
	      clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, extensionsSize, extensions, NULL);

	      cl_uint numDevices = 0;
	  
	      CL_err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);	    
	      if (CL_err == CL_SUCCESS)
		{
		  printf("  %u device(s) found\n", numDevices);
		  cl_device_id devices[numDevices];	    
		  clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);	
		  // for each device print critical attributes
		  for (unsigned int j = 0; j < numDevices; j++)
		    {
		      // print device name
		      size_t nameSize = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &nameSize);
		      char name[nameSize];
		      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, nameSize, name, NULL);
		      printf("  device: %s\n", name);
		    
		      // print hardware device version
		      size_t hwVersionSize = 0;		    
		      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &hwVersionSize);
		      char hwVersion[hwVersionSize];
		      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, hwVersionSize, hwVersion, NULL);
		      printf("    hardware version: %s\n", hwVersion);

		      // print software driver version
		      size_t swVersionSize = 0;		    
		      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &swVersionSize);
		      char swVersion[swVersionSize];		    
		      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, swVersionSize, swVersion, NULL);
		      printf("    software version: %s\n", swVersion);

		      //print device extensions
		      size_t extensionsSize = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionsSize);
		      char extensions[extensionsSize];
		      clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, extensionsSize, extensions, NULL);
		      printf("    device extensions: %s\n", extensions);
		    
		      // print parallel compute units
		      cl_uint maxComputeUnits = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
		      printf("    parallel compute units: %d\n", maxComputeUnits);

		      cl_uint maxWorkDimensions = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkDimensions), &maxWorkDimensions, NULL);
		      printf("    max work item dimensions: %d\n", maxWorkDimensions);

		      size_t maxWorkSizes[maxWorkDimensions];
		      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkDimensions*sizeof(size_t), maxWorkSizes, NULL);
		      printf("    max work item sizes: ");
		      for (unsigned int k=0; k<maxWorkDimensions; k++)
			printf("%lu ", maxWorkSizes[k]);
		      printf("\n");

		      size_t maxGroupSize = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxGroupSize), &maxGroupSize, NULL);
		      printf("    max work group size: %lu\n", maxGroupSize);

		      cl_uint maxClockFreq = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFreq), &maxClockFreq, NULL);
		      printf("    max clock frequency: %u\n", maxClockFreq);
		    
		      size_t maxParamSize = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(maxParamSize), &maxParamSize, NULL);
		      printf("    max parameter size: %lu\n", maxParamSize);
		      
		      cl_ulong maxLocalMemSize = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxLocalMemSize), &maxLocalMemSize, NULL);
		      printf("    max local mem size: %lu KB\n", maxLocalMemSize/1000);			

		      cl_ulong maxMemSize = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxMemSize), &maxMemSize, NULL);
		      printf("    max global mem size: %lu MB\n", maxMemSize/1000000);
		      
		      cl_ulong maxMemAllocSize = 0;
		      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
		      printf("    max mem allocation size: %lu MB\n", maxMemAllocSize/1000000);			
		      //get context for first platform first device
		      if (opts->opencl_platform == i && opts->opencl_device == j)//default platform has device
			{
			  device_ = devices[opts->opencl_device];
			  platform_ = platforms[opts->opencl_platform];
			  found_platform_and_device = true;
			}
		      std::cout << std::endl;
		    }//end loop over devices
		}
	      else
		printf("  clGetDeviceIDs(%i)\n", CL_err);	      
	    }
	  if (found_platform_and_device)
	    {
	      context_ = clCreateContext(NULL, 1, &device_, NULL, NULL, NULL);
	      queue_ = clCreateCommandQueueWithProperties(context_, device_, 0, &CL_err);

	      size_t platformSize = 0;
	      clGetPlatformInfo(platform_, CL_PLATFORM_NAME, 0, NULL, &platformSize);
	      char platformname[platformSize];
	      clGetPlatformInfo(platform_, CL_PLATFORM_NAME, platformSize, platformname, NULL);

	      size_t deviceSize = 0;
	      clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, NULL, &deviceSize);
	      char name[deviceSize];
	      clGetDeviceInfo(device_, CL_DEVICE_NAME, deviceSize, name, NULL);

	      if (CL_err == CL_SUCCESS)	
		std::cout << "Selected Device " << name << " from platform " << platformname << " for compute backend" << std::endl;
	      else
		{
		  std::cout << "Backend initialisation failed. Did you select an existing OpenCL platform and device? " << CL_err << std::endl;
		  assert(0);
		}
			  
	    }
	}
      else
	printf("clGetPlatformIDs(%i)\n", CL_err);
    }
    virtual int required_padding() const override
    {
      return opts_->opencl_workgroup_size;
    } 
    cl_context context() {return context_;};
    cl_command_queue queue() {return queue_;};
    cl_device_id device() {return device_;};  
    ~OpenCLBackend()
    {
      clReleaseContext(context_);
      clReleaseCommandQueue(queue_);
    }
  };

  template<typename kernelT, typename evalT, typename seedT=uint32_t> 
  class OpenCLBlock: public ComputeBlock<kernelT, evalT, seedT> {
  private:
    OpenCLBackend* backend_;
    std::string name_;
    unsigned int nevents_;
    unsigned int nevents_padded_;
    unsigned int nevents_per_workitem_;
    std::vector<dimension<evalT>> input_signature_;
    std::vector<dimension<evalT>> output_signature_;
    //opencl handles
    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;  
    cl_mem device_data_in_;
    cl_mem device_seed_in_;
    cl_mem device_data_out_;
    cl_mem device_parameters_;
    cl_mem kahan_result_;
    cl_kernel kernel_;
    cl_program program_;
    cl_kernel kahan_kernel_;
    cl_program kahan_program_;
    //perform kahan summation
    bool kahan_summation_;
    bool generate_kernel_;
  public:
    std::vector<double> kahan_results_;
    OpenCLBlock(OpenCLBackend* backend):
      backend_(backend),
      name_("kernel"),
      nevents_(0),
      nevents_padded_(0),
      nevents_per_workitem_(1),
      context_(backend->context()),
      queue_(backend->queue()),
      device_(backend->device()),
      device_data_in_(NULL),
      device_seed_in_(NULL),
      device_data_out_(NULL),
      device_parameters_(NULL),
      kahan_result_(NULL),
      kernel_(NULL),
      program_(NULL),
      kahan_kernel_(NULL),
      kahan_program_(NULL),
      kahan_summation_(false),
      generate_kernel_(false)
    {
    }
    ~OpenCLBlock()
    {      
      if (device_data_in_)
	clReleaseMemObject(device_data_in_);
      if (device_data_out_)
	clReleaseMemObject(device_data_out_);
      if (device_parameters_)
      	clReleaseMemObject(device_parameters_);
      if (device_seed_in_)
      	clReleaseMemObject(device_seed_in_);
      if (kernel_)
	clReleaseKernel(kernel_);
      if (program_)
	clReleaseProgram(program_);
    }
    virtual bool SetupInputBuffer(unsigned long int nbytes) override
    {
      if (device_data_in_)
	clReleaseMemObject(device_data_in_);
      cl_int CL_err = CL_SUCCESS;
      device_data_in_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, nbytes, NULL, &CL_err);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error setting up input buffer: " << CL_err << std::endl;
	  assert(0);
	}
      return true;
    }
    virtual bool SetupSeedBuffer(unsigned long int nbytes) override
    {
      if (device_seed_in_)
	clReleaseMemObject(device_seed_in_);
      cl_int CL_err = CL_SUCCESS;
      device_seed_in_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, nbytes, NULL, &CL_err);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error setting up seed buffer: " << CL_err << std::endl;
	  assert(0);
	}
      return true;
    }
    virtual bool SetupInputBuffer(ComputeBlock<kernelT, evalT>* input, bool use_data_in=false) override
    {
      OpenCLBlock<kernelT, evalT>* inputblock = dynamic_cast<OpenCLBlock<kernelT, evalT>*>(input);
      if (inputblock == NULL)
	{
	  std::cout << "Input block needs to be an OpenCLBlock"  << std::endl;
	  assert(0);
	}
      if (device_data_in_)
      	clReleaseMemObject(device_data_in_);
      if (use_data_in)
	device_data_in_ = inputblock->device_data_in_;
      else
	device_data_in_ = inputblock->device_data_out_;
      clRetainMemObject(device_data_in_);
      return true;
    }
    virtual bool SetupParameterBuffer(unsigned long int nbytes) override
    {
      if (device_parameters_)
      	clReleaseMemObject(device_parameters_);
      cl_int CL_err = CL_SUCCESS;
      device_parameters_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, nbytes, NULL, &CL_err);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error setting up parameter buffer: " << CL_err << std::endl;
	  assert(0);
	}
      return true;
    }
    virtual bool SetupOutputBuffer(unsigned long int nbytes) override
    {
      if (device_data_out_)
	clReleaseMemObject(device_data_out_);
      cl_int CL_err = CL_SUCCESS;
      device_data_out_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, nbytes, NULL, &CL_err);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error setting up output buffer: " << CL_err << std::endl;
	  assert(0);
	}    
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
      if (nevents_ != data.nevents())
	{
	  std::cout << "Error: event numbers inconsistent for input (Ndata=" << data.nevents() << ", Nevents=" << nevents_padded_ << ")" << nevents_ << std::endl;
	  assert(0);
	}
      cl_int CL_err = CL_SUCCESS;      
      CL_err = clEnqueueWriteBuffer(queue_, device_data_in_, CL_TRUE, 0, data.buffer_size(), data.get_data(), 0, NULL, NULL);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error writing input buffer: " << CL_err << std::endl;
	  assert(0);
	}    
      return true;
    }
    virtual bool CopyToSeedBuffer(const EventVector<seedT, evalT>& data) override
    {
      cl_int CL_err = CL_SUCCESS;      
      CL_err = clEnqueueWriteBuffer(queue_, device_seed_in_, CL_TRUE, 0, data.buffer_size(), data.get_data(), 0, NULL, NULL);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error writing seed buffer: " << CL_err << std::endl;
	  assert(0);
	}    
      return true;
    }
    virtual bool CopyToParameterBuffer(const std::vector<kernelT>& params) override
    {
      cl_int CL_err = CL_SUCCESS;
      kernelT param_buffer[params.size()];
      for (unsigned int i=0; i<params.size(); i++)
	param_buffer[i] = params.at(i);
      CL_err = clEnqueueWriteBuffer(queue_, device_parameters_, CL_TRUE, 0, params.size()*sizeof(kernelT), param_buffer, 0, NULL, NULL);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error writing parameter buffer: " << CL_err << std::endl;
	  assert(0);
	}    
      return true;      
    }
    virtual bool CopyFromOutputBuffer(EventVector<kernelT, evalT>& data) override
    {
      if (nevents_ != data.nevents())
	{
	  std::cout << "Error: event numbers inconsistent for input (Ndata=" << data.nevents() << ", Nevents=" << nevents_padded_ << ")" << std::endl;
	  assert(0);
	}
      //check and possibly correct padding of input data
      if (nevents_ != nevents_padded_ && data.nevents_padded() != nevents_padded_)
	{
	  data.set_padding(true, backend_->required_padding());
	  if (data.nevents_padded() != nevents_padded_)
	    {
	      std::cout << "Error: Inconsistent padding of output buffer" << std::endl;
	      assert(0);
	    }
	  std::cout << "Need to apply additional padding to output data, changing from " << nevents_ << " to " << nevents_padded_ << " (padded) events" << std::endl;
	}
      cl_int CL_err = CL_SUCCESS;            
      CL_err = clEnqueueReadBuffer(queue_, device_data_out_, CL_TRUE, 0, data.buffer_size(), data.get_data(), 0, NULL, NULL);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error reading output buffer: " << CL_err << std::endl;
	  std::cout <<  nevents_ << " " << nevents_padded_ << " " << data.buffer_size() << std::endl;
	  assert(0);
	}    
      return true;
    }
    virtual bool CopyFromKahanBuffer(std::vector<evalT>& kahan_sums)
    {
      kahan_sums = kahan_results_;
      return true;
    };
    virtual bool MakeGenerateKernel(std::string name, unsigned int nevents, std::vector<dimension<evalT>> input_signature,  std::vector<dimension<evalT>> output_signature,
				    const std::vector<std::string>& params, const std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& graphs, evalT maxprob=1.0) override
    {
      name_ = name;
      nevents_ = nevents;
      nevents_padded_ = nevents;
      if (nevents_ % backend_->required_padding() != 0)
	{
	  nevents_padded_ = (int(nevents_ / backend_->required_padding()) + 1)*backend_->required_padding();
	  std::cout << "Need to apply additional padding from " << nevents_ << " to " << nevents_padded_ <<" events" << std::endl;
	}
      input_signature_ = input_signature;
      output_signature_ = output_signature;
      if (backend_->opts_->opencl_max_generator_workitems > 0 && int(nevents_padded_) > backend_->opts_->opencl_max_generator_workitems)
	{
	  nevents_per_workitem_ = nevents_padded_/backend_->opts_->opencl_max_generator_workitems + 1;
	  std::cout << "Events per workitem " << nevents_per_workitem_ << " (nevents " << nevents_ << ", nevents_padded " << nevents_padded_ << ", max_generator_workitems " << backend_->opts_->opencl_max_generator_workitems << ")" << std::endl;
	}
      else
	nevents_per_workitem_ = 1;
      
      generate_kernel_ = true;

      //safeties
      assert(input_signature.size() == 4);
      assert(graphs.size() == 1);

      std::string kernelT_str = graphs.at(0)->kernelT_str();
      cl_int CL_err = CL_SUCCESS;            
      std::string kernel_code;
      if (kernelT_str == "double")
	kernel_code += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
      kernel_code += "\n";

      if constexpr (std::is_same_v<uint32_t, seedT>)
	{
	  //xoshiro128++
	  kernel_code += "inline uint rol32(uint x, int k)\n";
	  kernel_code += "{\n";
	  kernel_code += "  return (x << k) | (x >> (32 - k));\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";      
	  kernel_code += "inline " + kernelT_str+" xoshiro(uint* state)\n";
	  kernel_code += "{\n";      
	  kernel_code += "  uint const result = rol32(state[0] + state[3], 7) + state[0];\n";
	  kernel_code += "  uint const t = state[1] << 9;\n";
	  kernel_code += "  state[2] ^= state[0];\n";
	  kernel_code += "  state[3] ^= state[1];\n";
	  kernel_code += "  state[1] ^= state[2];\n";
	  kernel_code += "  state[0] ^= state[3];\n";
	  kernel_code += "  state[2] ^= t;\n";
	  kernel_code += "  state[3] = rol32(state[3], 11);\n";
	  kernel_code += "  return result * 2.3283064365386963e-10;\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";
	  kernel_code += "__kernel void "+name_+"(__const int nevents, __const int nevents_padded, __const int nevents_per_workitem, __global const uint * data, __global "+kernelT_str+"* output)\n";      
	}
      if constexpr (std::is_same_v<uint64_t, seedT>)
	{
	  //xoshiro256++
	  kernel_code += "inline ulong rol64(ulong x, int k)\n";
	  kernel_code += "{\n";
	  kernel_code += "  return (x << k) | (x >> (64 - k));\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";      
	  kernel_code += kernelT_str+" xoshiro(ulong* state)\n";
	  kernel_code += "{\n";      
	  kernel_code += "  ulong const result = rol64(state[0] + state[3], 23) + state[0];\n";
	  kernel_code += "  ulong const t = state[1] << 17;\n";
	  kernel_code += "  state[2] ^= state[0];\n";
	  kernel_code += "  state[3] ^= state[1];\n";
	  kernel_code += "  state[1] ^= state[2];\n";
	  kernel_code += "  state[0] ^= state[3];\n";
	  kernel_code += "  state[2] ^= t;\n";
	  kernel_code += "  state[3] = rol64(state[3], 45);\n";
	  kernel_code += "  return (result >> 11) * 0x1.0p-53;\n";
	  kernel_code += "}\n";
	  kernel_code += "\n";
	  kernel_code += "__kernel void "+name_+"(__const int nevents, __const int nevents_padded, __const int nevents_per_workitem, __global const ulong * data, __global "+kernelT_str+"* output)\n";      
	}

      kernel_code += "{\n";
      kernel_code += "  int idx = get_global_id(0);\n";      
      if (backend_->opts_->opencl_max_generator_workitems > 0)
	{
	  kernel_code += "  int from = idx*nevents_per_workitem;\n";
	  kernel_code += "  int to = (idx+1)*nevents_per_workitem;\n";
	  kernel_code += "  if (from >= nevents_padded)\n";
	  kernel_code += "    return;\n";
	  kernel_code += "  if (to >= nevents_padded)\n";
	  kernel_code += "    to = nevents_padded;\n";
	}
      //set up kernel inputs
      if constexpr (std::is_same_v<uint32_t, seedT>)
	kernel_code += "  uint rnd_state[4];\n";
      if constexpr (std::is_same_v<uint64_t, seedT>)
	kernel_code += "  ulong rnd_state[4];\n";
      for (unsigned int i=0; i<input_signature.size(); i++)
	kernel_code += "  rnd_state[" + std::to_string(i) + "] = data["+ (i==0 ? std::string("idx") : ("nevents_padded*" + std::to_string(i)+std::string("+idx"))) + "];\n";

      //adding parameter definitions
      for (unsigned int i=0; i<params.size(); i++)
	{
	  kernel_code += "  const "+kernelT_str+" ";
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

      if (backend_->opts_->opencl_max_generator_workitems > 0) //multiple events per kernel
	{
	  kernel_code += "  for (int i=from; i<to; i++)\n";
	  kernel_code += "  {\n";
	  kernel_code += "  bool finished = false;\n";
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
	  kernel_code += "      finished = true;\n";
	  kernel_code += "    }\n";      
	  kernel_code += "  }\n";
	  kernel_code += "  while(!finished);\n";
	  kernel_code += "  }\n";
	  kernel_code += "}\n";

	}
      else //one event per kernel
	{
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
	      kernel_code += "      output[" + (i==0 ? std::string("idx") : ("nevents_padded*" + std::to_string(i)+std::string("+idx"))) + "] = ";
	      kernel_code +=  output_signature.at(i).get_name(); 
	      kernel_code += ";\n";	  
	    }
	  kernel_code += "      return;\n";
	  kernel_code += "    }\n";      
	  kernel_code += "  }\n";
	  kernel_code += "  while(true);\n";
	  kernel_code += "}\n";
	}
      if (backend_->opts_->print_kernel)
	std::cout << std::endl << "KERNEL SOURCECODE:" << std::endl << kernel_code << std::endl << "END KERNEL SOURCECODE" << std::endl;
      const char* kernel_cstr = kernel_code.c_str();
      //create the compute program
      program_ = clCreateProgramWithSource(context_, 1, (const char **)& kernel_cstr, NULL, NULL);
      //build the compute program executable
      CL_err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
      if (CL_err != CL_SUCCESS)
	{
	  //print build log on problem
	  std::cout << "Error compiling kernel: " << CL_err << std::endl;
	  size_t logSize = 0;
	  clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	  char log[logSize];
	  clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	  printf("build log: %s\n", log);
	}
      else
	std::cout << "Generate kernel built successfully (CL_err " << CL_err << ")" << std::endl;	
      //create the compute kernel
      kernel_ = clCreateKernel(program_, name_.c_str(), NULL);

      //setting kernel arguments
      clSetKernelArg(kernel_, 0, sizeof(int), &nevents_); 
      clSetKernelArg(kernel_, 1, sizeof(int), &nevents_padded_);
      clSetKernelArg(kernel_, 2, sizeof(int), &nevents_per_workitem_);
      clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void *)&device_seed_in_); 
      clSetKernelArg(kernel_, 4, sizeof(cl_mem), (void *)&device_data_out_);
      if (params.size() > 0)
	clSetKernelArg(kernel_, 5, sizeof(cl_mem), (void *)&device_parameters_);
      return true;
    }
    virtual bool MakeComputeKernel(std::string name, unsigned int nevents, std::vector<dimension<evalT>> input_signature,  std::vector<dimension<evalT>> output_signature,
				   const std::vector<std::string>& params, const std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>>& graphs, bool kahan_summation=false) override
    {
      kahan_summation_ = kahan_summation;
      name_ = name;
      nevents_ = nevents;
      nevents_padded_ = nevents;
      if (nevents_ % backend_->required_padding() != 0)
	{
	  nevents_padded_ = (int(nevents_ / backend_->required_padding()) + 1)*backend_->required_padding();
	  std::cout << "Need to apply additional padding from " << nevents_ << " to " << nevents_padded_ <<" events" << std::endl;
	}
      input_signature_ = input_signature;
      output_signature_ = output_signature;
      if (backend_->opts_->opencl_max_compute_workitems > 0 && int(nevents_padded_) > backend_->opts_->opencl_max_compute_workitems)
	{
	  nevents_per_workitem_ = nevents_padded_/backend_->opts_->opencl_max_compute_workitems + 1;
	  std::cout << "Events per workitem " << nevents_per_workitem_ << " (nevents " << nevents_ << ", nevents_padded " << nevents_padded_ << ", max_compute_workitems " << backend_->opts_->opencl_max_compute_workitems << ")" << std::endl;
	}
      else
	nevents_per_workitem_ = 1;

      //cleanup if necessary
      if (kernel_)
	clReleaseKernel(kernel_);
      if (program_)
	clReleaseProgram(program_);
      //write kernel code
      std::string kernelT_str = graphs.at(0)->kernelT_str();
      cl_int CL_err = CL_SUCCESS;            
      std::string kernel_code;
      if (kernelT_str == "double")
	kernel_code += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
      kernel_code += "\n";
      if (backend_->opts_->opencl_max_compute_workitems > 0)//multiple events per kernel
	{
	  if (params.size() > 0)
	    kernel_code += ("__kernel void "+name_+"(__const int nevents, __const int nevents_padded, __const int nevents_per_workitem, __global const "+kernelT_str+"* data, __global "+kernelT_str+"* output, __global const "+kernelT_str+"* parameters)\n");
	  else
	    kernel_code += ("__kernel void "+name_+"(__const int nevents, __const int nevents_padded, __const int nevents_per_workitem, __global const "+kernelT_str+"* data, __global "+kernelT_str+"* output)\n");
	}
      else
	{
	  if (params.size() > 0)
	    kernel_code += ("__kernel void "+name_+"(__const int nevents, __const int nevents_padded, __global const "+kernelT_str+"* data, __global "+kernelT_str+"* output, __global const "+kernelT_str+"* parameters)\n");
	  else
	    kernel_code += ("__kernel void "+name_+"(__const int nevents, __const int nevents_padded, __global const "+kernelT_str+"* data, __global "+kernelT_str+"* output)\n");
	}
      kernel_code += "{\n";
      kernel_code += "int idx = get_global_id(0);\n";

      if (backend_->opts_->opencl_max_compute_workitems > 0)//multiple events per kernel
	{
	  kernel_code += "  int from = idx*nevents_per_workitem;\n";
	  kernel_code += "  int to = (idx+1)*nevents_per_workitem;\n";
	  kernel_code += "  if (from >= nevents_padded)\n";
	  kernel_code += "    return;\n";
	  kernel_code += "  if (to >= nevents_padded)\n";
	  kernel_code += "    to = nevents_padded;\n";
	}

      assert(graphs.size() == output_signature.size());

      if (backend_->opts_->opencl_max_compute_workitems > 0) //multiple events per kernel
	{
	  //adding parameter definitions
	  for (unsigned int i=0; i<params.size(); i++)
	    {
	      kernel_code += "  const "+kernelT_str+" ";
	      kernel_code += params.at(i);
	      kernel_code += " = parameters[";
	      kernel_code += std::to_string(i);
	      kernel_code += "];\n";
	    }
	  kernel_code += "  for (unsigned int i=from; i<to; i++)\n";
	  kernel_code += "  {\n";
	  //set up kernel inputs
	  for (unsigned int i=0; i<input_signature.size(); i++)
	    {
	      kernel_code += "    const " + kernelT_str + " ";
	      kernel_code += input_signature.at(i).get_name();
	      kernel_code += std::string(" = data[") + (i==0 ? std::string("i") : ("nevents_padded*" + std::to_string(i)+std::string("+i"))) + std::string("];\n");
	    }
	  //actual kernel calculating the prob
	  for (unsigned int i=0; i<graphs.size(); i++)
	    {
	      kernel_code += std::string("    output[") + (i==0 ? std::string("i") : ("nevents_padded*" + std::to_string(i)+std::string("+i"))) + std::string("] = ");
	      kernel_code +=  graphs.at(i)->get_kernel(); 
	      kernel_code += ";\n";
	    }
	  kernel_code += "  }\n";
	}
      else //single event per kernel
	{
	  //set up kernel inputs
	  for (unsigned int i=0; i<input_signature.size(); i++)
	    {
	      kernel_code += "const " + kernelT_str + " ";
	      kernel_code += input_signature.at(i).get_name();
	      kernel_code += std::string(" = data[") + (i==0 ? std::string("idx") : ("nevents_padded*" + std::to_string(i)+std::string("+idx"))) + std::string("];\n");
	    }
	  //adding parameter definitions
	  for (unsigned int i=0; i<params.size(); i++)
	    {
	      kernel_code += "const "+kernelT_str+" ";
	      kernel_code += params.at(i);
	      kernel_code += " = parameters[";
	      kernel_code += std::to_string(i);
	      kernel_code += "];\n";
	    }
	  //actual kernel calculating the prob
	  for (unsigned int i=0; i<graphs.size(); i++)
	    {
	      kernel_code += std::string("output[") + (i==0 ? std::string("idx") : ("nevents_padded*" + std::to_string(i)+std::string("+idx"))) + std::string("] = ");
	      kernel_code +=  graphs.at(i)->get_kernel(); 
	      kernel_code += ";\n";
	    }
	}
      kernel_code += "}\n";
      if (backend_->opts_->print_kernel)
	std::cout << std::endl << "KERNEL SOURCECODE:" << std::endl << kernel_code << std::endl << "END KERNEL SOURCECODE" << std::endl;
      const char* kernel_cstr = kernel_code.c_str();
      //create the compute program
      program_ = clCreateProgramWithSource(context_, 1, (const char **)& kernel_cstr, NULL, NULL);
      //build the compute program executable
      CL_err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
      if (CL_err != CL_SUCCESS)
	{
	  // print build log on problem
	  std::cout << "Error compiling kernel: " << CL_err << std::endl;
	  size_t logSize = 0;
	  clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	  char log[logSize];
	  clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	  printf("build log: %s\n", log);
	}
      else
      	std::cout << "Compute kernel built successfully (CL_err " << CL_err << ")" << std::endl;	

      // create the compute kernel
      kernel_ = clCreateKernel(program_, name_.c_str(), NULL);
      //setting kernel arguments
      clSetKernelArg(kernel_, 0, sizeof(int), &nevents_); 
      clSetKernelArg(kernel_, 1, sizeof(int), &nevents_padded_);
      if (backend_->opts_->opencl_max_compute_workitems > 0)
	{
	  clSetKernelArg(kernel_, 2, sizeof(int), &nevents_per_workitem_);
	  clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void *)&device_data_in_);
	  clSetKernelArg(kernel_, 4, sizeof(cl_mem), (void *)&device_data_out_);
	  if (params.size() > 0)
	    clSetKernelArg(kernel_, 5, sizeof(cl_mem), (void *)&device_parameters_);
	}
      else
	{
	  clSetKernelArg(kernel_, 2, sizeof(cl_mem), (void *)&device_data_in_);
	  clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void *)&device_data_out_);
	  if (params.size() > 0)
	    clSetKernelArg(kernel_, 4, sizeof(cl_mem), (void *)&device_parameters_);
	}
      
      kahan_summation_ = kahan_summation;
      if (kahan_summation_)//repeated Kahan summation each reducing previous stage by defined reduction factor
	{
	  //reduction algorithm
	  //start with minimal reduction factor, eg. 64
	  //repeatedly 
	  //reserve output memory = input memory / reduction factor
	  //run nevents / reduction factor work items
	  //add up event idx % reduction factor via kahan summation -> save to output memory[0...input length/reduction factor]	  
	  std::string kahan_kernel;
	  //simplified kahan reduction strategy
	  if (kernelT_str == "double")
	    kahan_kernel += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
	  kahan_kernel += "__kernel void kahan_"+name_+"(__global "+kernelT_str+"* buffer, __const int length, __const int length_padded, __const int noutput, __const int noutput_padded, __global "+kernelT_str+"* result)\n";
	  kahan_kernel += "{\n";
	  kahan_kernel += "  int global_index = get_global_id(0);\n";
	  kahan_kernel += "  "+kernelT_str+" accumulators["+std::to_string(graphs.size())+"];\n";
	  kahan_kernel += "  "+kernelT_str+" cs["+std::to_string(graphs.size())+"];\n";
	  //want to unroll these loops manually
	  for (unsigned int i=0; i<graphs.size(); i++)
	    {
	      kahan_kernel += "  accumulators["+std::to_string(i)+"] = 0.0;\n";
	      kahan_kernel += "  cs["+std::to_string(i)+"] = 0.0;\n";
	    }
	  kahan_kernel += "  //noutput workers add up all elements in input (length)\n";
	  kahan_kernel += "  while (global_index < length) {\n";
	  for (unsigned int i=0; i<graphs.size(); i++)
	    {
	      if (i==0)
		{
		  kahan_kernel += "    "+kernelT_str+" element = buffer[global_index+length_padded*"+std::to_string(i)+"] - cs["+std::to_string(i)+"];\n";
		  kahan_kernel += "    volatile "+kernelT_str+" t = accumulators["+std::to_string(i)+"] + element;\n";
		  kahan_kernel += "    cs["+std::to_string(i)+"] = (t - accumulators["+std::to_string(i)+"]) - element;\n";
		  kahan_kernel += "    accumulators["+std::to_string(i)+"] = t;\n";
		}
	      else
		{
		  kahan_kernel += "    element = buffer[global_index+length_padded*"+std::to_string(i)+"] - cs["+std::to_string(i)+"];\n";
		  kahan_kernel += "    t = accumulators["+std::to_string(i)+"] + element;\n";
		  kahan_kernel += "    cs["+std::to_string(i)+"] = (t - accumulators["+std::to_string(i)+"]) - element;\n";
		  kahan_kernel += "    accumulators["+std::to_string(i)+"] = t;\n";
		}
	    }	  
	  kahan_kernel += "    global_index += noutput;\n";
	  kahan_kernel += "  }\n";

	  for (unsigned int i=0; i<graphs.size(); i++)
	    kahan_kernel += "  result[get_global_id(0)+noutput_padded*"+std::to_string(i)+"] = accumulators["+std::to_string(i)+"];\n";
	  kahan_kernel += "}\n";
	  if (backend_->opts_->print_kernel)
	    std::cout << std::endl << "KAHAN KERNEL SOURCECODE:" << std::endl << kahan_kernel << std::endl << "END KAHAN KERNEL SOURCECODE" << std::endl;	  
	  const char* kahan_kernel_cstr = kahan_kernel.c_str();	  
	  //create the compute program
	  kahan_program_ = clCreateProgramWithSource(context_, 1, (const char **)& kahan_kernel_cstr, NULL, NULL);
	  //build the compute program executable
	  CL_err = clBuildProgram(kahan_program_, 0, NULL, NULL, NULL, NULL);
	  if (CL_err != CL_SUCCESS)
	    {
	      //print build log on problem
	      std::cout << "Error compiling kernel: " << CL_err << std::endl;
	      size_t logSize = 0;
	      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	      char log[logSize];
	      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	      printf("build log: %s\n", log);
	    }
	  else
	    std::cout << "Kahan kernel built successfully (CL_err " << CL_err << ")" << std::endl;	
	  //create the compute kernel
	  kahan_kernel_ = clCreateKernel(kahan_program_, ("kahan_"+name_).c_str(), NULL);

	}
      return true;
    }
    virtual bool SubmitKernel() override
    {
      clSetKernelArg(kernel_, 0, sizeof(int), &nevents_); 
      clSetKernelArg(kernel_, 1, sizeof(int), &nevents_padded_);
      if (generate_kernel_)
	{
	  clSetKernelArg(kernel_, 2, sizeof(int), &nevents_per_workitem_);
	  clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void *)&device_seed_in_);
	  clSetKernelArg(kernel_, 4, sizeof(cl_mem), (void *)&device_data_out_);
	}
      else
	{
	  if (backend_->opts_->opencl_max_compute_workitems>0)
	    {
	      clSetKernelArg(kernel_, 2, sizeof(int), &nevents_per_workitem_);
	      clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void *)&device_data_in_);
	      clSetKernelArg(kernel_, 4, sizeof(cl_mem), (void *)&device_data_out_);
	    }
	  else
	    {
	      clSetKernelArg(kernel_, 2, sizeof(cl_mem), (void *)&device_data_in_);
	      clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void *)&device_data_out_);
	    }
	}
      
      unsigned int globalworksize = nevents_padded_;
      if (generate_kernel_ && backend_->opts_->opencl_max_generator_workitems>0 && int(globalworksize) > backend_->opts_->opencl_max_generator_workitems)
	globalworksize = backend_->opts_->opencl_max_generator_workitems;//actually can run fewer kernels, however superfluous kerenels immediately return
      if (!generate_kernel_ && backend_->opts_->opencl_max_compute_workitems>0 && int(globalworksize) > backend_->opts_->opencl_max_compute_workitems)
	globalworksize = backend_->opts_->opencl_max_compute_workitems;
      
      unsigned int localworksize = backend_->required_padding();      
      cl_int CL_err = CL_SUCCESS;            
      size_t globalWorkSize[1] = {globalworksize};
      size_t localWorkSize[1] = {localworksize};
      CL_err = clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error submitting kernel: " << CL_err << " " << globalworksize << " " << localworksize << std::endl;
	  assert(0);
	}
      //output of compute kernel should be reduced on accelerator
      if (kahan_summation_)
	{
	  //the first time the input is the output of compute kernel
	  const unsigned int reduction_factor = backend_->opts_->opencl_kahan_reduction_factor;//try to sum at least 32 numbers per work item, best by test
	  
	  clSetKernelArg(kahan_kernel_, 0, sizeof(cl_mem), (void *)&device_data_out_);

	  bool finished = false;
	  int last_result_size = nevents_;
	  int nlength[1] = {int(nevents_)};
	  int nlength_padded[1] = {int(nevents_padded_)};
	  std::vector<cl_mem> kahan_cleanup;
	  //repeatedly reduce by reduction factor
	  while (!finished)
	    {
	      //global_work_size needs to be multiple of local_work_group_size
	      int resultsize = last_result_size/reduction_factor + (last_result_size % reduction_factor == 0 ? 0 : 1);
	      int resultsize_padded = resultsize + ((localworksize - (resultsize % localworksize)) % localworksize);
	      if (resultsize == 1)
		resultsize_padded = 1;
	      kahan_result_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, resultsize_padded*sizeof(kernelT)*output_signature_.size(), NULL, &CL_err);
	      kahan_cleanup.push_back(kahan_result_);

	      clSetKernelArg(kahan_kernel_, 1, sizeof(int), &nlength);
	      clSetKernelArg(kahan_kernel_, 2, sizeof(int), &nlength_padded);
	      int noutput[1] = {resultsize};
	      clSetKernelArg(kahan_kernel_, 3, sizeof(int), &noutput);
	      int noutput_padded[1] = {resultsize_padded};
	      clSetKernelArg(kahan_kernel_, 4, sizeof(int), &noutput_padded);
	      clSetKernelArg(kahan_kernel_, 5, sizeof(cl_mem), (void *)&kahan_result_); 
	      clFinish(queue_); //needs previous kernel to be completed
	      
	      const size_t currentGlobalWorkSize[1] = {resultsize_padded == 1 ? localworksize : resultsize_padded};
	      CL_err = clEnqueueNDRangeKernel(queue_, kahan_kernel_, 1, NULL, currentGlobalWorkSize, localWorkSize, 0, NULL, NULL);//submit reduction kernel	      
	      clFinish(queue_);

	      last_result_size = resultsize;
	      nlength[0] = resultsize;
	      nlength_padded[0] = resultsize_padded;

	      //at the end set new input = previous result
	      clSetKernelArg(kahan_kernel_, 0, sizeof(cl_mem), (void *)&kahan_result_);	      

	      if (false)//intermediate debug output check
		{
		  long unsigned int intermediate_size = resultsize_padded*output_signature_.size();
		  kernelT* intermediate_result = new kernelT[intermediate_size];
		  CL_err = clEnqueueReadBuffer(queue_, kahan_result_, CL_TRUE, 0, sizeof(kernelT)*intermediate_size, intermediate_result, 0, NULL, NULL);
		  std::cout << "signature size " << output_signature_.size() << " resultsize " << resultsize << " resultsize_padded " << resultsize_padded <<  std::endl;
		  delete[] intermediate_result;
		}
	      if (resultsize <= 1)
		{
		  finished = true;
		  kernelT result[output_signature_.size()];
		  CL_err = clEnqueueReadBuffer(queue_, kahan_result_, CL_TRUE, 0, sizeof(kernelT)*output_signature_.size(), result, 0, NULL, NULL);
		  if (CL_err != CL_SUCCESS)
		    {
		      std::cout << "Error reading output buffer: " << CL_err << std::endl;
		      assert(0);
		    }
		  kahan_results_.clear();
		  for (unsigned int i=0; i<output_signature_.size(); i++)
		    kahan_results_.push_back(result[i]);
		}

	    }
	  for (unsigned int i=0; i<kahan_cleanup.size(); i++)
	    clReleaseMemObject(kahan_cleanup.at(i));
	  
	}
      
      return true;
    }
    virtual bool Finish() override
    {
      cl_int CL_err = CL_SUCCESS;
      CL_err = clFinish(queue_);
      if (CL_err != CL_SUCCESS)
	{
	  std::cout << "Error finishing: " << CL_err << std::endl;
	  assert(0);
	}

      return true;
    } 
    virtual bool ProvidesKahan() {return true;}
    const cl_context context() {return context_;};
    const cl_command_queue queue() {return queue_;};
    const cl_device_id device() {return device_;};  
  };
  

}

#endif
