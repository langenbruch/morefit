/**
 * @file generator.hh
 * @author Christoph Langenbruch
 * @date 2024-11-16
 *
 */

#ifndef GENERATOR_H
#define GENERATOR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include <array>

#include "graph.hh"
#include "eventvector.hh"
#include "parametervector.hh"
#include "pdf.hh"
#include "physicspdfs.hh"
#include "compute.hh"
#include "random.hh"

namespace morefit {

  struct generator_options {
    enum randomization_type { on_accelerator, on_host };
    randomization_type rndtype;
    double chunksize_multiplier;
    int print_level;
    generator_options():
      rndtype(randomization_type::on_accelerator),
      chunksize_multiplier(1.0),
      print_level(2)
    {}
    void print()
    {
      const unsigned int width = 40;
      std::cout << "GENERATOR OPTIONS" << std::endl;
      std::cout << std::setw(width) << std::left << "  randomization type ";
      switch (rndtype) {
      case randomization_type::on_accelerator: std::cout << "ON ACCELERATOR" << std::endl; break;
      case randomization_type::on_host: std::cout << "ON HOST" << std::endl; break;
      default: std::cout << "UNKNOWN" << std::endl;
      }
      std::cout << std::setw(width) << std::left << "  chunksize multiplier " << chunksize_multiplier << std::endl;
      std::cout << std::setw(width) << std::left << "  print level " << print_level << std::endl;
    }
  };

  
  template<typename kernelT, typename evalT, typename backendT, typename computeT, typename seedT=uint32_t>
  class generator {
  private:
    generator_options* opts_;
    backendT* backend_;
    computeT block_;
    RandomGenerator* rnd_;
    std::vector<dimension<evalT>> input_dimensions_;
  public:
    generator(generator_options* opts, backendT* backend, RandomGenerator* rnd):
      opts_(opts),
      backend_(backend),
      block_(computeT(backend)),
      rnd_(rnd)
    {}
    bool generate(unsigned int nevents, PDF<kernelT, evalT>* prob, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>& result, bool regen=false)
    {
      auto t_before_generation = std::chrono::high_resolution_clock::now();     
      evalT maxprob = prob->get_max();
      if (opts_->rndtype == generator_options::randomization_type::on_accelerator)//random number generation on accelerator
	{
	  result.resize(nevents);
	  unsigned int localworksize = backend_->required_padding();
	  if (localworksize > 0 && nevents % localworksize != 0)
	    {
	      int nevents_before = nevents;
	      int nevents_padded = (int(nevents / localworksize) + 1)*localworksize;
	      std::cout << "Warning: nevents of " << nevents_before << " is not a multiple of work group size " << localworksize << ", automatically padded nevents to " << nevents_padded << std::endl;
	      result.set_padding(true, localworksize);
	    }

	  std::vector<dimension<evalT>*> input_dimensions;
	  for (unsigned int i=0; i<4; i++)
	    input_dimensions_.push_back(dimension<evalT>("rnd_seed"+std::to_string(i)));
	  for (unsigned int i=0; i<4; i++)
	    input_dimensions.push_back(&(input_dimensions_.at(i)));

	  EventVector<seedT, evalT> input_buffer(input_dimensions);
	  input_buffer.set_padding(localworksize != 0 ? true : false, localworksize);
	  //now distribute seeds
	  Xoshiro128pp* xoshiro = dynamic_cast<Xoshiro128pp*>(rnd_);
	  if (xoshiro == NULL)
	    {
	      std::cout << "Distributed random number generation currently only implemented for Xoshiro128++ random generator." << std::endl;
	      assert(0);
	    }
	  if (dynamic_cast<LLVMBackend*>(backend_) != NULL) //check if this is multithreading where we only need to generate nthread seeds
	    {
	      unsigned int nthreads = backend_->opts_->llvm_nthreads;
	      input_buffer.resize(nthreads);
	      //initialise Nthreads different seeds
	      for (unsigned int i=0; i<nthreads; i++)
		{
		  for (unsigned int j=0; j<4; j++)
		    input_buffer(i, j) = xoshiro->getSeed(j);
		  xoshiro->jump();
		}
	    }
	  else //opencl and other methods
	    {
	      input_buffer.resize(nevents);
	      //new approach using maximally opts_->opencl_max_generator_workitems
	      unsigned int nseeds = result.nevents_padded();
	      if (backend_->opts_->opencl_max_generator_workitems > 0 && int(nseeds) > backend_->opts_->opencl_max_generator_workitems)
		{
		  int nevents_per_workitem = result.nevents_padded()/backend_->opts_->opencl_max_generator_workitems + 1;
		  nseeds = result.nevents_padded()/nevents_per_workitem + 1;
		  if (opts_->print_level > 1)
		    std::cout << "Number of seeds to generate " << nseeds << " " << nevents_per_workitem << " " << backend_->opts_->opencl_max_generator_workitems << std::endl;
		}
	      for (unsigned int i=0; i<nseeds; i++)//need to account for potential padding since uninitialized PRNG will fail
		{
		  for (unsigned int j=0; j<4; j++)
		    input_buffer(i, j) = xoshiro->getSeed(j);
		  xoshiro->jump();
		}
	    }	  
	  //set up buffers
	  block_.SetupSeedBuffer(input_buffer.buffer_size());	  
	  block_.SetupOutputBuffer(result.buffer_size());
	  block_.SetNevents(result.nevents(), result.nevents_padded());
	  if (!regen)
	    {
	      //prepare kernel
	      std::vector<std::string> param_names;
	      std::vector<evalT> param_values;
	      for (auto param : params)
		{
		  param_names.push_back(param->get_name());
		  param_values.push_back(param->get_value());      
		}
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> graphs;
	      graphs.emplace_back(std::move(prob->prob_normalised()->substitute(param_names, param_values)->simplify()));
	      if (opts_->print_level > 1)
		{
		  //print graph of generator pdf
		  //graphs.at(0)->draw("gen_graph.tex");
		}
	      std::vector<std::string> dummy;

	      auto t_before_making = std::chrono::high_resolution_clock::now();
	      block_.MakeGenerateKernel("gen_kernel", nevents, input_buffer.copy_dimensions(), result.copy_dimensions(), dummy, graphs, maxprob);
	      auto t_after_making = std::chrono::high_resolution_clock::now();
	      if (opts_->print_level > 1)
		std::cout << "making generate kernel took " << std::chrono::duration<double, std::milli>(t_after_making-t_before_making).count() << " ms in total" << std::endl;

	    }
	  block_.Finish();
	  auto t_before_launch = std::chrono::high_resolution_clock::now(); 
	  //copy seeds to input buffer
	  block_.CopyToSeedBuffer(input_buffer);
	  //submit kernel
	  block_.SubmitKernel();
	  block_.Finish();
	  auto t_after_finish = std::chrono::high_resolution_clock::now();
	  if (opts_->print_level > 1)
	    std::cout << "generation kernel of " << nevents << " events took " << std::chrono::duration<double, std::milli>(t_after_finish-t_before_launch).count() << " ms in total" << std::endl;
	  //copy results back
	  block_.CopyFromOutputBuffer(result);
	}
      else if (opts_->rndtype == generator_options::randomization_type::on_host)//random number generation on host, only pdf evaluation on accelerator
	{
	  result.resize(nevents);
	  unsigned int chunksize = nevents * opts_->chunksize_multiplier;//the chunks in which events are generated, could also use some kind of heuristic
	  unsigned int localworksize = backend_->required_padding();
	  if (localworksize > 0 && chunksize % localworksize != 0)
	    {
	      int chunksize_before = chunksize;
	      chunksize = (int(chunksize / localworksize) + 1)*localworksize;
	      std::cout << "Warning: chunksize of " << chunksize_before << " is not a multiple of work group size " << localworksize << ", automatically padded chunksize to " << chunksize << std::endl;
	      result.set_padding(true, localworksize);
	    }
	  EventVector<kernelT, evalT> gen_buffer(result);
	  gen_buffer.resize(chunksize);
	  if (localworksize > 0)
	    result.set_padding(true, localworksize);
	  dimension res("res", 0.0, maxprob);
	  EventVector<kernelT, evalT> res_buffer({&res});
	  res_buffer.resize(chunksize);
	  if (localworksize > 0)
	    res_buffer.set_padding(true, localworksize);
	  //set up buffers
	  block_.SetupInputBuffer(gen_buffer.buffer_size());
	  block_.SetupOutputBuffer(res_buffer.buffer_size());
	  block_.SetNevents(res_buffer.nevents(), res_buffer.nevents_padded());
	  if (!regen)
	    {
	      //prepare kernel
	      std::vector<std::string> param_names;
	      std::vector<evalT> param_values;
	      for (auto param : params)
		{
		  param_names.push_back(param->get_name());
		  param_values.push_back(param->get_value());      
		}
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> graphs;
	      graphs.emplace_back(std::move(prob->prob_normalised()->substitute(param_names, param_values)->simplify()));
	      std::vector<std::string> dummy;
	      block_.MakeComputeKernel("gen_kernel", chunksize, gen_buffer.copy_dimensions(), res_buffer.copy_dimensions(), dummy, graphs);
	    }
	  block_.Finish();      
	  bool finished = false;
	  int chunkid = 0;
	  unsigned int naccepted = 0;
	  while (!finished)
	    {
	      //accept-reject in all dimensions
	      for (unsigned int i=0; i<result.get_dimensions().size(); i++)
		{
		  evalT min = result.get_dimensions().at(i)->get_min();
		  evalT max = result.get_dimensions().at(i)->get_max();
		  evalT delta = max-min;
		  for (unsigned int j=0; j<chunksize; j++)	    
		    gen_buffer(j,i) = min + delta * rnd_->random();
		}
	      //copy random events to input buffer
	      block_.CopyToInputBuffer(gen_buffer);
	      //submit kernel to calculate probs
	      block_.SubmitKernel();
	      block_.Finish();	  
	      //copy resulting probabilities back
	      block_.CopyFromOutputBuffer(res_buffer);
	      //accept-reject using maximal probability, copy to result buffer if accepted
	      for (unsigned int i=0; i<chunksize; i++)
		{
		  if (maxprob*rnd_->random() < res_buffer(i,0))
		    {
		      for (unsigned int j=0; j<gen_buffer.ndimensions(); j++)
			result(naccepted,j) = gen_buffer(i,j);
		      naccepted++;
		      if (naccepted >= nevents)
			{
			  finished = true;
			  break;
			}
		    }
		}
	      chunkid++;
	      if (opts_->print_level > 2)
		std::cout << "after chunk " << chunkid << " accepted "<< naccepted << " events in total (target "<< nevents << " events), accept rate " << naccepted/double(chunkid*chunksize) << std::endl;
	    }
	}
      else
	assert(0);
      auto t_after_generation = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "generation of " << nevents << " events took " << std::chrono::duration<double, std::milli>(t_after_generation-t_before_generation).count() << " ms in total" << std::endl;
      return true;
    }
  };

}

#endif
