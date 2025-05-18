/**
 * @file fitter.hh
 * @author Christoph Langenbruch
 * @date 2024-11-16
 *
 */

#ifndef FITTER_H
#define FITTER_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include <array>
#include <set>
#include <math.h>
#include <eigen3/Eigen/Dense>

#ifdef WITH_ROOT
#include "TMinuit.h"
#endif
#include "minuit2/inc/Minuit2/Minuit2Minimizer.h"

#include "graph.hh"
#include "eventvector.hh"
#include "parametervector.hh"
#include "pdf.hh"
#include "physicspdfs.hh"
#include "compute.hh"
#include "utils.hh"

namespace morefit {

  struct fitter_options {
    enum minimizer_type { TMinuit, Minuit2};
    minimizer_type minimizer;
    bool analytic_gradient;
    bool analytic_hessian;
    bool analytic_fisher;
    bool optimize_parameters;
    bool optimize_dimensions;
    bool kahan_on_accelerator;
    int print_level;
    bool correct_weighted_fit;
    //minuit options
    double minuit_strategy;
    int minuit_printlevel;
    double minuit_maxiterations;
    double minuit_maxcalls;
    double minuit_tolerance;
    double minuit_errordef;
    float buffering_cost_threshold;
    bool postrun_hesse;
    fitter_options():
      minimizer(minimizer_type::Minuit2),
      analytic_gradient(true),
      analytic_hessian(true),
      analytic_fisher(false),
      optimize_parameters(true),
      optimize_dimensions(false),
      kahan_on_accelerator(true),
      print_level(2),
      correct_weighted_fit(true),
      minuit_strategy(2),
      minuit_printlevel(1),
      minuit_maxiterations(4000),
      minuit_maxcalls(4000),
      minuit_tolerance(0.1),
      minuit_errordef(1.0),
      buffering_cost_threshold(2.0),
      postrun_hesse(true)
    {}
    void print() {
      const unsigned int width = 40;      
      std::cout << "FITTER OPTIONS" << std::endl;
      std::cout << "General:" << std::endl;
      std::cout << std::setw(width) << std::left << "  minimizer type ";
      switch (minimizer) {
      case minimizer_type::TMinuit: std::cout << "TMinuit" << std::endl; break;
      case minimizer_type::Minuit2: std::cout << "Minuit2" << std::endl; break;
      default: std::cout << "UNKNOWN" << std::endl;
      }
      std::cout << std::setw(width) << std::left << "  use analytic gradient " << (analytic_gradient ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  use analytic hessian " << (analytic_hessian ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  calculate analytic fisher information " << (analytic_fisher ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  optimize parameters " << (optimize_parameters ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  optimize dimensions " << (optimize_dimensions ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  kahan on accelerator " << (kahan_on_accelerator ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  print level " << print_level << std::endl;
      std::cout << std::setw(width) << std::left << "  correct weighted fit " << (correct_weighted_fit ? "YES" : "NO") << std::endl;
      std::cout << "Minuit:" << std::endl;
      std::cout << std::setw(width) << std::left << "  minuit strategy " << minuit_strategy << std::endl;
      std::cout << std::setw(width) << std::left << "  minuit print level " << minuit_printlevel << std::endl;
      std::cout << std::setw(width) << std::left << "  minuit max. iterations " << minuit_maxiterations << std::endl;
      std::cout << std::setw(width) << std::left << "  minuit max. calls " << minuit_maxcalls << std::endl;
      std::cout << std::setw(width) << std::left << "  minuit tolerance " << minuit_tolerance << std::endl;
      std::cout << std::setw(width) << std::left << "  minuit error definition " << minuit_errordef << std::endl;
      std::cout << std::setw(width) << std::left << "  buffering cost threashold " << buffering_cost_threshold << std::endl;
      std::cout << std::setw(width) << std::left << "  postrun Hesse " << (postrun_hesse ? "YES" : "NO") << std::endl;
      
    }
  };

  //forward definition
  template<typename kernelT, typename evalT, typename backendT, typename computeT> 
  class fitter;

  //class encapsulating likelihood and gradient calculation needed for minuit2
  template<typename kernelT, typename evalT, typename backendT, typename computeT> 
  class minuit2_grad_fcn : public ROOT::Math::IGradientFunctionMultiDimTempl<evalT> {
  private:
    typedef ROOT::Math::IBaseFunctionMultiDimTempl<evalT> BaseFunc;
    fitter<kernelT, evalT, backendT, computeT>* local_fitter_pointer;
  public:
    minuit2_grad_fcn(fitter<kernelT, evalT, backendT, computeT>* fitter):
      local_fitter_pointer(fitter)
    {}
    virtual ROOT::Math::IBaseFunctionMultiDimTempl<evalT> *Clone() const override
    {
      return new minuit2_grad_fcn<kernelT, evalT, backendT, computeT>(local_fitter_pointer);
    }
    virtual unsigned int NDim() const override
    {
      //all parameters including fixed params
      return local_fitter_pointer->params_.size();
    }
    virtual evalT DoEval(const evalT *x) const override
    {
      int ndim = NDim();
      local_fitter_pointer->update_parameters(ndim, x);
      return local_fitter_pointer->likelihood();
    }
    //grad methods
    virtual void Gradient(const evalT *x, evalT *grad) const override
    {
      int ndim = NDim();
      local_fitter_pointer->update_parameters(ndim, x);
      local_fitter_pointer->likelihood_and_gradient(ndim, grad);
      return;
    }
    virtual void FdF(const evalT *x, evalT &f, evalT *df) const override
    {
      int ndim = NDim();
      local_fitter_pointer->update_parameters(ndim, x);
      evalT lh = local_fitter_pointer->likelihood_and_gradient(ndim, df);
      f = lh;
      return;
    }
    //analytic Hessian, needs to be set explicitly
    bool HessianFunc(std::vector<evalT> x, evalT* h)
    {
      int ndim = NDim();
      //params array is actually as long as the number of defined parameters, does include the fixed parameters
      //npar is the number of floating parameters, different from the size of params (and the field params_)
      local_fitter_pointer->update_parameters(ndim, &x[0]);
      return local_fitter_pointer->hessian(ndim, h);
    }
  private:
    virtual evalT DoDerivative(const evalT *x, unsigned int icoord) const override //needs to be overwritten even if not efficient
    {
      evalT grad[NDim()];
      Gradient(x, grad);
      return grad[icoord];
    }
  };
  
  //class encapsulating likelihood calculation needed for minuit2
  template<typename kernelT, typename evalT, typename backendT, typename computeT> 
  class minuit2_fcn : public ROOT::Math::IBaseFunctionMultiDimTempl<evalT> {
  private:
    fitter<kernelT, evalT, backendT, computeT>* local_fitter_pointer;
  public:
    minuit2_fcn(fitter<kernelT, evalT, backendT, computeT>* fitter):
      local_fitter_pointer(fitter)
    {}
    virtual ROOT::Math::IBaseFunctionMultiDimTempl<evalT> *Clone() const override
    {
      return new minuit2_fcn<kernelT, evalT, backendT, computeT>(local_fitter_pointer);
    }
    virtual unsigned int NDim() const override
    {
      //all parameters including fixed params
      return local_fitter_pointer->params_.size();
    }
    virtual evalT DoEval(const evalT *x) const override
    {
      int ndim = NDim();
      local_fitter_pointer->update_parameters(ndim, x);
      return local_fitter_pointer->likelihood();
    }
  };

  //fitter class taking pdf, parameters and events as arguments
  template<typename kernelT, typename evalT, typename backendT, typename computeT> 
  class fitter {
  private:
    friend class minuit2_grad_fcn<kernelT, evalT, backendT, computeT>;
    friend class minuit2_fcn<kernelT, evalT, backendT, computeT>;
    fitter_options* opts_;
    computeT block_;
    computeT precompute_block_;
    computeT grad_block_;
    computeT hessian_block_;
    computeT fisher_block_;
    backendT* backend_;
    dimension<evalT> res_dim_;
    std::vector<dimension<evalT>> grad_res_dim_;
    std::vector<dimension<evalT>> hessian_res_dim_;
    std::vector<dimension<evalT>> fisher_res_dim_;
    EventVector<kernelT, evalT> res_buffer_;
    EventVector<kernelT, evalT> grad_res_buffer_;
    EventVector<kernelT, evalT> hessian_res_buffer_;
    EventVector<kernelT, evalT> fisher_res_buffer_;
    std::vector<parameter<evalT>*> params_;
    PDF<kernelT, evalT>* pdf_;
    EventVector<kernelT, evalT>* data_;
    std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> buffer_expressions_;
    std::vector<std::string> buffer_names_;
    std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> grad_buffer_expressions_;
    std::vector<std::string> grad_buffer_names_;
    std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> event_buffer_expressions_;
    std::vector<std::string> event_buffer_names_;
    std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> hessian_buffer_expressions_;
    std::vector<std::string> hessian_buffer_names_;
    std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> fisher_buffer_expressions_;
    std::vector<std::string> fisher_buffer_names_;
    //pointer for global fitter object needed for minuit_one
    static fitter<kernelT, evalT, backendT, computeT>* global_fitter_pointer;
    //temporary, to be moved to compute_block class
    int precompute_output_dimensions_{0};
#ifdef WITH_ROOT
    //minuit one fitter object (available only if linked against root)
    TMinuit* minuit_one;
    //static minuit one function, minuit_two does not need a global function pointer
    static void minuit_one_fcn(Int_t &npar, Double_t *grad, Double_t &lh, Double_t *params, Int_t iflag)
    {
      global_fitter_pointer->update_parameters(npar, params);      
      if (iflag == 2)//gradients
	lh = global_fitter_pointer->likelihood_and_gradient(npar, grad);
      else
	lh = global_fitter_pointer->likelihood();
      return;
    }
#endif
    //minuit2 fitter object (available standalone)
    ROOT::Minuit2::Minuit2Minimizer* minuit_two;
    minuit2_grad_fcn<kernelT, evalT, backendT, computeT>* minuit_two_grad_fcn;
    minuit2_fcn<kernelT, evalT, backendT, computeT>* minuit_two_fcn;
    //updating parameters to the new values from the minimizer 
    void update_parameters(int &npar, const double *params)
    {
      //params array is actually as long as the number of defined parameters, does include the fixed parameters
      //npar is the number of floating parameters, different from the size of params (and the field params_)
      for (unsigned int i=0; i<params_.size(); i++)
	params_.at(i)->set_value(params[i]);
      return;
    }
  public:
    fitter(fitter_options* options, backendT* backend):
      opts_(options),
      block_(computeT(backend)),
      precompute_block_(computeT(backend)),
      grad_block_(computeT(backend)),
      hessian_block_(computeT(backend)),
      fisher_block_(computeT(backend)),
      backend_(backend),
      res_dim_("res", 0.0, 1.0),
      res_buffer_({&res_dim_}),
      grad_res_buffer_({&res_dim_}),
      hessian_res_buffer_(),
      fisher_res_buffer_()
    {
      if (opts_->minimizer == fitter_options::minimizer_type::TMinuit)
	{
#ifdef WITH_ROOT
	  minuit_one = new TMinuit(1000);
	  minuit_one->SetFCN(&(minuit_one_fcn));
#else
	  std::cout << "Cannot use TMinuit when compiling without ROOT support." << std::endl;
	  assert(0);
#endif
	}
      if (opts_->minimizer == fitter_options::minimizer_type::Minuit2)
	{
	  minuit_two = new ROOT::Minuit2::Minuit2Minimizer(ROOT::Minuit2::EMinimizerType::kMigrad);
	  if (opts_->analytic_gradient)
	    {
	      minuit_two_grad_fcn = new minuit2_grad_fcn<kernelT, evalT, backendT, computeT>(this);
	      const ROOT::Math::IMultiGradFunction *gfunc = dynamic_cast<const ROOT::Math::IMultiGradFunction *>(minuit_two_grad_fcn);
	      minuit_two->SetFunction(*gfunc);
	    }
	  else
	    {
	      minuit_two_fcn = new minuit2_fcn<kernelT, evalT, backendT, computeT>(this);
	      const ROOT::Math::IMultiGenFunction *func = dynamic_cast<const ROOT::Math::IMultiGenFunction *>(minuit_two_fcn);
	      minuit_two->SetFunction(*func);	      
	    }
	}
    }
    ~fitter()
    {
#ifdef WITH_ROOT
      if (opts_->minimizer == fitter_options::minimizer_type::TMinuit)
	delete minuit_one;
#endif
      if (opts_->minimizer == fitter_options::minimizer_type::Minuit2)
	{
	  delete minuit_two;
	  if (opts_->analytic_gradient)
	    delete minuit_two_grad_fcn;
	  else
	    delete minuit_two_fcn;	    
	}
    }
    //generate kernels
    //could use member fields instead of arguments
    bool make_kernels(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data)
    {
      std::vector<std::string> fixed_params;
      std::vector<evalT> fixed_values;
      for (const auto * param : params)
	if (param->is_constant())
	  {
	    fixed_params.push_back(param->get_name());
	    fixed_values.push_back(param->get_value());
	  }
      std::vector<std::string> floating_params;
      for (auto param : params)
	if (!param->is_constant())
	  floating_params.push_back(param->get_name());
	  
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> logprob(pdf->logprob_normalised()->substitute(fixed_params, fixed_values)->simplify());
      if (data->event_weight_idx()!= -1)	
	logprob = std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT,evalT>>(data->event_weight_name()), std::move(logprob));
      //perform precalculations depending only on event, adding these as extra dimensions
      event_buffer_expressions_.clear();
      event_buffer_names_.clear();
      std::vector<std::string> dimensions = data->get_dimensions_str();
      unsigned int precompute_buffer_size = 0;
      std::vector<dimension<evalT>> precompute_output_dimensions(data->copy_dimensions());
      if (opts_->optimize_dimensions)
	{
	  for (unsigned int i=0; i<dimensions.size(); i++)
	    {
	      event_buffer_names_.push_back(dimensions.at(i));
	      event_buffer_expressions_.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(dimensions.at(i)));
	    }
	  logprob = std::move(logprob->optimize_buffering_constant_terms(event_buffer_names_, event_buffer_expressions_, dimensions, "morefit_eventbuffer_", opts_->buffering_cost_threshold));
	  
	  if (opts_->print_level > 1)
	    {
	      std::cout << std::endl;
	      for (unsigned int i=0; i<event_buffer_expressions_.size(); i++)
		std::cout << "EVENTBUFFER " << event_buffer_names_.at(i) << " = " << event_buffer_expressions_.at(i)->get_kernel() << std::endl;
	      std::cout << std::endl;
	    }
	  //run kernel
	  precompute_block_.SetupInputBuffer(data->buffer_size());
	  std::vector<std::string> dummy;
	  for (unsigned int i=data->get_dimensions().size(); i<event_buffer_names_.size(); i++)//add additional variables to buffer
	    precompute_output_dimensions.push_back(dimension<evalT>(event_buffer_names_.at(i)));
	  precompute_output_dimensions_ = precompute_output_dimensions.size();//TODO save dimension (input+output signature) in compute_block class
	  precompute_buffer_size = precompute_output_dimensions.size()*sizeof(kernelT)*data->nevents_padded();
	  precompute_block_.SetupOutputBuffer(precompute_buffer_size);
	  auto t_before_precompute_kernel = std::chrono::high_resolution_clock::now();
	  precompute_block_.MakeComputeKernel("precompute_kernel", data->nevents(), data->copy_dimensions(), precompute_output_dimensions, dummy, event_buffer_expressions_);
	  precompute_block_.Finish();
	  auto t_after_precompute_kernel = std::chrono::high_resolution_clock::now();
	  if (opts_->print_level > 1)
	    std::cout << "making precompute kernel takes " << std::chrono::duration<double, std::milli>(t_after_precompute_kernel-t_before_precompute_kernel).count() << " ms in total" << std::endl;
	}
      
      //potential optimisations kernel on expressions depending only on parameters
      buffer_expressions_.clear();
      buffer_names_.clear();
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> graphs;
      if (opts_->optimize_parameters)
	{
	  graphs.emplace_back(std::move(logprob->optimize_buffering_constant_terms(buffer_names_, buffer_expressions_, floating_params, "morefit_parambuffer_", opts_->buffering_cost_threshold)));	  
	  //output check
	  if (opts_->print_level > 1)	    
	    {
	      std::cout << std::endl;
	      for (unsigned int i=0; i<graphs.size(); i++)
		std::cout << "GRAPHS: " << graphs.at(i)->get_kernel() << std::endl;
	      for (unsigned int i=0; i< buffer_expressions_.size(); i++)
		std::cout << "BUFFER " << buffer_names_.at(i) << " = " << buffer_expressions_.at(i)->get_kernel() << std::endl;
	      std::cout << std::endl;
	    }
	}
      else
	graphs.emplace_back(std::move(logprob->copy()));
      //analogously for gradient calculation
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> grad_graphs;
      if (opts_->analytic_gradient)
	{
	  grad_buffer_expressions_.clear();
	  grad_buffer_names_.clear();
	  if (opts_->optimize_parameters)
	    {
	      grad_graphs.emplace_back(std::move(logprob->optimize_buffering_constant_terms(grad_buffer_names_, grad_buffer_expressions_, floating_params, "morefit_parambuffer_", opts_->buffering_cost_threshold)));
	      for (unsigned int i=0; i<floating_params.size(); i++)
		grad_graphs.emplace_back(std::move(logprob->diff(floating_params.at(i))->simplify()->optimize_buffering_constant_terms(grad_buffer_names_, grad_buffer_expressions_, floating_params, "morefit_parambuffer_", opts_->buffering_cost_threshold)));
	      //output check
	      if (opts_->print_level > 1)
		{
		  std::cout << std::endl;
		  for (unsigned int i=0; i<grad_graphs.size(); i++)
		    std::cout << "GRADGRAPH: " << grad_graphs.at(i)->get_kernel() << std::endl;
		  for (unsigned int i=0; i<grad_buffer_expressions_.size(); i++)
		    std::cout << "GRADBUFFER: " << grad_buffer_names_.at(i) << " = " << grad_buffer_expressions_.at(i)->get_kernel() << std::endl;
		  std::cout << std::endl;
		}
	    }
	  else
	    {
	      grad_graphs.emplace_back(std::move(logprob->copy()));
	      for (unsigned int i=0; i<floating_params.size(); i++)
		grad_graphs.emplace_back(std::move(logprob->diff(floating_params.at(i))->simplify()));
	    }
	}
      //analytic hessian calculation
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> hessian_graphs;
      if (opts_->analytic_hessian)
	{
	  hessian_buffer_expressions_.clear();
	  hessian_buffer_names_.clear();
	  if (opts_->optimize_parameters)
	    {
	      for (unsigned int i=0; i<floating_params.size(); i++)
		{
		  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diffi(std::move(logprob->diff(floating_params.at(i))));
		  for (unsigned int j=0; j<floating_params.size(); j++)
		    if (i>=j)
		      hessian_graphs.emplace_back(std::move(diffi->diff(floating_params.at(j))->simplify()->optimize_buffering_constant_terms(hessian_buffer_names_, hessian_buffer_expressions_, floating_params, "morefit_parambuffer_", opts_->buffering_cost_threshold)));
		}
	      //output check
	      if (opts_->print_level > 1)
		{
		  std::cout << std::endl;
		  for (unsigned int i=0; i<hessian_graphs.size(); i++)
		    std::cout << "HESSIANGRAPH: " << hessian_graphs.at(i)->get_kernel() << std::endl;
		  for (unsigned int i=0; i<hessian_buffer_expressions_.size(); i++)
		    std::cout << "HESSIANBUFFER: " << hessian_buffer_names_.at(i) << " = " << hessian_buffer_expressions_.at(i)->get_kernel() << std::endl;
		  std::cout << std::endl;
		}
	    }
	  else
	    {
	      for (unsigned int i=0; i<floating_params.size(); i++)
		{
		  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diffi(std::move(logprob->diff(floating_params.at(i))));
		  for (unsigned int j=0; j<floating_params.size(); j++)
		    if (i>=j)
		      hessian_graphs.emplace_back(std::move(diffi->diff(floating_params.at(j))->simplify()));
		}
	      //output check
	      if (opts_->print_level > 1)
		{
		  std::cout << std::endl;
		  for (unsigned int i=0; i<hessian_graphs.size(); i++)
		    std::cout << "HESSIANGRAPH: " << hessian_graphs.at(i)->get_kernel() << std::endl;
		  std::cout << std::endl;
		}
	    }
	}
      //analytic fisher information matrix calculation
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> fisher_graphs;
      if (opts_->analytic_fisher)
	{
	  fisher_buffer_expressions_.clear();
	  fisher_buffer_names_.clear();
	  if (opts_->optimize_parameters)
	    {
	      for (unsigned int i=0; i<floating_params.size(); i++)
		{
		  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diffi(std::move(logprob->diff(floating_params.at(i))));
		  for (unsigned int j=0; j<floating_params.size(); j++)
		    {
		      if (i>=j)
			{
			  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diffj(std::move(logprob->diff(floating_params.at(j))));		      
			  fisher_graphs.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::move(diffi->copy()), std::move(diffj))->simplify()->optimize_buffering_constant_terms(fisher_buffer_names_, fisher_buffer_expressions_, floating_params, "morefit_parambuffer_", opts_->buffering_cost_threshold));
			}
		    }
		}
	      //output check
	      if (opts_->print_level > 1)
		{
		  std::cout << std::endl;
		  for (unsigned int i=0; i<fisher_graphs.size(); i++)
		    std::cout << "FISHERGRAPH: " << fisher_graphs.at(i)->get_kernel() << std::endl;
		  for (unsigned int i=0; i<fisher_buffer_expressions_.size(); i++)
		    std::cout << "FISHERBUFFER: " << fisher_buffer_names_.at(i) << " = " << fisher_buffer_expressions_.at(i)->get_kernel() << std::endl;
		  std::cout << std::endl;
		}
	    }
	  else
	    {
	      for (unsigned int i=0; i<floating_params.size(); i++)
		{
		  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diffi(std::move(logprob->diff(floating_params.at(i))));
		  for (unsigned int j=0; j<floating_params.size(); j++)
		    if (i>=j)
		      {
			std::unique_ptr<ComputeGraphNode<kernelT, evalT>> diffj(std::move(logprob->diff(floating_params.at(j))));		      
			fisher_graphs.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::move(diffi->copy()), std::move(diffj))->simplify());
		      }
		}
	      //output check
	      if (opts_->print_level > 1)
		{
		  std::cout << std::endl;
		  for (unsigned int i=0; i<fisher_graphs.size(); i++)
		    std::cout << "FISHERGRAPH: " << fisher_graphs.at(i)->get_kernel() << std::endl;
		  std::cout << std::endl;
		}
	    }
	}

      //set up buffers
      evalT maxprob = pdf->get_max();
      res_dim_.set_max(maxprob);
      res_buffer_.set_padding(data->is_padded(), data->padding());
      res_buffer_.resize(data->nevents());      
      if (opts_->optimize_dimensions)
	block_.SetupInputBuffer(&precompute_block_, false);
      else	
	block_.SetupInputBuffer(data->buffer_size());
      if (opts_->kahan_on_accelerator)
	block_.SetupKahanBuffer(sizeof(evalT));
      
      block_.SetupOutputBuffer(res_buffer_.buffer_size());
      block_.SetupParameterBuffer((floating_params.size()+buffer_expressions_.size())*sizeof(kernelT));
      if (opts_->analytic_gradient)
	{
	  //this makes sure that the pointers are not lost and no memory is leaked
	  grad_res_buffer_.set_padding(data->is_padded(), data->padding());
	  for (auto variable : floating_params)
	    grad_res_dim_.push_back(dimension<evalT>("dlnL/d"+variable, -1.0, 1.0));
	  std::vector<dimension<evalT>*> arg;
	  for (unsigned int i=0; i<grad_res_dim_.size(); i++)
	    arg.push_back(&grad_res_dim_.at(i));
	  grad_res_buffer_.add_dimensions(arg);
	  grad_res_buffer_.resize(data->nevents());
	  if (opts_->optimize_dimensions)
	    grad_block_.SetupInputBuffer(&precompute_block_, false);
	  else
	    grad_block_.SetupInputBuffer(&block_, true);
	  grad_block_.SetupOutputBuffer(grad_res_buffer_.buffer_size());//still using this with kahan summation, does not seem to negatively affect performance
	  grad_block_.SetupParameterBuffer((floating_params.size()+grad_buffer_expressions_.size())*sizeof(kernelT));
	  if (opts_->kahan_on_accelerator)
	    grad_block_.SetupKahanBuffer(grad_res_buffer_.ndimensions()*sizeof(evalT));
	}
      if (opts_->analytic_hessian)
	{
	  //this makes sure that the pointers are not lost and no memory is leaked
	  hessian_res_buffer_.set_padding(data->is_padded(), data->padding());
	  for (unsigned int i=0; i<floating_params.size(); i++)
	    for (unsigned int j=0; j<floating_params.size(); j++)
	      if (i >= j)
		hessian_res_dim_.push_back(dimension<evalT>("d2lnL/d"+floating_params.at(i)+"d"+floating_params.at(j), -1.0, 1.0));
	  std::vector<dimension<evalT>*> arg;
	  for (unsigned int i=0; i<hessian_res_dim_.size(); i++)
	    arg.push_back(&hessian_res_dim_.at(i));
	      
	  hessian_res_buffer_.add_dimensions(arg);
	  hessian_res_buffer_.resize(data->nevents());
	  if (opts_->optimize_dimensions)
	    hessian_block_.SetupInputBuffer(&precompute_block_, false);
	  else
	    hessian_block_.SetupInputBuffer(&block_, true);
	  hessian_block_.SetupOutputBuffer(hessian_res_buffer_.buffer_size());//still using this with kahan summation, does not seem to negatively affect performance
	  hessian_block_.SetupParameterBuffer((floating_params.size()+hessian_buffer_expressions_.size())*sizeof(kernelT));
	  if (opts_->kahan_on_accelerator)
	    hessian_block_.SetupKahanBuffer(hessian_res_buffer_.ndimensions()*sizeof(evalT));
	}
      if (opts_->analytic_fisher)
	{
	  //this makes sure that the pointers are not lost and no memory is leaked
	  fisher_res_buffer_.set_padding(data->is_padded(), data->padding());
	  for (unsigned int i=0; i<floating_params.size(); i++)
	    for (unsigned int j=0; j<floating_params.size(); j++)
	      if (i >= j)
		fisher_res_dim_.push_back(dimension<evalT>("dlnL/d"+floating_params.at(i)+"*dlnL/d"+floating_params.at(j), -1.0, 1.0));
	  std::vector<dimension<evalT>*> arg;
	  for (unsigned int i=0; i<fisher_res_dim_.size(); i++)
	    arg.push_back(&fisher_res_dim_.at(i));
	      
	  fisher_res_buffer_.add_dimensions(arg);
	  fisher_res_buffer_.resize(data->nevents());
	  if (opts_->optimize_dimensions)
	    fisher_block_.SetupInputBuffer(&precompute_block_, false);
	  else
	    fisher_block_.SetupInputBuffer(&block_, true);
	  fisher_block_.SetupOutputBuffer(fisher_res_buffer_.buffer_size());//still using this with kahan summation, does not seem to negatively affect performance
	  fisher_block_.SetupParameterBuffer((floating_params.size()+fisher_buffer_expressions_.size())*sizeof(kernelT));
	  if (opts_->kahan_on_accelerator)
	    fisher_block_.SetupKahanBuffer(fisher_res_buffer_.ndimensions()*sizeof(evalT));
	}
	  
      std::vector<std::string> paramnames(floating_params);
      for (auto buffer_name : buffer_names_)
	paramnames.push_back(buffer_name);

      //make kernel
      auto t_before_kernel = std::chrono::high_resolution_clock::now();
      block_.MakeComputeKernel("lh_kernel", data->nevents(), opts_->optimize_dimensions ? precompute_output_dimensions : data->copy_dimensions(), res_buffer_.copy_dimensions(), paramnames, graphs, opts_->kahan_on_accelerator);
      block_.Finish();
      auto t_after_kernel = std::chrono::high_resolution_clock::now();

      if (opts_->print_level > 1)
	std::cout << "making lh kernel takes " << std::chrono::duration<double, std::milli>(t_after_kernel-t_before_kernel).count() << " ms in total" << std::endl;
      if (opts_->analytic_gradient)
	{
	  std::vector<std::string> grad_paramnames(floating_params);
	  for (auto buffer_name : grad_buffer_names_)
	    grad_paramnames.push_back(buffer_name);
	  auto t_before_grad_kernel = std::chrono::high_resolution_clock::now();
	  grad_block_.MakeComputeKernel("lh_grad_kernel", data->nevents(), opts_->optimize_dimensions ? precompute_output_dimensions : data->copy_dimensions(), grad_res_buffer_.copy_dimensions(), grad_paramnames, grad_graphs, opts_->kahan_on_accelerator);
	  grad_block_.Finish();

	  auto t_after_grad_kernel = std::chrono::high_resolution_clock::now();
	  if (opts_->print_level > 1)
	    std::cout << "making lh+grad kernel takes " << std::chrono::duration<double, std::milli>(t_after_grad_kernel-t_before_grad_kernel).count() << " ms in total" << std::endl;
	}
      if (opts_->analytic_hessian)
	{
	  std::vector<std::string> hessian_paramnames(floating_params);
	  for (auto buffer_name : hessian_buffer_names_)
	    hessian_paramnames.push_back(buffer_name);
	  auto t_before_hessian_kernel = std::chrono::high_resolution_clock::now();
	  hessian_block_.MakeComputeKernel("lh_hessian_kernel", data->nevents(), opts_->optimize_dimensions ? precompute_output_dimensions : data->copy_dimensions(), hessian_res_buffer_.copy_dimensions(), hessian_paramnames, hessian_graphs, opts_->kahan_on_accelerator);
	  hessian_block_.Finish();

	  auto t_after_hessian_kernel = std::chrono::high_resolution_clock::now();
	  if (opts_->print_level > 1)
	    std::cout << "making hessian kernel takes " << std::chrono::duration<double, std::milli>(t_after_hessian_kernel-t_before_hessian_kernel).count() << " ms in total" << std::endl;
	}
      if (opts_->analytic_fisher)
	{
	  std::vector<std::string> fisher_paramnames(floating_params);
	  for (auto buffer_name : fisher_buffer_names_)
	    fisher_paramnames.push_back(buffer_name);
	  auto t_before_fisher_kernel = std::chrono::high_resolution_clock::now();
	  fisher_block_.MakeComputeKernel("lh_fisher_kernel", data->nevents(), opts_->optimize_dimensions ? precompute_output_dimensions : data->copy_dimensions(), fisher_res_buffer_.copy_dimensions(), fisher_paramnames, fisher_graphs, opts_->kahan_on_accelerator);
	  fisher_block_.Finish();

	  auto t_after_fisher_kernel = std::chrono::high_resolution_clock::now();
	  if (opts_->print_level > 1)
	    std::cout << "making fisher kernel takes " << std::chrono::duration<double, std::milli>(t_after_fisher_kernel-t_before_fisher_kernel).count() << " ms in total" << std::endl;
	}
      return true;
    }
    //fit data
    bool fit(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data, bool refit = false)
    {      
      auto t_before_fit = std::chrono::high_resolution_clock::now();
      global_fitter_pointer = this;

      //save pointers to arguments, needed for external likelihood calculation
      pdf_ = pdf;
      params_ = params;
      data_ = data;

      //fixed parameters need to be replaced by constants, this means that the kernel/parameter vectors only include floating parameters
      std::vector<std::string> fixed_params;
      std::vector<evalT> fixed_values;
      for (const auto * param : params)
	if (param->is_constant())
	  {
	    fixed_params.push_back(param->get_name());
	    fixed_values.push_back(param->get_value());
	  }
      std::vector<std::string> floating_params;
      for (auto param : params)
	if (!param->is_constant())
	  floating_params.push_back(param->get_name());

      //only needs to be redone if nevents changes (extended fit)
      if (backend_->required_padding() > 0 && data->nevents_padded() % backend_->required_padding() != 0)
	{
	  std::cout << "buffer size before " << data->buffer_size() << std::endl;
	  int nbefore = data->nevents_padded();
	  data->set_padding(true, backend_->required_padding());
	  int nafter = data->nevents_padded();
	  std::cout << "Kernel needs to apply additional padding (to " << backend_->required_padding() << ") to input data, changing from " << nbefore << " to " << nafter << " (padded) events" << std::endl;
	  std::cout << "buffer size after " << data->buffer_size() << ", " << data->get_data() << std::endl;
	}
      if (backend_->required_padding() == 0 && data->is_padded())
	{
	  std::cout << "backend does not require padding but input data is padded." << std::endl;
	  assert(0);
	}
      //make the kernel only if necessary 
      if (!refit)
	make_kernels(pdf, params, data);
      
      //copy actual data
      if (opts_->optimize_dimensions)
	{
	  precompute_block_.SetupInputBuffer(data->buffer_size());
	  precompute_block_.SetupOutputBuffer(precompute_output_dimensions_*sizeof(kernelT)*data->nevents_padded());
	  precompute_block_.SetNevents(data->nevents(), data->nevents_padded());
	  precompute_block_.CopyToInputBuffer(*data);
	  precompute_block_.SubmitKernel();
	  precompute_block_.Finish();
	}
      
      res_buffer_.resize(data->nevents());
      if (opts_->optimize_dimensions)
	block_.SetupInputBuffer(&precompute_block_, false);
      else	
	block_.SetupInputBuffer(data->buffer_size());      
      block_.SetupOutputBuffer(res_buffer_.buffer_size());
      block_.SetNevents(data->nevents(), data->nevents_padded());
      
      if (opts_->analytic_gradient)
	{
	  grad_res_buffer_.set_padding(data->is_padded(), data->padding());
	  grad_res_buffer_.resize(data->nevents());
	  if (opts_->optimize_dimensions)
	    grad_block_.SetupInputBuffer(&precompute_block_, false);
	  else
	    grad_block_.SetupInputBuffer(&block_, true);
	  grad_block_.SetupOutputBuffer(grad_res_buffer_.buffer_size());//still using this with kahan summation, does not seem to negatively affect performance
	  grad_block_.SetNevents(data->nevents(), data->nevents_padded());
	}
      if (opts_->analytic_hessian)
	{
	  hessian_res_buffer_.set_padding(data->is_padded(), data->padding());
	  hessian_res_buffer_.resize(data->nevents());
	  if (opts_->optimize_dimensions)
	    hessian_block_.SetupInputBuffer(&precompute_block_, false);
	  else
	    hessian_block_.SetupInputBuffer(&block_, true);
	  hessian_block_.SetupOutputBuffer(hessian_res_buffer_.buffer_size());//still using this with kahan summation, does not seem to negatively affect performance
	  hessian_block_.SetNevents(data->nevents(), data->nevents_padded());
	}
      if (opts_->analytic_fisher)
	{
	  fisher_res_buffer_.set_padding(data->is_padded(), data->padding());
	  fisher_res_buffer_.resize(data->nevents());
	  if (opts_->optimize_dimensions)
	    fisher_block_.SetupInputBuffer(&precompute_block_, false);
	  else
	    fisher_block_.SetupInputBuffer(&block_, true);
	  fisher_block_.SetupOutputBuffer(fisher_res_buffer_.buffer_size());//still using this with kahan summation, does not seem to negatively affect performance
	  fisher_block_.SetNevents(data->nevents(), data->nevents_padded());
	}

      if (!opts_->optimize_dimensions)
	block_.CopyToInputBuffer(*data);
            
      //setup minuit
      if (opts_->minimizer == fitter_options::minimizer_type::Minuit2)
	{
	  minuit_two->SetStrategy(opts_->minuit_strategy);
	  minuit_two->SetPrintLevel(opts_->minuit_printlevel);
	  minuit_two->SetMaxIterations(opts_->minuit_maxiterations);
	  minuit_two->SetMaxFunctionCalls(opts_->minuit_maxcalls);
	  minuit_two->SetErrorDef(opts_->minuit_errordef);
	  minuit_two->SetValidError(true);
	}
#ifdef WITH_ROOT
      if (opts_->minimizer == fitter_options::minimizer_type::TMinuit)
	{
	  int errorcode;
	  double strategy_list[1] = {opts_->minuit_strategy};
	  minuit_one->mnexcm("SET STR", strategy_list, 1, errorcode);
	  minuit_one->SetPrintLevel(opts_->minuit_printlevel);
	  minuit_one->SetMaxIterations(opts_->minuit_maxiterations);
	  minuit_one->SetErrorDef(opts_->minuit_errordef);
	  if (opts_->analytic_gradient)
	    {
	      double gradient_list[1] = {1.0};//provide our own analytic derivatives
	      minuit_one->mnexcm("SET GRA", gradient_list, 1, errorcode);
	    }
	  else
	    minuit_one->mnexcm("SET NOG", NULL, 0, errorcode);
	}
#endif      
      //set up parameters in minuit
      define_parameters(true);
      //prepare minuit fit
      int result;
      double tmp_cov[floating_params.size()*floating_params.size()];
      int nfree, ntot, status_cov;
      if (opts_->minimizer == fitter_options::minimizer_type::Minuit2)
	{
	  //need to set function for minuit2 at this time
	  if (opts_->analytic_gradient)
	    {
	      const ROOT::Math::IMultiGradFunction *gfunc = dynamic_cast<const ROOT::Math::IMultiGradFunction *>(minuit_two_grad_fcn);
	      minuit_two->SetFunction(*gfunc);
	    }
	  else
	    {
	      const ROOT::Math::IMultiGenFunction *func = dynamic_cast<const ROOT::Math::IMultiGenFunction *>(minuit_two_fcn);
	      minuit_two->SetFunction(*func);	      
	    }
	  if (opts_->analytic_hessian)
	    {
	      assert(opts_->analytic_gradient);
	      assert(opts_->minimizer == fitter_options::minimizer_type::Minuit2);
	      auto hessianFcn = [=](const std::vector<double> &x, double *hess) {
		return minuit_two_grad_fcn->HessianFunc(x, hess);
	      };
	      minuit_two->SetHessianFunction(hessianFcn);
	    }
	  minuit_two->Minimize();	  
	  if (opts_->postrun_hesse)
	    minuit_two->Hesse();
	  result = minuit_two->Status();
	  status_cov = minuit_two->CovMatrixStatus();
	  if (status_cov == 3)
	    std::cout << "Hesse returns " << status_cov << " -> ALL GOOD" << std::endl;
	  else
	    std::cout << "Hesse returns " << status_cov << " -> SOMETHING WRONG" << std::endl;
	  //extract covariance matrix
	  const auto& minuit_cov = minuit_two->State().Covariance();
	  for (unsigned int irow=0; irow<floating_params.size(); irow++)
	    {
	      for (unsigned int icol=0; icol<floating_params.size(); icol++)
		tmp_cov[floating_params.size()*irow+icol] = minuit_cov(irow, icol);
	    }
	  //crosscheck analytic fisher
	  if (false && opts_->analytic_fisher)
	    {
	      //call Fisher explicitly and calculate H^{-1} F H^{-1}
	      double f[params_.size()*params_.size()];
	      fisher(pdf, params, data, floating_params.size(), f);
	      std::cout << "Fisher information matrix:" << std::endl;
	      for (unsigned int i=0; i<params_.size(); i++)
		{
		  for (unsigned int j=0; j<params_.size(); j++)
		    std::cout << f[params_.size()*i+j] << " ";
		  std::cout << std::endl;
		}
	      double h[params_.size()*params_.size()];
	      hessian(pdf, params, data, floating_params.size(), h);
	      std::cout << "Hesse matrix for comparison:" << std::endl;
	      for (unsigned int i=0; i<params_.size(); i++)
		{
		  for (unsigned int j=0; j<params_.size(); j++)
		    std::cout << h[params_.size()*i+j] << " ";
		  std::cout << std::endl;
		}
	      Eigen::MatrixXd inv_f = Eigen::MatrixXd::Zero(floating_params.size(), floating_params.size());
	      for (unsigned int i=0; i<floating_params.size(); i++)
		for (unsigned int j=0; j<floating_params.size(); j++)
		  inv_f(i,j) = f[params_.size()*i+j];
	      inv_f = inv_f.inverse();
	      std::cout << "inverted Fisher information matrix:" << std::endl;
	      for (unsigned int i=0; i<floating_params.size(); i++)
		{
		  for (unsigned int j=0; j<floating_params.size(); j++)
		    std::cout << inv_f(i,j) << " ";
		  std::cout << std::endl;
		}
	    }
	  //debug output, crosscheck of analytic hessian	  
	  if (false && opts_->analytic_gradient)
	    {
	      assert(opts_->analytic_gradient);
	      std::cout << "MINUIT HESSIAN" <<  std::endl;
	      const auto & minuit_hessian = minuit_two->State().Hessian();
	      for (unsigned int irow=0; irow<floating_params.size(); irow++)
		{
		  for (unsigned int icol=0; icol<floating_params.size(); icol++)
		    std::cout << minuit_hessian(irow,icol) << "  ";
		  std::cout << std::endl;
		}
	      double h[params_.size()*params_.size()];
	      for (unsigned int i=0; i<params_.size()*params_.size(); i++)
		h[0] = 0.0;
	      const double* vec = minuit_two->X();
	      std::vector<evalT> res(vec, vec + params_.size());
	      for (unsigned int i=0; i<res.size(); i++)
		std::cout << "DEBUG result " << res.at(i) << std::endl;
	      minuit_two_grad_fcn->HessianFunc(res, h);//includes fixed params
	      std::cout << "ANALYTIC HESSIAN" <<  std::endl;
	      for (unsigned int irow=0; irow<params_.size(); irow++)
		{
		  for (unsigned int icol=0; icol<params_.size(); icol++)
		    std::cout << 0.5 * h[irow+icol*params.size()] << "  ";//factor 0.5 here because of factor 2.0 in calculation
		  std::cout << std::endl;
		}
	    }
	  if (opts_->correct_weighted_fit && data->event_weight_idx() != -1)
	    {
	      if (opts_->analytic_fisher && opts_->analytic_hessian)
		{
		  //do correction here
		  double f[params_.size()*params_.size()];
		  fisher(pdf, params, data, floating_params.size(), f);
		  std::cout << "Fisher information matrix:" << std::endl;
		  for (unsigned int i=0; i<params_.size(); i++)
		    {
		      for (unsigned int j=0; j<params_.size(); j++)
			std::cout << f[params_.size()*i+j] << " ";
		      std::cout << std::endl;
		    }
		  double h[params_.size()*params_.size()];
		  hessian(pdf, params, data, floating_params.size(), h);
		  std::cout << "Hesse matrix for comparison:" << std::endl;
		  for (unsigned int i=0; i<params_.size(); i++)
		    {
		      for (unsigned int j=0; j<params_.size(); j++)
			std::cout << h[params_.size()*i+j] << " ";
		      std::cout << std::endl;
		    }
		  Eigen::MatrixXd eigen_f = Eigen::MatrixXd::Zero(floating_params.size(), floating_params.size());
		  for (unsigned int i=0; i<floating_params.size(); i++)
		    for (unsigned int j=0; j<floating_params.size(); j++)
		      eigen_f(i,j) = f[params_.size()*i+j];
		  Eigen::MatrixXd inv_h = Eigen::MatrixXd::Zero(floating_params.size(), floating_params.size());
		  for (unsigned int i=0; i<floating_params.size(); i++)
		    for (unsigned int j=0; j<floating_params.size(); j++)
		      inv_h(i,j) = h[params_.size()*i+j];
		  inv_h = inv_h.inverse();
		  std::cout << "inverted Hesse matrix:" << std::endl;
		  for (unsigned int i=0; i<floating_params.size(); i++)
		    {
		      for (unsigned int j=0; j<floating_params.size(); j++)
			std::cout << inv_h(i,j) << " ";
		      std::cout << std::endl;
		    }
		  //calculate corrected covariance matrix
		  Eigen::MatrixXd corr_cov = Eigen::MatrixXd::Zero(floating_params.size(), floating_params.size());
		  corr_cov = -2.0 * inv_h * eigen_f * inv_h;
		  std::cout << "Corrected covariance matrix:" << std::endl;
		  std::cout << corr_cov << std::endl;
		  for (unsigned int i=0; i<floating_params.size(); i++)
		    for (unsigned int j=0; j<floating_params.size(); j++)
		      tmp_cov[i*floating_params.size()+j] = corr_cov(i,j);
		}
	      else
		{
		  std::cout << "Correction for weighted fit only supported with analytic hessian and fisher matrix." << std::endl;
		  assert(0);
		}
	    }
	  
	  //extract parameters
	  const double* par = minuit_two->X();
	  const double* errors = minuit_two->Errors();
	  for (unsigned int i=0; i<params_.size(); i++)
	    if (!params_.at(i)->is_constant())
	      {
		params_.at(i)->set_value(par[i]);
		params_.at(i)->set_error(errors[i]);
	      }
	}
#ifdef WITH_ROOT
      if (opts_->minimizer == fitter_options::minimizer_type::TMinuit)
	{
	  int errorcode;
	  double migrad_options[2] = {opts_->minuit_maxiterations, opts_->minuit_tolerance};      
	  minuit_one->mnexcm("MIG", migrad_options, 2, errorcode);
	  result = errorcode;
	  //extract parameters
	  for (unsigned int i=0; i<params_.size(); i++)
	    if (!params_.at(i)->is_constant())
	      {
		double v, e;
		minuit_one->GetParameter(i, v, e);
		params_.at(i)->set_value(v);
		params_.at(i)->set_error(e);
	      }
	  double m_fcn, m_edm, up;
	  if (opts_->postrun_hesse)
	    minuit_one->mnhess();
	  minuit_one->mnstat(m_fcn, m_edm, up, nfree, ntot, status_cov);
	  //status_cov  0=not calculated, 1=approximation, 2=full but forced pos. def., 3=full accurate
	  result += 100*status_cov;
	  if (status_cov == 3)
	    std::cout << "Hesse returns " << status_cov << " -> ALL GOOD" << std::endl;
	  else
	    std::cout << "Hesse returns " << status_cov << " -> SOMETHING WRONG" << std::endl;
	  //extract covariance matrix
	  minuit_one->mnemat(tmp_cov, floating_params.size());
	}
#endif
      for (auto * param: params)
	{
	  int minuit_idx = param->get_minuit_idx();
	  if (minuit_idx >= 0)
	    param->set_error(sqrt(tmp_cov[minuit_idx*floating_params.size()+minuit_idx]));
	}
      
      if (opts_->print_level > 1)
	std::cout << "fit procedure finished" << std::endl;
      auto t_after_fit = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "fit of " << data->nevents() << " events took " << std::chrono::duration<double, std::milli>(t_after_fit-t_before_fit).count() << " ms in total" << std::endl;
      return true;
    }
    //compute likelihood at current point
    double likelihood()
    {
      //call likelihood method with the pointers saved during fit method
      return likelihood(pdf_, params_, data_);
    }
    double likelihood_and_gradient(int npar, double* grad)
    {
      //call likelihood method with the pointers saved during fit method      
      if (false) //debug check derivatives
	{
	  double buffers[params_.size()];
	  for (unsigned int i=0; i<params_.size(); i++)
	    buffers[i] = params_.at(i)->get_value();      

	  double eps = 1.0e-6;
	  double lhplus=0.0, lhminus=0.0, lhzero=0.0;
	  double gradzero[params_.size()];
	  double gradplus[params_.size()];
	  double gradminus[params_.size()];
	  std::cout << "gradient: params: ";
	  for (unsigned int i=0; i< params_.size(); i++)
	    std::cout << buffers[i] << ", ";
	  std::cout << std::endl;

	  int floating_idx = 0;
	  for (unsigned int j=0; j< params_.size(); j++)
	    {
	      if (params_.at(j)->is_constant())
		continue;
	      for (unsigned int i=0; i<params_.size(); i++)
		params_.at(i)->set_value(buffers[i]);      
	      lhzero = likelihood_and_gradient(pdf_, params_, data_, npar, gradzero);
	      params_.at(j)->set_value(buffers[j]+eps);
	      lhplus = likelihood_and_gradient(pdf_, params_, data_, npar, gradplus);
	      params_.at(j)->set_value(buffers[j]-eps);
	      lhminus = likelihood_and_gradient(pdf_, params_, data_, npar, gradminus);
	      std::cout << "gradient: npar " << npar << ", " << j << ": " << "derivzero " << (lhplus-lhminus)/(2.0*eps) << " derivplus " << (lhzero-lhminus)/(eps)  << " derivminus " << (lhplus-lhzero)/(eps)  << std::endl;
	      std::cout << "gradient: npar " << npar << ", " << j << ": " << "gradzero " << gradzero[floating_idx] << " gradplus " << gradplus[floating_idx] << " gradminus " << gradminus[floating_idx] << std::endl;
	      floating_idx++;
	    }
      
	  for (unsigned int i=0; i<params_.size(); i++)
	    params_.at(i)->set_value(buffers[i]);
	}
      return likelihood_and_gradient(pdf_, params_, data_, npar, grad);
    }
    bool hessian(int npar, double* h)
    {
      return hessian(pdf_, params_, data_, npar, h);
    }
    evalT likelihood_and_gradient(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data, int npar, double* grad)
    {
      auto t_before_lhgrad = std::chrono::high_resolution_clock::now();
      std::vector<std::string> parameter_names;
      std::vector<evalT> parameter_values;
      std::vector<kernelT> parameter_buffer;
      unsigned int nfloating_parameters=0;

      auto t_before_paramcalc = std::chrono::high_resolution_clock::now();
      for (const auto& param: params)
	if (!param->is_constant())
	  {
	    parameter_names.push_back(param->get_name());
	    parameter_buffer.push_back(param->get_value());
	    parameter_values.push_back(param->get_value());
	    nfloating_parameters++;
	  }
      for (const auto& expression : grad_buffer_expressions_)
	parameter_buffer.push_back(expression->eval(parameter_names, parameter_values));
      auto t_after_paramcalc = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "param calc (lh+grad) takes " << std::chrono::duration<double, std::milli>(t_after_paramcalc-t_before_paramcalc).count() << " ms in total" << std::endl;      
      grad_block_.CopyToParameterBuffer(parameter_buffer);
      grad_block_.SubmitKernel();
      grad_block_.Finish();
      auto t_before_copy = std::chrono::high_resolution_clock::now();	  
      if (!opts_->kahan_on_accelerator)
	grad_block_.CopyFromOutputBuffer(grad_res_buffer_);
      auto t_after_copy = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "copy (lh+grad) takes " << std::chrono::duration<double, std::milli>(t_after_copy-t_before_copy).count() << " ms in total" << std::endl;
      auto t_after_lhgrad = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "lh and gradient determination of " << data->nevents() << " events took " << std::chrono::duration<double, std::milli>(t_after_lhgrad-t_before_lhgrad).count() << " ms in total" << std::endl;
      auto t_before_kahan = std::chrono::high_resolution_clock::now();
      
      evalT result = 0.0;
      std::vector<evalT> kahan_sum(nfloating_parameters+1);
      //params array is actually as long as the number of defined parameters, does include the fixed parameters
      //npar is the number of floating parameters, different from the size of params (and the field params_)
      if (opts_->kahan_on_accelerator)
	{
	  grad_block_.CopyFromKahanBuffer(kahan_sum);
	  result = -2.0*kahan_sum.at(0);
	  int floating_idx = 0;
	  for (unsigned int i=0; i< params.size(); i++)
	    {
	      if (params.at(i)->is_constant())
		grad[i] = 0.0;
	      else
		{
		  grad[i] = -2.0*kahan_sum.at(floating_idx+1);
		  floating_idx++;
		}
	    }
	}
      else
	{
	  result = -2.0*kahan_summation<evalT,kernelT>(grad_res_buffer_, 0);
	  int floating_idx = 0;
	  for (unsigned int i=0; i< params.size(); i++)
	    {
	      if (params.at(i)->is_constant())
		grad[i] = 0.0;
	      else
		{
		  grad[i] = -2.0*kahan_summation<evalT,kernelT>(grad_res_buffer_, floating_idx+1);
		  floating_idx++;
		}
	    }
	}
      if (pdf->is_extended())
	{
	  evalT total_yield = 0.0;
	  for (unsigned int i=0; i<pdf->parameters_.size(); i++)
	    total_yield += pdf->parameters_.at(i)->get_value();
	  result -= 2.0*(data->nevents()*log(total_yield) - total_yield);
	  for (unsigned int i=0; i< params.size(); i++)
	    {
	      if (!params.at(i)->is_constant())
		{
		  bool is_yield = false;
		  for (unsigned int j=0; j<pdf->parameters_.size(); j++)
		    if (params.at(i)->get_name() == pdf->parameters_.at(j)->get_name())
		      {
			is_yield = true;
			break;
		      }
		  if (is_yield)
		    grad[i] -= 2.0*(data->nevents()/total_yield - 1.0);
		}
	    }
	}
      auto t_after_kahan = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "kahan (lh+grad) takes " << std::chrono::duration<double, std::milli>(t_after_kahan-t_before_kahan).count() << " ms in total" << std::endl;
      return result;
    }
    evalT likelihood(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data)
    {
      auto t_before_lh = std::chrono::high_resolution_clock::now();
      std::vector<std::string> parameter_names;
      std::vector<evalT> parameter_values;
      std::vector<kernelT> parameter_buffer;
      unsigned int nfloating_parameters=0;
      auto t_before_paramcalc = std::chrono::high_resolution_clock::now();
      for (const auto& param: params)
	if (!param->is_constant())
	  {
	    parameter_names.push_back(param->get_name());
	    parameter_buffer.push_back(param->get_value());
	    parameter_values.push_back(param->get_value());
	    nfloating_parameters++;
	  }
      for (const auto& expression : buffer_expressions_)
	parameter_buffer.push_back(expression->eval(parameter_names, parameter_values));
      auto t_after_paramcalc = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "param calc (lh) takes " << std::chrono::duration<double, std::milli>(t_after_paramcalc-t_before_paramcalc).count() << " ms in total" << std::endl;
      block_.CopyToParameterBuffer(parameter_buffer);
      block_.SubmitKernel();
      block_.Finish();

      auto t_before_copy = std::chrono::high_resolution_clock::now();
      if (!opts_->kahan_on_accelerator)
	block_.CopyFromOutputBuffer(res_buffer_);
      auto t_after_copy = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "copy (lh) takes " << std::chrono::duration<double, std::milli>(t_after_copy-t_before_copy).count() << " ms in total" << std::endl;
      auto t_after_lh = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "lh determination of " << data->nevents() << " events took " << std::chrono::duration<double, std::milli>(t_after_lh-t_before_lh).count() << " ms in total" << std::endl;
      auto t_before_kahan = std::chrono::high_resolution_clock::now();
      evalT result = 0.0;
      std::vector<evalT> kahan_sum(1);
      if (opts_->kahan_on_accelerator)
	{
	  block_.CopyFromKahanBuffer(kahan_sum);
	  result = kahan_sum.at(0);
	}
      else
	{
	  result = kahan_summation<evalT,kernelT>(res_buffer_, 0);
	  if (std::isnan(result))
	    {
	      std::cout << "Kahan summation returns NaN" << std::endl;
	      for (auto p : params)
		std::cout <<"parameter values: " << p->get_name() << " " << p->get_value() << std::endl;
	      res_buffer_.print();
	      assert(0);
	    }
	}
      auto t_after_kahan = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "kahan (lh) takes " << std::chrono::duration<double, std::milli>(t_after_kahan-t_before_kahan).count() << " ms in total" << std::endl;
      //extended term in the current implementation is Ntot*ln(Ns+Nb)-Ns-Nb
      if (pdf->is_extended())
	{
	  evalT total_yield = 0.0;
	  for (unsigned int i=0; i<pdf->parameters_.size(); i++)
	    total_yield += pdf->parameters_.at(i)->get_value();
	  result += data->nevents()*log(total_yield) - total_yield;
	}
      if (opts_->print_level > 1)
	std::cout << "determined -2 log likelihood of " << -2.0 * result << std::endl;;
      return -2.0*result;
    }
    bool hessian(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data, int npar, double* h)
    {
      auto t_before_hessian = std::chrono::high_resolution_clock::now();
      std::vector<std::string> parameter_names;
      std::vector<evalT> parameter_values;
      std::vector<kernelT> parameter_buffer;
      unsigned int nfloating_parameters=0;
      auto t_before_paramcalc = std::chrono::high_resolution_clock::now();
      std::vector<int> floating_idx_to_global_idx;
      for (unsigned int i=0; i<params_.size(); i++)
	if (!params_.at(i)->is_constant())
	  {
	    parameter_names.push_back(params.at(i)->get_name());
	    parameter_buffer.push_back(params.at(i)->get_value());
	    parameter_values.push_back(params.at(i)->get_value());
	    nfloating_parameters++;
	    floating_idx_to_global_idx.push_back(i);
	  }
      for (const auto& expression : hessian_buffer_expressions_)
	parameter_buffer.push_back(expression->eval(parameter_names, parameter_values));
      auto t_after_paramcalc = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "param calc (hessian) takes " << std::chrono::duration<double, std::milli>(t_after_paramcalc-t_before_paramcalc).count() << " ms in total" << std::endl;
      
      hessian_block_.CopyToParameterBuffer(parameter_buffer);
      hessian_block_.SubmitKernel();
      hessian_block_.Finish();

      auto t_before_copy = std::chrono::high_resolution_clock::now();
      if (!opts_->kahan_on_accelerator)
	hessian_block_.CopyFromOutputBuffer(hessian_res_buffer_);
      auto t_after_copy = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "copy (hessian) takes " << std::chrono::duration<double, std::milli>(t_after_copy-t_before_copy).count() << " ms in total" << std::endl;
      auto t_after_hessian = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "determination of hessian for " << data->nevents() << " events took " << std::chrono::duration<double, std::milli>(t_after_hessian-t_before_hessian).count() << " ms in total" << std::endl;
      auto t_before_kahan = std::chrono::high_resolution_clock::now();

      std::vector<evalT> kahan_sum(nfloating_parameters*(nfloating_parameters+1)/2);
      //params array is actually as long as the number of defined parameters, does include the fixed parameters
      //npar is the number of floating parameters, different from the size of params (and the field params_)      
      for (unsigned int i=0; i<params_.size(); i++)
       	for (unsigned int j=0; j<params_.size(); j++)
       	  h[i+j*params_.size()] = 0.0;
      if (opts_->kahan_on_accelerator)
	hessian_block_.CopyFromKahanBuffer(kahan_sum);
      else
	for (unsigned int i=0; i<nfloating_parameters*(nfloating_parameters+1)/2; i++)
	  kahan_sum.at(i) = kahan_summation<evalT,kernelT>(hessian_res_buffer_, i);
      int idx = 0;
      for (unsigned int i=0; i<nfloating_parameters; i++)
	for (unsigned int j=0; j<nfloating_parameters; j++)
	  if (i>=j)
	    {
	      evalT result = -2.0*kahan_sum.at(idx);
	      idx++;
	      h[i+params_.size()*j] = result;
	      h[j+params_.size()*i] = result;
	    }      
      if (pdf->is_extended())
	{
	  //we have to add Nevents/(Nsig+Nbkg)^2
	  evalT total_yield = 0.0;
	  for (unsigned int i=0; i<pdf->parameters_.size(); i++)
	    total_yield += pdf->parameters_.at(i)->get_value();
	  unsigned int floating_i = 0;
	  unsigned int floating_j = 0;
	  for (unsigned int i=0; i< params.size(); i++)
	    {
	      floating_j = 0;
	      for (unsigned int j=0; j< params.size(); j++)
		{
		  if (i>=j && !params.at(i)->is_constant() && !params.at(j)->is_constant())
		    {
		      bool is_yield_i = false;
		      bool is_yield_j = false;
		      for (unsigned int k=0; k<pdf->parameters_.size(); k++)
			{
			  if (params.at(i)->get_name() == pdf->parameters_.at(k)->get_name())
			    is_yield_i = true;
			  if (params.at(j)->get_name() == pdf->parameters_.at(k)->get_name())
			    is_yield_j = true;
			}
		      if (is_yield_i && is_yield_j)
			{
			  double extended_term = 2.0*data->nevents()/(total_yield*total_yield);
			  h[floating_i+params_.size()*floating_j] += extended_term;
			  if (i != j)
			    h[floating_j+params_.size()*floating_i] += extended_term;
			}
		    }
		  if (!params.at(j)->is_constant())
		    floating_j++;
		}
	      if (!params.at(i)->is_constant())
		floating_i++;
	    }
	}

      auto t_after_kahan = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "kahan (hessian) takes " << std::chrono::duration<double, std::milli>(t_after_kahan-t_before_kahan).count() << " ms in total" << std::endl;
      
      return true;      
    }
    //calculate fisher matrix
    bool fisher(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data, int npar, double* h)
    {
      auto t_before_fisher = std::chrono::high_resolution_clock::now();
      std::vector<std::string> parameter_names;
      std::vector<evalT> parameter_values;
      std::vector<kernelT> parameter_buffer;
      unsigned int nfloating_parameters=0;
      auto t_before_paramcalc = std::chrono::high_resolution_clock::now();
      std::vector<int> floating_idx_to_global_idx;
      for (unsigned int i=0; i<params_.size(); i++)
	if (!params_.at(i)->is_constant())
	  {
	    parameter_names.push_back(params.at(i)->get_name());
	    parameter_buffer.push_back(params.at(i)->get_value());
	    parameter_values.push_back(params.at(i)->get_value());
	    nfloating_parameters++;
	    floating_idx_to_global_idx.push_back(i);
	  }
      for (const auto& expression : fisher_buffer_expressions_)
	parameter_buffer.push_back(expression->eval(parameter_names, parameter_values));
      auto t_after_paramcalc = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "param calc (fisher) takes " << std::chrono::duration<double, std::milli>(t_after_paramcalc-t_before_paramcalc).count() << " ms in total" << std::endl;
      
      fisher_block_.CopyToParameterBuffer(parameter_buffer);
      fisher_block_.SubmitKernel();
      fisher_block_.Finish();
      
      auto t_before_copy = std::chrono::high_resolution_clock::now();
      if (!opts_->kahan_on_accelerator)
	fisher_block_.CopyFromOutputBuffer(fisher_res_buffer_);
      auto t_after_copy = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "copy (fisher) takes " << std::chrono::duration<double, std::milli>(t_after_copy-t_before_copy).count() << " ms in total" << std::endl;
      auto t_after_fisher = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "determination of fisher for " << data->nevents() << " events took " << std::chrono::duration<double, std::milli>(t_after_fisher-t_before_fisher).count() << " ms in total" << std::endl;
      auto t_before_kahan = std::chrono::high_resolution_clock::now();

      std::vector<evalT> kahan_sum(nfloating_parameters*(nfloating_parameters+1)/2);
      //params array is actually as long as the number of defined parameters, does include the fixed parameters
      //npar is the number of floating parameters, different from the size of params (and the field params_)      
      for (unsigned int i=0; i<params_.size(); i++)
       	for (unsigned int j=0; j<params_.size(); j++)
       	  h[i+j*params_.size()] = 0.0;
      if (opts_->kahan_on_accelerator)
	fisher_block_.CopyFromKahanBuffer(kahan_sum);
      else
	for (unsigned int i=0; i<nfloating_parameters*(nfloating_parameters+1)/2; i++)
	  kahan_sum.at(i) = kahan_summation<evalT,kernelT>(fisher_res_buffer_, i);
      int idx = 0;
      for (unsigned int i=0; i<nfloating_parameters; i++)
	for (unsigned int j=0; j<nfloating_parameters; j++)
	  if (i>=j)
	    {
	      evalT result = -2.0*kahan_sum.at(idx);
	      idx++;
	      h[i+params_.size()*j] = result;
	      h[j+params_.size()*i] = result;
	    }      
      if (pdf->is_extended())
	{
	  //we have to add Nevents/(Nsig+Nbkg)^2
	  evalT total_yield = 0.0;
	  for (unsigned int i=0; i<pdf->parameters_.size(); i++)
	    total_yield += pdf->parameters_.at(i)->get_value();
	  unsigned int floating_i = 0;
	  unsigned int floating_j = 0;
	  for (unsigned int i=0; i< params.size(); i++)
	    {
	      floating_j = 0;
	      for (unsigned int j=0; j< params.size(); j++)
		{
		  if (i>=j && !params.at(i)->is_constant() && !params.at(j)->is_constant())
		    {
		      bool is_yield_i = false;
		      bool is_yield_j = false;
		      for (unsigned int k=0; k<pdf->parameters_.size(); k++)
			{
			  if (params.at(i)->get_name() == pdf->parameters_.at(k)->get_name())
			    is_yield_i = true;
			  if (params.at(j)->get_name() == pdf->parameters_.at(k)->get_name())
			    is_yield_j = true;
			}
		      if (is_yield_i && is_yield_j)
			{
			  double extended_term = 2.0*data->nevents()/(total_yield*total_yield);
			  h[floating_i+params_.size()*floating_j] += extended_term;
			  if (i != j)
			    h[floating_j+params_.size()*floating_i] += extended_term;
			}
		    }
		  if (!params.at(j)->is_constant())
		    floating_j++;
		}
	      if (!params.at(i)->is_constant())
		floating_i++;
	    }
	}
      auto t_after_kahan = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "kahan (fisher) takes " << std::chrono::duration<double, std::milli>(t_after_kahan-t_before_kahan).count() << " ms in total" << std::endl;
      
      return true;      
    }
    void define_parameter(int i, parameter<evalT>* p)
    {
      if (opts_->minimizer == fitter_options::minimizer_type::Minuit2)
	{
	  //note that this also defines constant parameters (with step_size 0.0)
	  if (p->is_unlimited())
	    minuit_two->SetVariable(i, p->get_name().c_str(), p->get_value(), p->get_step_size());
	  else
	    minuit_two->SetLimitedVariable(i, p->get_name().c_str(), p->get_value(), p->get_step_size(), p->get_min(), p->get_max());
	}
#ifdef WITH_ROOT
      if (opts_->minimizer == fitter_options::minimizer_type::TMinuit)
	{      
	  if (p->is_unlimited())
	    minuit_one->DefineParameter(i, p->get_name().c_str(), p->get_value(), p->get_step_size(), 0.0, 0.0);
	  else
	    minuit_one->DefineParameter(i, p->get_name().c_str(), p->get_value(), p->get_step_size(), p->get_min(), p->get_max());
	}
#endif
    }
    void define_parameters(bool reset=false)
    {
      if (opts_->minimizer == fitter_options::minimizer_type::Minuit2)
	minuit_two->Clear();
#ifdef WITH_ROOT
      if (opts_->minimizer == fitter_options::minimizer_type::TMinuit)      
	minuit_one->mncler();
#endif
      unsigned int minuit_idx = 0;
      for (unsigned int i = 0; i < params_.size(); i++)
	{
	  if (reset)
	    params_.at(i)->set_value(params_.at(i)->get_start_value());
	  define_parameter(i, params_.at(i));
	  if (!params_.at(i)->is_constant())
	    {
	      params_.at(i)->set_minuit_idx(minuit_idx);
	      minuit_idx++;
	    }
	  else
	    params_.at(i)->set_minuit_idx(-1);	    
	}
    }
  };

  template<typename kernelT, typename evalT, typename backendT, typename computeT> fitter<kernelT, evalT,  backendT, computeT>* fitter<kernelT, evalT, backendT, computeT>::global_fitter_pointer = 0;
  
}

#endif
