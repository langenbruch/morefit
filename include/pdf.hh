/**
 * @file pdf.hh
 * @author Christoph Langenbruch
 * @date 2023-02-20
 *
 */

#ifndef PDF_H
#define PDF_H

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

#include "graph.hh"
#include "eventvector.hh"
#include "parametervector.hh"
#include "random.hh"

namespace morefit {

  template<typename kernelT=double, typename evalT=double> 
  class PDF {
  public:
    std::vector<dimension<evalT>*> dimensions_;//pointers allow settings to be changed (eg. change min/max) after object (in this case pdf) creation
    std::vector<parameter<evalT>*> parameters_;//pointers allow settings to be changed (eg. fix parameters) after object (in this case pdf) creation
    std::vector<PDF<kernelT, evalT>*> children_;
    //bool has_acceptance_{false};    
    enum acceptance_type {none, histogram, bdt};
    acceptance_type acceptance_type_{acceptance_type::none};
    enum montecarlo_type {flat, importance_sampling};
    montecarlo_type montecarlo_type_{montecarlo_type::flat};
    EventVector<kernelT, evalT>* acceptance_vector_{nullptr};
    EventVector<kernelT, evalT>* montecarlo_vector_{nullptr};//TODO are there advantages of owning the vector? both for acceptance and montecarlo should be possible?
    //do we really need to change dimensions on the fly? is everything not known at compile-time?
    std::vector<int> acceptance_bins_;
    std::vector<dimension<evalT>> acceptance_dims_;
    std::vector<dimension<evalT>> montecarlo_dims_;
  public:
    void prepare_monte_carlo(EventVector<kernelT, evalT>& montecarlo_vector, int nsamples)
    {
      //std::vector<dimension<evalT>> montecarlo_dims_;
      montecarlo_dims_.clear();
      //montecarlo_dims_.push_back(dimension<evalT>("morefit_numerator", 0.0, 1.0));//could do this for importance sampling
      for (int i=0; i<this->dimensions_.size(); i++)
	{
	  montecarlo_dims_.push_back(dimension<evalT>(this->dimensions_.at(i)->get_name()+"_loop", this->dimensions_.at(i)->get_min(), this->dimensions_.at(i)->get_max()));
	  //montecarlo_dims_.push_back(dimension<evalT>(this->dimensions_.at(i)->get_to_name()+"_loop", this->dimensions_.at(i)->get_min(), this->dimensions_.at(i)->get_max()));
	}
      std::vector<dimension<evalT>*> arg;
      for (int i=0; i<montecarlo_dims_.size(); i++)
	arg.push_back(&montecarlo_dims_.at(i));
      montecarlo_vector.add_dimensions(arg);

      montecarlo_vector.resize(nsamples);
      uint64_t seed[4] = {uint64_t(987364), uint64_t(1354987), uint64_t(2680409), uint64_t(826521243)}; 
      Xoshiro256pp rnd(seed);//TODO move this to central
      for (unsigned j=0; j<nsamples; j++)
	for (unsigned int i=0; i<this->dimensions_.size(); i++)
	  montecarlo_vector.operator()(j, i) = rnd.random()*(this->dimensions_.at(i)->get_max()-this->dimensions_.at(i)->get_min())+this->dimensions_.at(i)->get_min();//initialisation
      montecarlo_vector_ = &montecarlo_vector;
      montecarlo_type_ = montecarlo_type::flat;
    }
    void set_acceptance_bdt(EventVector<kernelT, evalT>& acceptance_vector, int nnodes)
    {
      acceptance_dims_.clear();
      acceptance_dims_.push_back(dimension<evalT>("morefit_eff", 0.0, 1.0));
      for (int i=0; i<this->dimensions_.size(); i++)
	{
	  acceptance_dims_.push_back(dimension<evalT>(this->dimensions_.at(i)->get_from_name()+"_loop", this->dimensions_.at(i)->get_min(), this->dimensions_.at(i)->get_max()));
	  acceptance_dims_.push_back(dimension<evalT>(this->dimensions_.at(i)->get_to_name()+"_loop", this->dimensions_.at(i)->get_min(), this->dimensions_.at(i)->get_max()));
	}
      std::vector<dimension<evalT>*> arg;
      for (int i=0; i<acceptance_dims_.size(); i++)
	arg.push_back(&acceptance_dims_.at(i));
      acceptance_vector.add_dimensions(arg);

      acceptance_vector.resize(nnodes);
      for (unsigned j=0; j<nnodes; j++)
	acceptance_vector.operator()(j, 0) = 1.0;//initialisation

      acceptance_vector_ = &acceptance_vector;
      acceptance_type_ = acceptance_type::bdt;
    }
    void set_acceptance_histo(EventVector<kernelT, evalT>& acceptance_vector, std::vector<int> nbins)
    {
      int nallbins = 1;
      for (int i=0; i<nbins.size(); i++)
	nallbins *= nbins.at(i);

      acceptance_dims_.clear();
      acceptance_dims_.push_back(dimension<evalT>("morefit_eff", 0.0, 1.0));
      for (int i=0; i<this->dimensions_.size(); i++)
	{
	  acceptance_dims_.push_back(dimension<evalT>(this->dimensions_.at(i)->get_from_name()+"_loop", this->dimensions_.at(i)->get_min(), this->dimensions_.at(i)->get_max()));
	  acceptance_dims_.push_back(dimension<evalT>(this->dimensions_.at(i)->get_to_name()+"_loop", this->dimensions_.at(i)->get_min(), this->dimensions_.at(i)->get_max()));
	}
      std::vector<dimension<evalT>*> arg;
      for (int i=0; i<acceptance_dims_.size(); i++)
	arg.push_back(&acceptance_dims_.at(i));
      acceptance_vector.add_dimensions(arg);

      acceptance_vector.resize(nallbins);
      for (unsigned j=0; j<nallbins; j++)
	acceptance_vector.operator()(j, 0) = 1.0;//initialisation
	
      //set bin ranges
      int nbinslocal = 1;
      int previousnbinslocal = 1;
      for (int i=0; i<this->dimensions_.size(); i++)
	{
	  nbinslocal *= nbins.at(i);
	  evalT xmin = this->dimensions_.at(i)->get_min();
	  evalT xmax = this->dimensions_.at(i)->get_max();
	  evalT dx = (xmax - xmin)/nbins.at(i);
	  for (int j=0; j<nallbins; j++)//goes over all bins, for each dimension
	    {
	      int localidx = j / previousnbinslocal % nbins.at(i);	      
	      evalT from = localidx*dx + xmin;
	      evalT to = (localidx+1)*dx + xmin;
	      acceptance_vector.operator()(j, 2*i + 1) = from;
	      acceptance_vector.operator()(j, 2*i + 2) = to;
	    }
	  previousnbinslocal *= nbins.at(i);
	}
      
      acceptance_vector_ = &acceptance_vector;
      acceptance_bins_ = nbins;
      acceptance_type_ = acceptance_type::histogram;
    }
    unsigned int nparameters()
    {
      return parameters_.size();
    }
    unsigned int ndimensions()
    {
      return dimensions_.size();
    }
    const std::vector<dimension<evalT>*>& dimensions() const
    {
      return dimensions_;
    }
    const std::vector<parameter<evalT>*>& parameters() const
    {
      return parameters_;
    }
    evalT from(unsigned int idx) const
    {
      return dimensions_.at(idx).get_min();
    }
    evalT to(unsigned int idx) const
    {
      return dimensions_.at(idx).get_max();
    }
    virtual evalT get_max() const = 0;
    virtual bool provides_analytic_norm() const = 0;//needs to implement this, if false is returned will perform monte carlo integration
    //returns efficiency depending on dimension variables
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> efficiency() const
    {
      //return std::make_unique<ConstantNode<kernelT, evalT>>(1.0);//FIXME REMOVE TEST
      switch (acceptance_type_) {
      case acceptance_type::none:
	return std::make_unique<ConstantNode<kernelT, evalT>>(1.0);
	break;
      case acceptance_type::histogram:
	//assumes the eventvector efficiency as first column
	//need to determine index from dimension variables, in 1D idx = int((x-xmin)/(xmax-xmin)),
	{
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> index_sum_children;
	  for (unsigned int i=0; i<this->dimensions_.size(); i++)
	    {
	      int nbins = 1;
	      for (unsigned int j=0; j<i; j++)
		nbins *= acceptance_bins_.at(j);
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ratio = Prod<kernelT,evalT>(Prod<kernelT,evalT>(Variable<kernelT, evalT>(dimensions_.at(i)->get_name()) - Constant<kernelT,evalT>(dimensions_.at(i)->get_min()),
														Constant<kernelT,evalT>(1.0/(dimensions_.at(i)->get_max()-dimensions_.at(i)->get_min()))
														), Constant<kernelT, evalT>(acceptance_bins_.at(i)));
	      index_sum_children.emplace_back(std::make_unique<ProdNode<kernelT,evalT>>(Constant<kernelT,evalT>(nbins), std::make_unique<FloorNode<kernelT,evalT>>(std::move(ratio))));	    
	    }
	  return std::make_unique<EventVectorNode<kernelT,evalT>>(acceptance_vector_, std::make_unique<SumNode<kernelT, evalT>>(std::move(index_sum_children)), 0);
	}
      case acceptance_type::bdt:
	{
	  //return std::make_unique<ConstantNode<kernelT, evalT>>(1.0);//TODO FIXME REMOVE TEST
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>> > prefactors_theta;
	  for (int i=0; i<this->dimensions_.size(); i++)
	    {
	      std::string name = this->dimensions_.at(i)->get_name();
	      std::string loop_from_name = this->dimensions_.at(i)->get_from_name() + "_loop";
	      std::string loop_to_name = this->dimensions_.at(i)->get_to_name() + "_loop";
	      prefactors_theta.emplace_back(ConditionalLarger<kernelT,evalT>(Variable<kernelT,evalT>(name)-Variable<kernelT,evalT>(loop_from_name),
									     Constant<kernelT,evalT>(1.0), Constant<kernelT,evalT>(0.0)));
	      prefactors_theta.emplace_back(ConditionalSmaller<kernelT,evalT>(Variable<kernelT,evalT>(name)-Variable<kernelT,evalT>(loop_to_name),
									      Constant<kernelT,evalT>(1.0), Constant<kernelT,evalT>(0.0)));
	    }
	  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prefactor = std::make_unique<ProdNode<kernelT, evalT>>(std::move(prefactors_theta));
	  return std::make_unique<LoopAndSumNode<kernelT,evalT>>(acceptance_vector_, "morefit_index_" + IDCreator::Instance()->get_name(),
								 Variable<kernelT,evalT>(acceptance_dims_.at(0).get_name())*std::move(prefactor));

	}
      default:
	std::cout << "Acceptance method not implemented." << std::endl;
	assert(0);
	break;
      }
      return std::make_unique<ConstantNode<kernelT, evalT>>(1.0);
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob_eff() const
    {
      switch (acceptance_type_) {
      case acceptance_type::none:
	return prob();
      case acceptance_type::histogram:
      case acceptance_type::bdt:
	return std::make_unique<ProdNode<kernelT, evalT>>(std::move(efficiency()), std::move(prob()));
      default:
	std::cout << "Acceptance method not implemented." << std::endl;
	assert(0);	
      }
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> logprob_eff() const
    {
      return std::make_unique<LogNode<kernelT, evalT>>(std::move(prob_eff()));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm_eff() const
    {
      switch (acceptance_type_) {
      case acceptance_type::none:
	return norm();
      case acceptance_type::histogram:
      case acceptance_type::bdt://should be identical to above
	{
	  if (this->provides_analytic_norm())
	    {
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>> > replacements;
	      std::vector<std::string> names;
	      for (int i=0; i<this->dimensions_.size(); i++)
		{
		  names.push_back(this->dimensions_.at(i)->get_from_name());
		  replacements.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(this->dimensions_.at(i)->get_from_name()+"_loop"));
		  names.push_back(this->dimensions_.at(i)->get_to_name());
		  replacements.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(this->dimensions_.at(i)->get_to_name()+"_loop"));
		}
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> integral(definite_integral()->substitute(names, replacements));
	      return std::make_unique<LoopAndSumNode<kernelT,evalT>>(acceptance_vector_, "morefit_index_" + IDCreator::Instance()->get_name(),
								     Variable<kernelT,evalT>(acceptance_dims_.at(0).get_name())*std::move(integral));
	    }
	  else//numeric integration
	    {
	      //assert(0);
	      //TODO implement
	      //return std::make_unique<Constant<kernelT,evalT>>(0.0);
	      return norm();
	    }
	}
      default:
	std::cout << "Acceptance method not implemented." << std::endl;
	assert(0);	
      }
    }
    //only used for plotting!
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral_eff() const
    {
      switch (acceptance_type_) {
      case acceptance_type::none:
	return definite_integral();
      case acceptance_type::histogram:
      case acceptance_type::bdt:
	//return std::make_unique<ProdNode<kernelT, evalT>>(std::move(efficiency()), std::move(definite_integral()));
	{
	  //general approach working for multiple dimensions, but potentially slower
	  if (true)
	    {
	      //make sure that at there is overlap between [from,to] and EventVector boundaries
	      //TODO, this can be optimized slightly to fail immediately, for this do not use a product but short-circuit instead
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>> > prefactors_theta;
	      for (int i=0; i<this->dimensions_.size(); i++)
		{
		  std::string from_name = this->dimensions_.at(i)->get_from_name();
		  std::string loop_from_name = from_name + "_loop";
		  std::string to_name = this->dimensions_.at(i)->get_to_name();
		  std::string loop_to_name = to_name + "_loop";
		  //make sure that to > loop_from and from < loop_to
		  prefactors_theta.emplace_back(ConditionalLarger<kernelT,evalT>(Variable<kernelT,evalT>(to_name)-Variable<kernelT,evalT>(loop_from_name),
										 Constant<kernelT,evalT>(1.0), Constant<kernelT,evalT>(0.0)));
		  prefactors_theta.emplace_back(ConditionalSmaller<kernelT,evalT>(Variable<kernelT,evalT>(from_name)-Variable<kernelT,evalT>(loop_to_name),
										  Constant<kernelT,evalT>(1.0), Constant<kernelT,evalT>(0.0)));
		}
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prefactor = std::make_unique<ProdNode<kernelT, evalT>>(std::move(prefactors_theta));
	      //make sure that you only integrate over the overlap between [from, to] and the bin
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>> > replacements;
	      std::vector<std::string> names;
	      for (int i=0; i<this->dimensions_.size(); i++)
		{
		  std::string from_name = this->dimensions_.at(i)->get_from_name();
		  std::string loop_from_name = from_name + "_loop";
		  std::string to_name = this->dimensions_.at(i)->get_to_name();
		  std::string loop_to_name = to_name + "_loop";
		  //the integral will always be [from, to]
		  //need to change this to [max(from,lowbin), min(to,highbin)]
		  names.push_back(this->dimensions_.at(i)->get_from_name());
		  replacements.emplace_back(ConditionalLarger<kernelT,evalT>(Variable<kernelT,evalT>(from_name)-Variable<kernelT,evalT>(loop_from_name),
									     Variable<kernelT,evalT>(from_name), Variable<kernelT,evalT>(loop_from_name)));

		  names.push_back(this->dimensions_.at(i)->get_to_name());
		  replacements.emplace_back(ConditionalSmaller<kernelT,evalT>(Variable<kernelT,evalT>(to_name)-Variable<kernelT,evalT>(loop_to_name),
									      Variable<kernelT,evalT>(to_name), Variable<kernelT,evalT>(loop_to_name)));
		}
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> integral(definite_integral()->substitute(names, replacements));

	      return std::make_unique<LoopAndSumNode<kernelT,evalT>>(acceptance_vector_, "morefit_index_" + IDCreator::Instance()->get_name(),
								     Variable<kernelT,evalT>(acceptance_dims_.at(0).get_name())*std::move(prefactor)*std::move(integral));

	    }
	  else//approximate method valid for 1D only, deactivated for now
	    {
	      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> index_sum_children;
	      for (unsigned int i=0; i<this->dimensions_.size(); i++)
		{
		  int nbins = 1;
		  for (unsigned int j=0; j<i+1; j++)
		    nbins *= acceptance_bins_.at(j);		  
		  //this assumes 1D, plotting bins much smaller than efficiency histo bins
		  //->take efficiency constant over plotted bin
		  //this is to get the proper index in the event vector (even for mmultiple dimensions, should just do the plotting dimensions)
		  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> ratio = Prod<kernelT,evalT>(Prod<kernelT,evalT>(Constant<kernelT,evalT>(0.5),
														    Variable<kernelT, evalT>(dimensions_.at(i)->get_from_name())
														    +Variable<kernelT,evalT>(dimensions_.at(i)->get_to_name()))
												-Constant<kernelT,evalT>(dimensions_.at(i)->get_min()),
												Constant<kernelT,evalT>(1.0/(dimensions_.at(i)->get_max()-dimensions_.at(i)->get_min()))
												);
		  index_sum_children.emplace_back(std::make_unique<ProdNode<kernelT,evalT>>(Constant<kernelT,evalT>(nbins), std::move(ratio)));	    
		}
	      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> integral(definite_integral());
	      //this is an approximation as we do not actually loop over all bins
	      return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<EventVectorNode<kernelT,evalT>>(acceptance_vector_, std::make_unique<SumNode<kernelT, evalT>>(std::move(index_sum_children)), 0), std::move(integral));
	    }
	}
      default:
	std::cout << "Acceptance method not implemented." << std::endl;
	assert(0);
      }
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral_normalised_eff() const
    {
      return definite_integral_eff()/norm_eff();
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> lognorm_eff() const
    {
      return std::make_unique<LogNode<kernelT, evalT>>(std::move(norm_eff()));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob_normalised_eff() const
    {
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(prob_eff()), std::make_unique<InvNode<kernelT, evalT>>(std::move(norm_eff())));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> logprob_normalised_eff() const
    {
      return std::make_unique<SumNode<kernelT, evalT>>(std::move(logprob_eff()), std::make_unique<NegNode<kernelT, evalT>>(std::move(lognorm_eff())));
    }    
    //below all methods without efficiency, these are actually overwritten by derived classes
    //prob, but not normalised
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const = 0;
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> logprob() const
    {
      return std::make_unique<LogNode<kernelT, evalT>>(std::move(prob()));
    }
    //integral over prob, range [from...to]
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const
    {
      if (this->provides_analytic_norm())
	{
	  std::cout << "Please provide the analytic integral over your PDF or use the numeric integration" << std::endl;
	  std::cout << "by implementing \"virtual bool provides_analytic_norm() const {return false;};\"" << std::endl;
	  assert(0);
	}
      //numeric integration, will be overwritten in derived pdfs that implement it
      /*
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>> > replacements;
      std::vector<std::string> names;
      for (int i=0; i<this->dimensions_.size(); i++)
	{
	  names.push_back(this->dimensions_.at(i)->get_from_name());
	  replacements.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(this->dimensions_.at(i)->get_from_name()+"_loop"));
	  names.push_back(this->dimensions_.at(i)->get_to_name());
	  replacements.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(this->dimensions_.at(i)->get_to_name()+"_loop"));
	}
      */
      //std::unique_ptr<ComputeGraphNode<kernelT, evalT>> integral(definite_integral()->substitute(names, replacements));
      switch (montecarlo_type_) {
      case montecarlo_type::flat:
	{
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>> > replacements;
	  std::vector<std::string> names;
	  for (int i=0; i<this->dimensions_.size(); i++)
	    {
	      names.push_back(this->dimensions_.at(i)->get_name());
	      replacements.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(this->dimensions_.at(i)->get_name()+"_loop"));
	    }
	  std::unique_ptr<ComputeGraphNode<kernelT, evalT>> peff(this->prob_eff()->substitute(names, replacements));

	  //std::unique_ptr<ComputeGraphNode<kernelT, evalT>> peff(this->prob_eff());
	  evalT volume = 1.0;
	  for (int i=0; i<this->dimensions_.size(); i++)
	    volume *= (this->dimensions_.at(i)->get_max() - this->dimensions_.at(i)->get_min());
	  return std::make_unique<LoopAndSumNode<kernelT,evalT>>(montecarlo_vector_, "morefit_index_" + IDCreator::Instance()->get_name(),
								 Constant<kernelT,evalT>(volume/evalT(montecarlo_vector_->nevents()))*std::move(peff));
	}
      default:
	{
	  std::cout << "Monte Carlo type not implemented." << std::endl;
	  assert(0);
	}
      }
    }
    //definite integral over prob range [from ... to], the dimension names are replaced by dimension->get_from_name(), dimension->get_to_name(), this is used for eg. plotting
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const
    {
      std::cout << "definite integral is not implemented" << std::endl;
      return Constant<kernelT,evalT>(0.0);
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral_normalised() const
    {
      return definite_integral()/norm();
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> lognorm() const
    {
      return std::make_unique<LogNode<kernelT, evalT>>(std::move(norm()));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob_normalised() const
    {
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(prob()), std::make_unique<InvNode<kernelT, evalT>>(std::move(norm())));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> logprob_normalised() const
    {
      return std::make_unique<SumNode<kernelT, evalT>>(std::move(logprob()), std::make_unique<NegNode<kernelT, evalT>>(std::move(lognorm())));
    }
    virtual std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> logprob_diffs() const
    {
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> result;
      for (unsigned int i=0; i<this->parameters_.size(); i++)
	result.emplace_back(std::move(this->logprob()->diff(this->parameters_.at(i)->get_name())));
      return result;
    }  
    virtual std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> lognorm_diffs() const
    {
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> result;
      for (unsigned int i=0; i<this->parameters_.size(); i++)
	result.emplace_back(std::move(this->lognorm()->diff(this->parameters_.at(i)->get_name())));
      return result;
    }
    virtual std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> logprob_normalised_diffs() const
    {
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> result;
      for (unsigned int i=0; i<this->parameters_.size(); i++)
	result.emplace_back(std::move(this->logprob_normalised()->diff(this->parameters_.at(i)->get_name())));
      return result;
    }
    virtual bool is_extended() const
    {
      return false;
    }
    // virtual bool has_acceptance() const
    // {
    //   return has_acceptance_;
    // }
  };

  //one-dimensional Gaussian PDF
  template <typename kernelT=double, typename evalT=double> 
  class GaussianPDF: public PDF<kernelT, evalT> {
  public:
    GaussianPDF(dimension<evalT>* x, parameter<evalT>* mu, parameter<evalT>* sigma)
    {
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({mu, sigma});
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const
    {
      return (1.0/sqrt(2.0*M_PI)/Variable<kernelT, evalT>(sigma()->get_name())*Exp<kernelT, evalT>(-((Variable<kernelT, evalT>(x()->get_name())-Variable<kernelT, evalT>(mu()->get_name()))*(Variable<kernelT, evalT>(x()->get_name())-Variable<kernelT, evalT>(mu()->get_name()))/(2.0*Variable<kernelT, evalT>(sigma()->get_name())*Variable<kernelT, evalT>(sigma()->get_name())))));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const
    {
      return Constant<kernelT, evalT>(0.5)*(Erf<kernelT, evalT>((Constant<kernelT, evalT>(to())-Variable<kernelT, evalT>(mu()->get_name()))
								/(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name())))
					    -Erf<kernelT, evalT>((Constant<kernelT, evalT>(from())-Variable<kernelT, evalT>(mu()->get_name()))
								 /(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name()))));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      return Constant<kernelT, evalT>(0.5)*(Erf<kernelT, evalT>((Variable<kernelT, evalT>(x()->get_to_name())-Variable<kernelT, evalT>(mu()->get_name()))
								/(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name())))
					    -Erf<kernelT, evalT>((Variable<kernelT, evalT>(x()->get_from_name())-Variable<kernelT, evalT>(mu()->get_name()))
								 /(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name())))
					    );
    }    
    evalT from() const
    {
      return this->dimensions_.at(0)->get_min();
    }
    evalT to() const
    {
      return this->dimensions_.at(0)->get_max();
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* mu() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* sigma() const
    {
      return this->parameters_.at(1);
    }
    virtual evalT get_max() const
    {
      evalT n = 0.5*(erf((to()-mu()->get_value())/(sqrt(2.0)*sigma()->get_value()))-erf((from()-mu()->get_value())/(sqrt(2.0)*sigma()->get_value())));
      return 1.0/sqrt(2.0*M_PI)/sigma()->get_value()/n;
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
    }
  };

  //one-dimensional Crystalball PDF
  template <typename kernelT=double, typename evalT=double> 
  class CrystalballPDF: public PDF<kernelT, evalT> {
  public:
    CrystalballPDF(dimension<evalT>* x, parameter<evalT>* mu, parameter<evalT>* sigma, parameter<evalT>* alpha, parameter<evalT>* n)
    {
      //nb assumption from < (mean-alpha*sigma), to > (mean-alpha*sigma)
      assert(x->get_min() < mu->get_value()-alpha->get_value()*sigma->get_value() && x->get_max() > mu->get_value()-alpha->get_value()*sigma->get_value());
      assert(alpha->get_value() > 0.0);
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({mu, sigma, alpha, n});
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const
    {
      //condition m>mean-alpha*sigma
      //1) exp(-0.5*(m-mean)*(m-mean)/(sigma*sigma));
      //2) pow(n/alpha, n)*exp(-0.5*alpha*alpha)/pow(n/alpha-alpha-(m-mean)/sigma, n)
      return ConditionalLarger(Variable<kernelT, evalT>(x()->get_name())-Variable<kernelT, evalT>(mu()->get_name())+(Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(sigma()->get_name())),
			       Exp<kernelT, evalT>(-0.5*(Variable<kernelT, evalT>(x()->get_name())-Variable<kernelT, evalT>(mu()->get_name()))*(Variable<kernelT, evalT>(x()->get_name())-Variable<kernelT, evalT>(mu()->get_name()))/(Variable<kernelT, evalT>(sigma()->get_name())*Variable<kernelT, evalT>(sigma()->get_name()))),
			       Pow<kernelT, evalT>(Variable<kernelT, evalT>(n()->get_name())/Variable<kernelT, evalT>(alpha()->get_name()), Variable<kernelT, evalT>(n()->get_name()))*Exp<kernelT,evalT>(-0.5*Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(alpha()->get_name()))/
			       Pow<kernelT, evalT>(Variable<kernelT, evalT>(n()->get_name())/Variable<kernelT, evalT>(alpha()->get_name())-Variable<kernelT, evalT>(alpha()->get_name())-(Variable<kernelT, evalT>(x()->get_name())-Variable<kernelT, evalT>(mu()->get_name()))/Variable<kernelT, evalT>(sigma()->get_name()), Variable<kernelT, evalT>(n()->get_name()))
			       );
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const
    {
      //double A = pow(n/falpha, n) * exp(-0.5*falpha*falpha);
      //double B = n/falpha-falpha;
      //double C = sigma*B+mean;
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> A = Pow<kernelT,evalT>(Variable<kernelT,evalT>(n()->get_name())/Variable<kernelT,evalT>(alpha()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
	* Exp<kernelT,evalT>(-0.5*Variable<kernelT,evalT>(alpha()->get_name())*Variable<kernelT,evalT>(alpha()->get_name()));
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> B = Variable<kernelT,evalT>(n()->get_name())/Variable<kernelT,evalT>(alpha()->get_name())-Variable<kernelT,evalT>(alpha()->get_name());
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> C = Variable<kernelT,evalT>(sigma()->get_name())*B->copy()+Variable<kernelT,evalT>(mu()->get_name());
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> midpoint = Variable<kernelT,evalT>(mu()->get_name())-Variable<kernelT,evalT>(alpha()->get_name())*Variable<kernelT,evalT>(sigma()->get_name());

      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> gaussint = Sqrt<kernelT,evalT>(M_PI/2.0)*Variable<kernelT,evalT>(sigma()->get_name())
	*(Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-midpoint->copy())/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0)))
	  -Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-Constant<kernelT,evalT>(to()))/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0))));
      
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> powint =  ConditionalUnequal(Variable<kernelT, evalT>(n()->get_name())-1.0,
										    A->copy()*Pow<kernelT,evalT>(Variable<kernelT,evalT>(sigma()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
										    *(Pow<kernelT,evalT>(C->copy()-midpoint->copy(),1.0-Variable<kernelT,evalT>(n()->get_name()))
										      - Pow<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()),1.0-Variable<kernelT,evalT>(n()->get_name())))
										    /(Variable<kernelT,evalT>(n()->get_name())-1.0),
										    A->copy()*Variable<kernelT,evalT>(sigma()->get_name())
										    *(Log<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()))
										      - Log<kernelT,evalT>(C->copy()-midpoint->copy()))
										    );
      return powint->copy() + gaussint->copy();
      //nb assumption from < (mean-alpha*sigma), to > (mean-alpha*sigma)
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> A = Pow<kernelT,evalT>(Variable<kernelT,evalT>(n()->get_name())/Variable<kernelT,evalT>(alpha()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
	* Exp<kernelT,evalT>(-0.5*Variable<kernelT,evalT>(alpha()->get_name())*Variable<kernelT,evalT>(alpha()->get_name()));
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> B = Variable<kernelT,evalT>(n()->get_name())/Variable<kernelT,evalT>(alpha()->get_name())-Variable<kernelT,evalT>(alpha()->get_name());
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> C = Variable<kernelT,evalT>(sigma()->get_name())*B->copy()+Variable<kernelT,evalT>(mu()->get_name());
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> midpoint = Variable<kernelT,evalT>(mu()->get_name())-Variable<kernelT,evalT>(alpha()->get_name())*Variable<kernelT,evalT>(sigma()->get_name());

      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> gaussint = Sqrt<kernelT,evalT>(M_PI/2.0)*Variable<kernelT,evalT>(sigma()->get_name())
	*(Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-Variable<kernelT,evalT>(x()->get_from_name()))/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0)))
	  -Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-Variable<kernelT, evalT>(x()->get_to_name()))/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0))));

      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> gauss_from_midpoint = Sqrt<kernelT,evalT>(M_PI/2.0)*Variable<kernelT,evalT>(sigma()->get_name())
	*(Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-midpoint->copy())/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0)))
	  -Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-Variable<kernelT,evalT>(x()->get_to_name()))/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0))));
      
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> powint =
	ConditionalUnequal(Variable<kernelT, evalT>(n()->get_name())-1.0,
			   A->copy()*Pow<kernelT,evalT>(Variable<kernelT,evalT>(sigma()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
			   *(Pow<kernelT,evalT>(C->copy()-Variable<kernelT,evalT>(x()->get_to_name()),1.0-Variable<kernelT,evalT>(n()->get_name()))
			     - Pow<kernelT,evalT>(C->copy()-Variable<kernelT,evalT>(x()->get_from_name()),1.0-Variable<kernelT,evalT>(n()->get_name())))
			   /(Variable<kernelT,evalT>(n()->get_name())-1.0),
			   A->copy()*Variable<kernelT,evalT>(sigma()->get_name())
			   *(Log<kernelT,evalT>(C->copy()-Variable<kernelT,evalT>(x()->get_from_name()))
			     - Log<kernelT,evalT>(C->copy()-Variable<kernelT,evalT>(x()->get_to_name())))
			   );
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> powint_to_midpoint =
	ConditionalUnequal(Variable<kernelT, evalT>(n()->get_name())-1.0,
			   A->copy()*Pow<kernelT,evalT>(Variable<kernelT,evalT>(sigma()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
			   *(Pow<kernelT,evalT>(C->copy()-midpoint->copy(),1.0-Variable<kernelT,evalT>(n()->get_name()))
			     - Pow<kernelT,evalT>(C->copy()-Variable<kernelT,evalT>(x()->get_from_name()),1.0-Variable<kernelT,evalT>(n()->get_name())))
			   /(Variable<kernelT,evalT>(n()->get_name())-1.0),
			   A->copy()*Variable<kernelT,evalT>(sigma()->get_name())
			   *(Log<kernelT,evalT>(C->copy()-Variable<kernelT,evalT>(x()->get_from_name()))
			     - Log<kernelT,evalT>(C->copy()-midpoint->copy()))
			   );

      return ConditionalSmaller(Variable<kernelT, evalT>(x()->get_to_name())-midpoint->copy(),
				powint->copy(),//only lower tail
				ConditionalLarger(Variable<kernelT, evalT>(x()->get_from_name())-midpoint->copy(),
						  gaussint->copy(),//only upper gauss
						  powint_to_midpoint->copy() + gauss_from_midpoint->copy())//from is lower tail, to is upper gauss
				);
    }
    evalT from() const
    {
      return this->dimensions_.at(0)->get_min();
    }
    evalT to() const
    {
      return this->dimensions_.at(0)->get_max();
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* mu() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* sigma() const
    {
      return this->parameters_.at(1);
    }
    parameter<evalT>* alpha() const
    {
      return this->parameters_.at(2);
    }
    parameter<evalT>* n() const
    {
      return this->parameters_.at(3);
    }
    virtual evalT get_max() const
    {
      evalT n_ = n()->get_value();
      evalT sigma_ = sigma()->get_value();
      evalT alpha_ = alpha()->get_value();
      evalT mu_ = mu()->get_value();

      evalT A = pow(n_/alpha_, n_) * exp(-0.5*alpha_*alpha_);
      evalT B = n_/alpha_-alpha_;
      evalT C = sigma_*B+mu_;
      evalT midpoint = mu_-alpha_*sigma_;
      evalT from = x()->get_min();
      evalT to = x()->get_max();
      
      evalT integral = 0.0;
      if (n_!=1.0)
	integral += A*pow(sigma_,n_)*(pow(C-midpoint,1.0-n_) - pow(C-from,1.0-n_))/(n_-1.0);
      else
	integral += A*sigma_*(log(C-from) - log(C-midpoint));
      std::cout << "integral " << integral << std::endl;
      integral += sqrt(M_PI/2.0)*sigma_*(erf((mu_-midpoint)/(sigma_*sqrt(2.0))) - erf((mu_-to)/(sigma_*sqrt(2.0))));	  
      std::cout << "integral " << integral << std::endl;

      return 1.0/integral;
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
    }
  };
  
  
  //one-dimensional Exponential PDF
  template <typename kernelT=double, typename evalT=double> 
  class ExponentialPDF: public PDF<kernelT, evalT> {
  public:
    ExponentialPDF(dimension<evalT>* x, parameter<evalT>* alpha)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({alpha});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const
    {
      return Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(x()->get_name()));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const
    {
      return ConditionalEqual(Variable<kernelT, evalT>(alpha()->get_name()),
			      Constant<kernelT, evalT>(to()-from()),
			      (Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Constant<kernelT, evalT>(to()))
			       -Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Constant<kernelT, evalT>(from())))
			      /Variable<kernelT, evalT>(alpha()->get_name()));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      return ConditionalEqual(Variable<kernelT, evalT>(alpha()->get_name()),
			      (Variable<kernelT, evalT>(x()->get_to_name())-Variable<kernelT, evalT>(x()->get_from_name())),
			      (Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(x()->get_to_name()))
			       -Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(x()->get_from_name())))
			      /(Variable<kernelT, evalT>(alpha()->get_name()))
			      );
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* alpha() const
    {
      return this->parameters_.at(0);
    }
    evalT from() const
    {
      return this->dimensions_.at(0)->get_min();
    }
    evalT to() const
    {
      return this->dimensions_.at(0)->get_max();
    }
    virtual evalT get_max() const
    {
      evalT n = exp(alpha()->get_value()*to()) - exp(alpha()->get_value()*from());
      if (alpha()->get_value()==0.0)
	return 1.0/(to()-from());
      else if (alpha()->get_value()>0.0)
	return alpha()->get_value()*exp(alpha()->get_value()*to())/n;
      else
	return alpha()->get_value()*exp(alpha()->get_value()*from())/n;
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
    }
  };
  
  template <typename kernelT, typename evalT=double> 
  class SumPDF: public PDF<kernelT, evalT> {
  private:
    bool extended_{false};
  public:
    SumPDF(const std::vector<PDF<kernelT, evalT>*>& pdfs, const std::vector<parameter<evalT>*>& fractions)
    {
      this->children_ = pdfs;
      assert(this->children_.size() >= 2);
      assert((this->children_.size() == fractions.size()+1) || (this->children_.size() == fractions.size()));
      if (this->children_.size() == fractions.size()+1)
	extended_ = false;
      else if (this->children_.size() == fractions.size())
	extended_ = true;
      //check that the dimensions of all children are equal
      this->dimensions_ = std::vector<dimension<evalT>*>(this->children_.at(0)->dimensions());
      for (unsigned int i=1; i<this->children_.size(); i++)
	{	  
	  assert(this->dimensions_.size() == this->children_.at(i)->ndimensions());
	  bool identical = true;
	  for (unsigned int j =0; j<this->dimensions_.size(); j++)
	    {
	      std::string name = this->dimensions_.at(j)->get_name();
	      if (std::find_if(this->children_.at(i)->dimensions().begin(), this->children_.at(i)->dimensions().end(),
			       [&name](const dimension<evalT>* x) { return x->get_name() == name;})
		  == this->children_.at(i)->dimensions().end())
		identical = false; //did not find dimension
	    }
	  assert(identical);
	}
      this->parameters_ = fractions;
    }    
    SumPDF(PDF<kernelT, evalT>* pdfa, PDF<kernelT, evalT>* pdfb, parameter<evalT>* fraction)
      :SumPDF(std::vector<PDF<kernelT, evalT>*>{pdfa, pdfb}, std::vector<parameter<evalT>*>{fraction})
    {
    }
    SumPDF(PDF<kernelT, evalT>* pdfa, PDF<kernelT, evalT>* pdfb, parameter<evalT>* na, parameter<evalT>* nb)
      :SumPDF(std::vector<PDF<kernelT, evalT>*>{pdfa, pdfb}, std::vector<parameter<evalT>*>{na, nb})
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const
    {
      if (!extended_)
	{
	  //we add pdfs, these should be normalised
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
	  for (unsigned int i=0; i<this->children_.size()-1; i++)
	    args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(fractions().at(i)->get_name()),this->children_.at(i)->prob_normalised()));      
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sum_args;
	  //last factor is (1-sum_i f_i) * lastchild
	  sum_args.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(1.0));
	  for (unsigned int i=0; i<this->parameters_.size(); i++)
	    sum_args.emplace_back(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name())));      
	  args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_args)), this->children_.at(this->children_.size()-1)->prob_normalised()));
	  return std::make_unique<SumNode<kernelT, evalT>>(std::move(args));
	}
      else
	{
	  //arguments for sum of all yields
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sum_args;
	  for (unsigned int i=0; i<this->parameters_.size(); i++)
	    sum_args.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name()));
	  //loop over all children 
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
	  for (unsigned int i=0; i<this->children_.size(); i++)
	    args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name()),this->children_.at(i)->prob_normalised()));
	  return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(args)),std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_args))));
	}
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const
    {
      return std::make_unique<ConstantNode<kernelT, evalT>>(1.0);
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      if (!extended_)
	{
	  //we add pdfs, these should be normalised
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
	  for (unsigned int i=0; i<this->children_.size()-1; i++)
	    args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(fractions().at(i)->get_name()),this->children_.at(i)->definite_integral_normalised()));//was not normalised before? TODO fixme
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sum_args;
	  //last factor is (1-sum_i f_i) * lastchild
	  sum_args.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(1.0));
	  for (unsigned int i=0; i<this->parameters_.size(); i++)
	    sum_args.emplace_back(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name())));
	  args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_args)), this->children_.at(this->children_.size()-1)->definite_integral_normalised()));//was not normalised before? TODO fixme
	  return std::make_unique<SumNode<kernelT, evalT>>(std::move(args));
	}
      else
	{
	  //arguments for sum of all yields
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sum_args;
	  for (unsigned int i=0; i<this->parameters_.size(); i++)
	    sum_args.emplace_back(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name()));
	  //loop over all children 
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
	  for (unsigned int i=0; i<this->children_.size(); i++)
	    args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name()), this->children_.at(i)->definite_integral()));
	  return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(args)),std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_args))));
	}
    }
    const std::vector<parameter<evalT>*>& fractions() const
    {
      return this->parameters_;
    }
    virtual evalT get_max() const
    {
      if (!extended_)
	{
	  evalT result = 0.0;
	  for (unsigned int i=0; i<this->children_.size()-1; i++)
	    result += fractions().at(i)->get_value() * this->children_.at(i)->get_max();
	  evalT lastfraction = 1.0;
	  for (unsigned int i=0; i<this->fractions().size(); i++)
	    lastfraction -= this->fractions().at(i)->get_value();
	  result += lastfraction*this->children_.at(this->children_.size()-1)->get_max();
	  return result;
	}
      else
	{
	  evalT yield_sum = 0.0;
	  for (unsigned int i=0; i<this->children_.size(); i++)
	    yield_sum += fractions().at(i)->get_value();
	  evalT result = 0.0;
	  for (unsigned int i=0; i<this->children_.size(); i++)
	    result += fractions().at(i)->get_value()/yield_sum * this->children_.at(i)->get_max();
	  return result;
	}
    }
    virtual bool is_extended() const override
    {
      return extended_;
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
    }
  };


  template <typename kernelT, typename evalT=double>
  class ProdPDF: public PDF<kernelT, evalT> {
  public:
    ProdPDF(const std::vector<PDF<kernelT, evalT>*>& pdfs)
    {
      this->children_ = pdfs;
      assert(this->children_.size() >= 2);
      for (unsigned int i=0; i<this->children_.size(); i++)
	for (unsigned int j=0; j<this->children_.at(i).ndimensions(); j++)
	  this->dimensions_.push_back(this->children_.at(i).dimensions().at(j));
      //check that the dimensions of all children are all different
      std::set<std::string> unique_dimensions;
      for (unsigned int i=0; i<this->ndimensions(); i++)
	unique_dimensions.insert(this->dimensions().at(i).get_name());
      if (unique_dimensions.size() != this->ndimensions())
	{
	  std::cout << "All dimensions of a product PDF need to be unique" << std::endl;
	  assert(0);
	}
    }
    ProdPDF(PDF<kernelT, evalT>* pdfa, PDF<kernelT, evalT>* pdfb)
      :ProdPDF(std::vector<PDF<kernelT, evalT>*>{pdfa, pdfb})
    {
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const
    {
      //simple product
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
      for (unsigned int i=0; i<this->children_.size(); i++)
	args.emplace_back(this->children_.at(i)->prob_normalised());
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(args));
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const
    {
      return std::make_unique<ConstantNode<kernelT, evalT>>(1.0);
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
      for (unsigned int i=0; i<this->children_.size(); i++)
	args.emplace_back(this->children_.at(i)->definite_integral());
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(args));
    }    
    virtual evalT get_max() const
    {
      //worst case assumption
      evalT result = 1.0;
      for (unsigned int i=0; i<this->children_.size()-1; i++)
	result *= this->children_.at(i)->get_max();
      return result;
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
    }
  };

  //one-dimensional Polynomial PDF
  template <typename kernelT=double, typename evalT=double> 
  class PolynomialPDF: public PDF<kernelT, evalT> {
  public:
    PolynomialPDF(dimension<evalT>* x, std::vector<parameter<evalT>*> coefficients)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = coefficients;
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const
    {
      assert(coefficients().size()>0);
      //horner method
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> poly = Variable<kernelT,evalT>(coefficients().at(coefficients().size()-1)->get_name());
      for (int i=1; i<coefficients().size(); i++)
	poly = Sum<kernelT,evalT>(Prod<kernelT, evalT>(std::move(poly), Variable<kernelT,evalT>(x()->get_name())),
				  Variable<kernelT,evalT>(coefficients().at(coefficients().size()-i-1)->get_name()));
      poly = Sum<kernelT,evalT>(Prod<kernelT,evalT>(std::move(poly), Variable<kernelT,evalT>(x()->get_name())), Constant<kernelT,evalT>(1.0));
      return poly;
    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const
    {
      //horner method
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> poly_max = Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-1)->get_name()));
      for (int i=1; i<coefficients().size(); i++)
	poly_max = Sum<kernelT,evalT>(Prod<kernelT, evalT>(std::move(poly_max), Constant<kernelT,evalT>(x()->get_max())),
				      Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1-i)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-i-1)->get_name())));
      poly_max = Sum<kernelT,evalT>(Prod<kernelT,evalT>(std::move(poly_max), Constant<kernelT,evalT>(x()->get_max())), Constant<kernelT,evalT>(1.0));
      poly_max = Prod<kernelT,evalT>(std::move(poly_max), Constant<kernelT,evalT>(x()->get_max()));
      //horner method
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> poly_min = Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-1)->get_name()));
      for (int i=1; i<coefficients().size(); i++)
	poly_min = Sum<kernelT,evalT>(Prod<kernelT, evalT>(std::move(poly_min), Constant<kernelT,evalT>(x()->get_min())),
				      Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1-i)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-i-1)->get_name())));
      poly_min = Sum<kernelT,evalT>(Prod<kernelT,evalT>(std::move(poly_min), Constant<kernelT,evalT>(x()->get_min())), Constant<kernelT,evalT>(1.0));
      poly_min = Prod<kernelT,evalT>(std::move(poly_min), Constant<kernelT,evalT>(x()->get_min()));
      return Sum<kernelT,evalT>(std::move(poly_max),Neg<kernelT,evalT>(std::move(poly_min)));

    }
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      //horner method
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> poly_from = Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-1)->get_name()));
      for (int i=1; i<coefficients().size(); i++)
	poly_from = Sum<kernelT,evalT>(Prod<kernelT, evalT>(std::move(poly_from), Variable<kernelT,evalT>(x()->get_from_name())),
				  Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1-i)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-i-1)->get_name())));
      poly_from = Sum<kernelT,evalT>(Prod<kernelT,evalT>(std::move(poly_from), Variable<kernelT,evalT>(x()->get_from_name())), Constant<kernelT,evalT>(1.0));
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> poly_to = Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-1)->get_name()));
      for (int i=1; i<coefficients().size(); i++)
	poly_to = Sum<kernelT,evalT>(Prod<kernelT, evalT>(std::move(poly_to), Variable<kernelT,evalT>(x()->get_to_name())),
				  Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1-i)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-i-1)->get_name())));
      poly_to = Sum<kernelT,evalT>(Prod<kernelT,evalT>(std::move(poly_to), Variable<kernelT,evalT>(x()->get_to_name())), Constant<kernelT,evalT>(1.0));
      return Prod<kernelT,evalT>(std::move(poly_to), Variable<kernelT,evalT>(x()->get_to_name()))
	      -Prod<kernelT,evalT>(std::move(poly_from), Variable<kernelT,evalT>(x()->get_from_name()));
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    evalT from() const
    {
      return this->dimensions_.at(0)->get_min();
    }
    evalT to() const
    {
      return this->dimensions_.at(0)->get_max();
    }
    //max of normalised prob
    virtual evalT get_max() const
    {
      unsigned int ncoeffs = coefficients().size();
      std::cout << "coeffs [1.0] ";
      for (unsigned int i=0; i<ncoeffs; i++)
	std::cout << coefficients().at(i)->get_value() << " ";
      std::cout << std::endl;
      //find largest order coefficient that is non-zero
      for (unsigned int i=0; i<coefficients().size(); i++)
	{
	  if (coefficients().at(coefficients().size()-1-i)->get_value() == 0.0)	  
	    ncoeffs--;
	  else
	    break;
	}
      std::cout << "coefficients().size() " << coefficients().size() << " ncoeffs!=0 " << ncoeffs << std::endl;
      
      Eigen::VectorX<std::complex<evalT>> eigenvalues;
      if (ncoeffs >= 2)//for linear polynomials just check the limits
	{
	  //coefficients of derivative
	  std::vector<evalT> coeffs(ncoeffs);
	  for (unsigned int i=0; i<ncoeffs; i++)
	    coeffs.at(i) = (i+1)*coefficients().at(i)->get_value();

	  std::cout << "diff coeffs ";
	  for (unsigned int i=0; i<ncoeffs; i++)
	    std::cout << coeffs.at(i) << " ";
	  std::cout << std::endl;

	  for (unsigned int i=0; i<ncoeffs; i++)
	    coeffs.at(i) /= coeffs.at(ncoeffs-1);

	  std::cout << "normed diff coeffs ";
	  for (unsigned int i=0; i<ncoeffs; i++)
	    std::cout << coeffs.at(i) << " ";
	  std::cout << std::endl;

	  //companion matrix method to determine roots of derivative
	  Eigen::MatrixXd companion(ncoeffs-1, ncoeffs-1);
	  companion = Eigen::MatrixXd::Zero(ncoeffs-1, ncoeffs-1);
	  for (unsigned int i=0; i<ncoeffs-1; i++)
	    companion(i,ncoeffs-2) = -coeffs.at(i);
	  for (unsigned int i=0; i<ncoeffs-2; i++)
	    companion(i+1,i) = 1.0;
	  std::cout << "Companion matrix" << std::endl << companion << std::endl;
	  Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(companion);
	  Eigen::VectorX<std::complex<evalT>> eigenvalues = eigensolver.eigenvalues();
	  std::cout << "eigenvalues " << eigenvalues << std::endl;
	}
      //check maximum for all extrema (and limits)
      //unnormalised prob evaluation
      std::function<evalT(evalT)> probx = [&](evalT x) {
	unsigned int ncoeffs = coefficients().size();
	evalT result = coefficients().at(ncoeffs-1)->get_value();
	for (unsigned int i=1; i<ncoeffs; i++)	  
	  result = result*x + coefficients().at(ncoeffs-i-1)->get_value();
	result = result*x + 1.0;	
	return result;
      };
      std::vector<double> xvalues;
      xvalues.push_back(x()->get_min());
      xvalues.push_back(x()->get_max());
      for (unsigned int i=0; i<eigenvalues.size(); i++)
	{
	  double cur = eigenvalues(i).real();
	  if (cur > x()->get_min() && cur < x()->get_max())
	    xvalues.push_back(cur);
	}
      double max = -1.0;
      for (unsigned int i=0; i<xvalues.size(); i++)
	{
	  double p = probx(xvalues.at(i));
	  std::cout << "x " << xvalues.at(i) << " p(x) " << p << std::endl;
	  if (p > max)
	    max = p;
	}
      //now need to normalise
      std::function<evalT(evalT)> integral = [&](evalT x) {
	unsigned int ncoeffs = coefficients().size();
	evalT result = coefficients().at(ncoeffs-1)->get_value()/evalT(ncoeffs+1);
	for (unsigned int i=1; i<ncoeffs; i++)	  
	  result = result*x + coefficients().at(ncoeffs-i-1)->get_value()/evalT(ncoeffs+1-i);
	result = x*(result*x + 1.0);	
	return result;
      };
      double n = integral(x()->get_max())-integral(x()->get_min());
      std::cout << "norm " << n << std::endl;
      return 1.001*max/n;
    }
    const std::vector<parameter<evalT>*>& coefficients() const
    {
      return this->parameters_;
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
    }
  };

  
}

#endif
