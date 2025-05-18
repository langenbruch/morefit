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

namespace morefit {

  template<typename kernelT=double, typename evalT=double> 
  class PDF {
  public:
    std::vector<dimension<evalT>*> dimensions_;//pointers allow settings to be changed (eg. change min/max) after object (in this case pdf) creation
    std::vector<parameter<evalT>*> parameters_;//pointers allow settings to be changed (eg. fix parameters) after object (in this case pdf) creation
    std::vector<PDF<kernelT, evalT>*> children_;
  public:
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
    virtual evalT get_max() const
    {
      return -1.0;
    }
    //prob, but not normalised
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const = 0;
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> logprob() const
    {
      return std::make_unique<LogNode<kernelT, evalT>>(std::move(prob()));
    }
    //integral over prob, range [from...to]
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const = 0;
    //indefinite integral over prob/norm, range [from ... x] 
    //virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> indefinite_integral() const = 0;
    //definite integral over prob/norm, range [from ... to], the dimension names are replaced by dimension->get_from_name(), dimension->get_to_name()
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const
    {
      std::cout << "definite integral is not implemented" << std::endl;
      return Constant<kernelT,evalT>(0.0);
    }
    //const = 0;
    //definite integral over prob/norm, range [from ... to] 
    /*
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> old_definite_integral(const std::vector<std::string>& dimensions, const std::vector<evalT>& from, const std::vector<evalT>& to) const
    {
      if (dimensions.size() != from.size() || dimensions.size() != to.size())
	{
	  std::cout << "Invalid integration boundaries" << std::endl;
	  assert(0);
	}
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> a = indefinite_integral()->substitute(dimensions, from);
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> b = indefinite_integral()->substitute(dimensions, to);
      return std::make_unique<SumNode<kernelT, evalT>>(std::move(b), std::make_unique<NegNode<kernelT, evalT>>(std::move(a)));
    }
    */
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
    std::string get_kernel() const
    {
      return logprob_normalised()->get_kernel();
    }
    virtual bool is_extended() const
    {
      return false;
    }
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
      return Constant<kernelT, evalT>(0.5)*(Erf<kernelT, evalT>((Constant<kernelT, evalT>(to())-Variable<kernelT, evalT>(mu()->get_name()))/(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name())))-Erf<kernelT, evalT>((Constant<kernelT, evalT>(from())-Variable<kernelT, evalT>(mu()->get_name()))/(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name()))));
    }
    /*
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> indefinite_integral() const override
    {
      return Constant<kernelT, evalT>(0.5)*(Erf<kernelT, evalT>((Variable<kernelT, evalT>(x()->get_name())-Variable<kernelT, evalT>(mu()->get_name()))/(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name()))))/norm();
    }
    */
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      return Constant<kernelT, evalT>(0.5)*(Erf<kernelT, evalT>((Variable<kernelT, evalT>(x()->get_to_name())-Variable<kernelT, evalT>(mu()->get_name()))
								/(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name())))
					    -Erf<kernelT, evalT>((Variable<kernelT, evalT>(x()->get_from_name())-Variable<kernelT, evalT>(mu()->get_name()))
								 /(Sqrt<kernelT,evalT>(2.0)*Variable<kernelT, evalT>(sigma()->get_name())))
					    )/norm();	
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
    /*
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> indefinite_integral() const override
    {
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> A = Pow<kernelT,evalT>(Variable<kernelT,evalT>(n()->get_name())/Variable<kernelT,evalT>(alpha()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
	* Exp<kernelT,evalT>(-0.5*Variable<kernelT,evalT>(alpha()->get_name())*Variable<kernelT,evalT>(alpha()->get_name()));
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> B = Variable<kernelT,evalT>(n()->get_name())/Variable<kernelT,evalT>(alpha()->get_name())-Variable<kernelT,evalT>(alpha()->get_name());
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> C = Variable<kernelT,evalT>(sigma()->get_name())*B->copy()+Variable<kernelT,evalT>(mu()->get_name());
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> midpoint = Variable<kernelT,evalT>(mu()->get_name())-Variable<kernelT,evalT>(alpha()->get_name())*Variable<kernelT,evalT>(sigma()->get_name());


      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> gaussint = Sqrt<kernelT,evalT>(M_PI/2.0)*Variable<kernelT,evalT>(sigma()->get_name())
	*(Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-midpoint->copy())/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0)))
	  -Erf<kernelT,evalT>((Variable<kernelT,evalT>(mu()->get_name())-Variable<kernelT, evalT>(x()->get_name()))/(Variable<kernelT,evalT>(sigma()->get_name())*sqrt(2.0))));

      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> powint_tox =  ConditionalUnequal(Variable<kernelT, evalT>(n()->get_name())-1.0,
											A->copy()*Pow<kernelT,evalT>(Variable<kernelT,evalT>(sigma()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
											*(Pow<kernelT,evalT>(C->copy()-Variable<kernelT, evalT>(x()->get_name()),1.0-Variable<kernelT,evalT>(n()->get_name()))
											  - Pow<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()),1.0-Variable<kernelT,evalT>(n()->get_name())))
											/(Variable<kernelT,evalT>(n()->get_name())-1.0),
											A->copy()*Variable<kernelT,evalT>(sigma()->get_name())
											*(Log<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()))
											  - Log<kernelT,evalT>(C->copy()-Variable<kernelT, evalT>(x()->get_name())))
											);
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> powint_tomidpoint = ConditionalUnequal(Variable<kernelT, evalT>(n()->get_name())-1.0,
											A->copy()*Pow<kernelT,evalT>(Variable<kernelT,evalT>(sigma()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
											*(Pow<kernelT,evalT>(C->copy()-midpoint->copy(),1.0-Variable<kernelT,evalT>(n()->get_name()))
											  - Pow<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()),1.0-Variable<kernelT,evalT>(n()->get_name())))
											/(Variable<kernelT,evalT>(n()->get_name())-1.0),
											A->copy()*Variable<kernelT,evalT>(sigma()->get_name())
											*(Log<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()))
											  - Log<kernelT,evalT>(C->copy()-midpoint->copy()))
											);
      return ConditionalLarger(Variable<kernelT, evalT>(x()->get_name())-midpoint->copy(),
			       powint_tomidpoint->copy()+gaussint->copy(),
			       powint_tox->copy())/norm();
    }
    */
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      //TODO FIXME
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
      /*
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

       */

      /*
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> powint =  ConditionalUnequal(Variable<kernelT, evalT>(n()->get_name())-1.0,
										    A->copy()*Pow<kernelT,evalT>(Variable<kernelT,evalT>(sigma()->get_name()), Variable<kernelT,evalT>(n()->get_name()))
										    *(Pow<kernelT,evalT>(C->copy()-midpoint->copy(),1.0-Variable<kernelT,evalT>(n()->get_name()))
										      - Pow<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()),1.0-Variable<kernelT,evalT>(n()->get_name())))
										    /(Variable<kernelT,evalT>(n()->get_name())-1.0),
										    A->copy()*Variable<kernelT,evalT>(sigma()->get_name())
										    *(Log<kernelT,evalT>(C->copy()-Constant<kernelT,evalT>(from()))
										      - Log<kernelT,evalT>(C->copy()-midpoint->copy()))
										    );
      */
      
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
				)/norm();
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
			      (Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Constant<kernelT, evalT>(to()))-Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Constant<kernelT, evalT>(from())))/Variable<kernelT, evalT>(alpha()->get_name()));
    }
    /*
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> indefinite_integral() const override
    {
      return ConditionalEqual(Variable<kernelT, evalT>(alpha()->get_name()),
			      Variable<kernelT, evalT>(x()->get_name())/norm(),
			      Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(x()->get_name()))/(Variable<kernelT, evalT>(alpha()->get_name())*norm()));
    }
    */
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      return ConditionalEqual(Variable<kernelT, evalT>(alpha()->get_name()),
			      (Variable<kernelT, evalT>(x()->get_to_name())-Variable<kernelT, evalT>(x()->get_from_name()))/norm(),
			      (Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(x()->get_to_name()))
			       -Exp<kernelT, evalT>(Variable<kernelT, evalT>(alpha()->get_name())*Variable<kernelT, evalT>(x()->get_from_name())))
			      /(Variable<kernelT, evalT>(alpha()->get_name())*norm())
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
    /*
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> indefinite_integral() const override 
    {
      if (!extended_)
	{
	  //we add pdfs, these should be normalised
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
	  for (unsigned int i=0; i<this->children_.size()-1; i++)
	    args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(fractions().at(i)->get_name()),this->children_.at(i)->indefinite_integral()));      
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sum_args;
	  //last factor is (1-sum_i f_i) * lastchild
	  sum_args.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(1.0));
	  for (unsigned int i=0; i<this->parameters_.size(); i++)
	    sum_args.emplace_back(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name())));      
	  args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_args)), this->children_.at(this->children_.size()-1)->indefinite_integral()));
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
	    args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name()), this->children_.at(i)->indefinite_integral()));
	  return std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(args)),std::make_unique<InvNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_args))));
	}
    }
    */
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override
    {
      if (!extended_)
	{
	  //we add pdfs, these should be normalised
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
	  for (unsigned int i=0; i<this->children_.size()-1; i++)
	    args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(fractions().at(i)->get_name()),this->children_.at(i)->definite_integral()));
	  std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> sum_args;
	  //last factor is (1-sum_i f_i) * lastchild
	  sum_args.emplace_back(std::make_unique<ConstantNode<kernelT, evalT>>(1.0));
	  for (unsigned int i=0; i<this->parameters_.size(); i++)
	    sum_args.emplace_back(std::make_unique<NegNode<kernelT, evalT>>(std::make_unique<VariableNode<kernelT, evalT>>(this->parameters_.at(i)->get_name())));
	  args.emplace_back(std::make_unique<ProdNode<kernelT, evalT>>(std::make_unique<SumNode<kernelT, evalT>>(std::move(sum_args)), this->children_.at(this->children_.size()-1)->definite_integral()));
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
    /*
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> indefinite_integral() const override
    {
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> args;
      for (unsigned int i=0; i<this->children_.size(); i++)
	args.emplace_back(this->children_.at(i)->indefinite_integral());
      return std::make_unique<ProdNode<kernelT, evalT>>(std::move(args));
    }
    */
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
    /*
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> indefinite_integral() const override
    {
      //horner method
      std::unique_ptr<ComputeGraphNode<kernelT,evalT>> poly = Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-1)->get_name()));
      for (int i=1; i<coefficients().size(); i++)
	poly = Sum<kernelT,evalT>(Prod<kernelT, evalT>(std::move(poly), Variable<kernelT,evalT>(x()->get_name())),
				  Prod<kernelT,evalT>(Constant<kernelT,evalT>(1.0/evalT(coefficients().size()+1-i)), Variable<kernelT,evalT>(coefficients().at(coefficients().size()-i-1)->get_name())));
      poly = Sum<kernelT,evalT>(Prod<kernelT,evalT>(std::move(poly), Variable<kernelT,evalT>(x()->get_name())), Constant<kernelT,evalT>(1.0));
      return Prod<kernelT,evalT>(std::move(poly), Variable<kernelT,evalT>(x()->get_name()))/norm();
    }
    */
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
      return (Prod<kernelT,evalT>(std::move(poly_to), Variable<kernelT,evalT>(x()->get_to_name()))
	      -Prod<kernelT,evalT>(std::move(poly_from), Variable<kernelT,evalT>(x()->get_from_name())))/norm();
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

  };

  
}

#endif
