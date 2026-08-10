/**
 * @file physicspdfs.hh
 * @author Christoph Langenbruch
 * @date 2024-10-18
 *
 */

#ifndef PHYSICSPDFS_H
#define PHYSICSPDFS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include <array>
#include <math.h>

#include "graph.hh"
#include "eventvector.hh"
#include "parametervector.hh"
#include "pdf.hh"

namespace morefit {

  template <typename kernelT=double, typename evalT=double> 
  class KstarmumuAngularPDF: public PDF<kernelT, evalT> {
  public:
    KstarmumuAngularPDF(dimension<evalT>* ctl, dimension<evalT>* ctk, dimension<evalT>* phi,
			parameter<evalT>* Fl, parameter<evalT>* S3, parameter<evalT>* S4, parameter<evalT>* S5, parameter<evalT>* Afb, 
			parameter<evalT>* S7, parameter<evalT>* S8, parameter<evalT>* S9)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({ctl, ctk, phi});
      this->parameters_ = std::vector<parameter<evalT>*>({Fl, S3, S4, S5, Afb, 
	  S7, S8, S9});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      
      Ptr c = Constant_(9.0/32.0/M_PI);
      Ptr costhetal = Variable_(ctl()->get_name());
      Ptr costhetak = Variable_(ctk()->get_name());
      Ptr costhetal2 = Variable_(ctl()->get_name())*Variable_(ctl()->get_name());
      Ptr costhetak2 = Variable_(ctk()->get_name())*Variable_(ctk()->get_name());
      Ptr cos2thetal = 2.0*costhetal2->copy() - 1.0;
      Ptr cos2thetak = 2.0*costhetak2->copy() - 1.0;
      Ptr sinthetal2 = 1.0 - costhetal2->copy();
      Ptr sinthetak2 = 1.0 - costhetak2->copy();
      Ptr sinthetal = Sqrt<kernelT,evalT>(sinthetal2->copy());
      Ptr sinthetak = Sqrt<kernelT,evalT>(sinthetak2->copy());
      Ptr sin2thetal = 2.0*sinthetal->copy()*Variable_(ctl()->get_name());
      Ptr sin2thetak = 2.0*sinthetak->copy()*Variable_(ctk()->get_name());
      Ptr cosphi = Cos<kernelT,evalT>(Variable_(phi()->get_name()));
      Ptr cos2phi = Cos<kernelT,evalT>(2.0*Variable_(phi()->get_name()));
      Ptr sinphi = Sin<kernelT,evalT>(Variable_(phi()->get_name()));
      Ptr sin2phi = Sin<kernelT,evalT>(2.0*Variable_(phi()->get_name()));
      
      return
	  c->copy() * sinthetak2->copy() * 3.0/4.0 * (1.0-Variable_(Fl()->get_name()))
	+ c->copy() * costhetak2->copy() * Variable_(Fl()->get_name())
	+ c->copy() * sinthetak2->copy() * cos2thetal->copy() * 1.0/4.0 * (1.0-Variable_(Fl()->get_name()))
	+ c->copy() * costhetak2->copy() * cos2thetal->copy() * (-Variable_(Fl()->get_name()))
	+ c->copy() * sinthetak2->copy() * sinthetal2->copy() * cos2phi->copy() * Variable_(S3()->get_name())
	+ c->copy() * sin2thetak->copy() * sin2thetal->copy() * cosphi->copy() * Variable_(S4()->get_name())
	+ c->copy() * sin2thetak->copy() * sinthetal->copy() * cosphi->copy() * Variable_(S5()->get_name())
	+ c->copy() * sinthetak2->copy() * costhetal->copy() * 4.0/3.0 * Variable_(Afb()->get_name())
	+ c->copy() * sin2thetak->copy() * sinthetal->copy() * sinphi->copy() * Variable_(S7()->get_name())
	+ c->copy() * sin2thetak->copy() * sin2thetal->copy() * sinphi->copy() * Variable_(S8()->get_name())
	+ c->copy() * sinthetak2->copy() * sinthetal2->copy() * sin2phi->copy() * Variable_(S9()->get_name())
	;
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      return Constant<kernelT, evalT>(1.0);
    }    
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      //auto Sin_ = [](std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A) {return Sin<kernelT,evalT>(std::move(A));};
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;

      Ptr ctk_from_2 = Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name());
      Ptr ctl_from_2 = Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name());
      Ptr ctk_from_3 = Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name());
      Ptr ctl_from_3 = Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name());
      
      Ptr ctk_to_2 = Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name());
      Ptr ctl_to_2 = Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name());
      Ptr ctk_to_3 = Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name());
      Ptr ctl_to_3 = Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name());

      Ptr c = Constant_(9.0/32.0/M_PI);
      return c->copy() * (
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*(costhetalb-costhetala)*(phib-phia));
			  ((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name())) - ( ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *(Variable_(ctl()->get_to_name())-Variable_(ctl()->get_from_name()))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *3.0/4.0*(1.0-Variable_(Fl()->get_name()))
			  //((pow(costhetakb,3)/3.0-pow(costhetaka,3)/3.0)*(costhetalb-costhetala)*(phib-phia))
			  +(ctk_to_3->copy()/3.0-ctk_from_3->copy()/3.0)
			  *(Variable_(ctl()->get_to_name())-Variable_(ctl()->get_from_name()))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *Variable_(Fl()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((2.0*pow(costhetalb,3)-3.0*costhetalb)/3.0-(2.0*pow(costhetala,3)-3.0*costhetala)/3.0)*(phib-phia));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((2.0/3.0*ctl_to_3->copy()-Variable_(ctl()->get_to_name()))-(2.0/3.0*ctl_from_3->copy()-Variable_(ctl()->get_from_name())))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *1.0/4.0*(1.0-Variable_(Fl()->get_name()))
			  //((pow(costhetakb,3)/3.0-pow(costhetaka,3)/3.0)*((2.0*pow(costhetalb,3)-3.0*costhetalb)/3.0-(2.0*pow(costhetala,3)-3.0*costhetala)/3.0)*(phib-phia));
			  +(ctk_to_3->copy()/3.0-ctk_from_3->copy()/3.0)
			  *((2.0/3.0*ctl_to_3->copy()-Variable_(ctl()->get_to_name()))-(2.0/3.0*ctl_from_3->copy()-Variable_(ctl()->get_from_name())))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *(-Variable_(Fl()->get_name()))

			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((pow(costhetala,3)-3.0*costhetala)/3.0-(pow(costhetalb,3)-3.0*costhetalb)/3.0)*(sin(2.0*phib)/2.0-sin(2.0*phia)/2.0));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((ctl_from_3->copy()/3.0 - Variable_(ctl()->get_from_name()))-(ctl_to_3->copy()/3.0 - Variable_(ctl()->get_to_name())))
			  *(Sin<kernelT,evalT>(2.0*Variable_(phi()->get_to_name()))/2.0 - Sin<kernelT,evalT>(2.0*Variable_(phi()->get_from_name()))/2.0)
			  *Variable_(S3()->get_name())
			  //(4*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*(sqrt(1-pow(costhetalb,2))*(pow(costhetalb,2)-1)/3.0-sqrt(1-pow(costhetala,2))*(pow(costhetala,2)-1)/3.0)*(sin(phib)-sin(phia)));
			  +4.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *(Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy())*(ctl_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctl_from_2->copy())*(ctl_from_2->copy()-1.0)/3.0)
			  *(Sin<kernelT,evalT>(phi()->get_to_name())-Sin<kernelT,evalT>(phi()->get_from_name()))
			  * Variable_(S4()->get_name())
			  //(2.0*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*((asin(costhetalb)+costhetalb*sqrt(1-pow(costhetalb,2)))/2.0-(asin(costhetala)+costhetala*sqrt(1-pow(costhetala,2)))/2.0)*(sin(phib)-sin(phia)));
			  +2.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *((Asin<kernelT,evalT>(ctl()->get_to_name())+Variable_(ctl()->get_to_name())*Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy()))/2.0-(Asin<kernelT,evalT>(ctl()->get_from_name())+Variable_(ctl()->get_from_name())*Sqrt<kernelT,evalT>(1.0- ctl_from_2->copy()))/2.0)
			  *(Sin<kernelT,evalT>(phi()->get_to_name())-Sin<kernelT,evalT>(phi()->get_from_name()))
			  * Variable_(S5()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*(pow(costhetalb,2)/2.0-pow(costhetala,2)/2.0)*(phib-phia));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *(ctl_to_2->copy()/2.0-ctl_from_2->copy()/2.0)
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  * 4.0/3.0*Variable_(Afb()->get_name())

			  //(2.0*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*((asin(costhetalb)+costhetalb*sqrt(1-pow(costhetalb,2)))/2.0-(asin(costhetala)+costhetala*sqrt(1-pow(costhetala,2)))/2.0)*(cos(phia)-cos(phib)));
			  +2.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *((Asin<kernelT,evalT>(ctl()->get_to_name())+Variable_(ctl()->get_to_name())*Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy()))/2.0-(Asin<kernelT,evalT>(ctl()->get_from_name())+Variable_(ctl()->get_from_name())*Sqrt<kernelT,evalT>(1.0- ctl_from_2->copy()))/2.0)
			  *(Cos<kernelT,evalT>(Variable_(phi()->get_from_name()))-Cos<kernelT,evalT>(Variable_(phi()->get_to_name())))
			  *Variable_(S7()->get_name())
			  //(4*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*(sqrt(1-pow(costhetalb,2))*(pow(costhetalb,2)-1)/3.0-sqrt(1-pow(costhetala,2))*(pow(costhetala,2)-1)/3.0)*(cos(phia)-cos(phib)));
			  +4.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *(Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy())*(ctl_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctl_from_2->copy())*(ctl_from_2->copy()-1.0)/3.0)
			  *(Cos<kernelT,evalT>(Variable_(phi()->get_from_name()))-Cos<kernelT,evalT>(Variable_(phi()->get_to_name())))
			  *Variable_(S8()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((pow(costhetala,3)-3.0*costhetala)/3.0-(pow(costhetalb,3)-3.0*costhetalb)/3.0)*(cos(2.0*phia)/2.0-cos(2.0*phib)/2.0));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((ctl_from_3->copy()/3.0 - Variable_(ctl()->get_from_name()))-(ctl_to_3->copy()/3.0 - Variable_(ctl()->get_to_name())))
			  *(Cos<kernelT,evalT>(2.0*Variable_(phi()->get_from_name()))/2.0-Cos<kernelT,evalT>(2.0*Variable_(phi()->get_to_name()))/2.0)
			  *Variable_(S9()->get_name())
			  );
    }
    dimension<evalT>* ctl() const
    {
      return this->dimensions_.at(0);
    }
    dimension<evalT>* ctk() const
    {
      return this->dimensions_.at(1);
    }
    dimension<evalT>* phi() const
    {
      return this->dimensions_.at(2);
    }
    parameter<evalT>* Fl() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* S3() const
    {
      return this->parameters_.at(1);
    }
    parameter<evalT>* S4() const
    {
      return this->parameters_.at(2);
    }
    parameter<evalT>* S5() const
    {
      return this->parameters_.at(3);
    }
    parameter<evalT>* Afb() const
    {
      return this->parameters_.at(4);
    }
    parameter<evalT>* S7() const
    {
      return this->parameters_.at(5);
    }
    parameter<evalT>* S8() const
    {
      return this->parameters_.at(6);
    }
    parameter<evalT>* S9() const
    {
      return this->parameters_.at(7);
    }
    virtual evalT get_max() const override
    {
      return 9.0/32.0/M_PI*(4.0 + fabs(S3()->get_value()) + fabs(S4()->get_value()) + fabs(S5()->get_value()) + 4.0/3.0*fabs(Afb()->get_value())
			    + fabs(S7()->get_value()) + fabs(S8()->get_value()) + fabs(S9()->get_value()));
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };



  template <typename kernelT=double, typename evalT=double> 
  class KstarmumuAngularPDFAnalyticEps: public PDF<kernelT, evalT> {
  public:
    KstarmumuAngularPDFAnalyticEps(dimension<evalT>* ctl, dimension<evalT>* ctk, dimension<evalT>* phi,
				   parameter<evalT>* Fl, parameter<evalT>* S3, parameter<evalT>* S4, parameter<evalT>* S5, parameter<evalT>* Afb, 
				   parameter<evalT>* S7, parameter<evalT>* S8, parameter<evalT>* S9)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({ctl, ctk, phi});
      this->parameters_ = std::vector<parameter<evalT>*>({Fl, S3, S4, S5, Afb, 
	  S7, S8, S9});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      
      Ptr c = Constant_(9.0/32.0/M_PI);
      Ptr costhetal = Variable_(ctl()->get_name());
      Ptr costhetak = Variable_(ctk()->get_name());
      //Ptr phi_ = Variable_(phi()->get_name());
      Ptr costhetal2 = Variable_(ctl()->get_name())*Variable_(ctl()->get_name());
      Ptr costhetak2 = Variable_(ctk()->get_name())*Variable_(ctk()->get_name());
      Ptr cos2thetal = 2.0*costhetal2->copy() - 1.0;
      Ptr cos2thetak = 2.0*costhetak2->copy() - 1.0;
      Ptr sinthetal2 = 1.0 - costhetal2->copy();
      Ptr sinthetak2 = 1.0 - costhetak2->copy();
      Ptr sinthetal = Sqrt<kernelT,evalT>(sinthetal2->copy());
      Ptr sinthetak = Sqrt<kernelT,evalT>(sinthetak2->copy());
      Ptr sin2thetal = 2.0*sinthetal->copy()*Variable_(ctl()->get_name());
      Ptr sin2thetak = 2.0*sinthetak->copy()*Variable_(ctk()->get_name());
      Ptr cosphi = Cos<kernelT,evalT>(Variable_(phi()->get_name()));
      Ptr cos2phi = Cos<kernelT,evalT>(2.0*Variable_(phi()->get_name()));
      Ptr sinphi = Sin<kernelT,evalT>(Variable_(phi()->get_name()));
      Ptr sin2phi = Sin<kernelT,evalT>(2.0*Variable_(phi()->get_name()));
      /*
      Ptr eps_ = (Constant_(0.7)+Constant_(0.3)*costhetal->copy())
	*(Constant_(1.0)-Constant_(0.2)*costhetak->copy()*costhetak->copy())
	*(Constant_(0.7)+Constant_(0.3)*Sin<kernelT,evalT>(Constant_(2.0)*Variable_(phi()->get_name())));
      //    return (0.7+0.3*ctl)*(1.0-0.2*ctk*ctk)*(0.7+0.3*np.sin(2*phi))
      */
      /*
      Ptr eps_ = (Constant_(0.7)+Constant_(0.3)*costhetal->copy())
	*(Constant_(1.0)-Constant_(0.2)*costhetak->copy()*costhetak->copy())
	*(Constant_(0.8)+Constant_(0.2)*Cos<kernelT,evalT>(Constant_(2.0)*Variable_(phi()->get_name())));
	*/
      Ptr eps_ = (Constant_(0.7)-Constant_(0.3)*costhetak->copy())
	*(Constant_(1.0)-Constant_(0.2)*costhetal->copy()*costhetal->copy())
	*(Constant_(0.9)+Constant_(0.1)*Cos<kernelT,evalT>(Constant_(2.0)*Variable_(phi()->get_name())));

      return eps_->copy()*(
			   //		   (
	c->copy() * sinthetak2->copy() * 3.0/4.0 * (1.0-Variable_(Fl()->get_name()))
	+ c->copy() * costhetak2->copy() * Variable_(Fl()->get_name())
	+ c->copy() * sinthetak2->copy() * cos2thetal->copy() * 1.0/4.0 * (1.0-Variable_(Fl()->get_name()))
	+ c->copy() * costhetak2->copy() * cos2thetal->copy() * (-Variable_(Fl()->get_name()))
	+ c->copy() * sinthetak2->copy() * sinthetal2->copy() * cos2phi->copy() * Variable_(S3()->get_name())
	+ c->copy() * sin2thetak->copy() * sin2thetal->copy() * cosphi->copy() * Variable_(S4()->get_name())
	+ c->copy() * sin2thetak->copy() * sinthetal->copy() * cosphi->copy() * Variable_(S5()->get_name())
	+ c->copy() * sinthetak2->copy() * costhetal->copy() * 4.0/3.0 * Variable_(Afb()->get_name())
	+ c->copy() * sin2thetak->copy() * sinthetal->copy() * sinphi->copy() * Variable_(S7()->get_name())
	+ c->copy() * sin2thetak->copy() * sin2thetal->copy() * sinphi->copy() * Variable_(S8()->get_name())
	+ c->copy() * sinthetak2->copy() * sinthetal2->copy() * sin2phi->copy() * Variable_(S9()->get_name())
			   );
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      /*
      Ptr j1s = Constant_(3.0/4.0)*(Constant_(1.0) - Variable_(Fl()->get_name()));
      Ptr j6s = Constant_(4.0/3.0)*Variable_(Afb()->get_name());
      Ptr j9 = Variable_(S9()->get_name());
      return Constant_(63.0/625.0)*j9->copy()
	+ Constant_(63.0/625.0)*j6s->copy()
	+ Constant_(98.0/1875.0)*j1s->copy()
	+ Constant_(539.0/1250.0);
      */
      Ptr j1s = Constant_(3.0/4.0)*(Constant_(1.0) - Variable_(Fl()->get_name()));
      //Ptr j6s = Constant_(4.0/3.0)*Variable_(Afb()->get_name());
      Ptr j3 = Variable_(S3()->get_name());
      //(21.0*j3)/625.0-(21.0*j1s)/625.0+378.0/625.0
      return Constant_(21.0/625.0)*j3->copy()
	- Constant_(21.0/625.0)*j1s->copy()
	+ Constant_(378.0/625.0);
      
    }
    //TODO, JUST USED FOR PLOTTING
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;

      Ptr ctk_from_2 = Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name());
      Ptr ctl_from_2 = Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name());
      Ptr ctk_from_3 = Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name());
      Ptr ctl_from_3 = Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name());
      
      Ptr ctk_to_2 = Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name());
      Ptr ctl_to_2 = Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name());
      Ptr ctk_to_3 = Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name());
      Ptr ctl_to_3 = Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name());

      Ptr c = Constant_(9.0/32.0/M_PI);
      return c->copy() * (
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*(costhetalb-costhetala)*(phib-phia));
			  ((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name())) - ( ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *(Variable_(ctl()->get_to_name())-Variable_(ctl()->get_from_name()))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *3.0/4.0*(1.0-Variable_(Fl()->get_name()))
			  //((pow(costhetakb,3)/3.0-pow(costhetaka,3)/3.0)*(costhetalb-costhetala)*(phib-phia))
			  +(ctk_to_3->copy()/3.0-ctk_from_3->copy()/3.0)
			  *(Variable_(ctl()->get_to_name())-Variable_(ctl()->get_from_name()))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *Variable_(Fl()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((2.0*pow(costhetalb,3)-3.0*costhetalb)/3.0-(2.0*pow(costhetala,3)-3.0*costhetala)/3.0)*(phib-phia));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((2.0/3.0*ctl_to_3->copy()-Variable_(ctl()->get_to_name()))-(2.0/3.0*ctl_from_3->copy()-Variable_(ctl()->get_from_name())))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *1.0/4.0*(1.0-Variable_(Fl()->get_name()))
			  //((pow(costhetakb,3)/3.0-pow(costhetaka,3)/3.0)*((2.0*pow(costhetalb,3)-3.0*costhetalb)/3.0-(2.0*pow(costhetala,3)-3.0*costhetala)/3.0)*(phib-phia));
			  +(ctk_to_3->copy()/3.0-ctk_from_3->copy()/3.0)
			  *((2.0/3.0*ctl_to_3->copy()-Variable_(ctl()->get_to_name()))-(2.0/3.0*ctl_from_3->copy()-Variable_(ctl()->get_from_name())))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *(-Variable_(Fl()->get_name()))

			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((pow(costhetala,3)-3.0*costhetala)/3.0-(pow(costhetalb,3)-3.0*costhetalb)/3.0)*(sin(2.0*phib)/2.0-sin(2.0*phia)/2.0));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((ctl_from_3->copy()/3.0 - Variable_(ctl()->get_from_name()))-(ctl_to_3->copy()/3.0 - Variable_(ctl()->get_to_name())))
			  *(Sin<kernelT,evalT>(2.0*Variable_(phi()->get_to_name()))/2.0 - Sin<kernelT,evalT>(2.0*Variable_(phi()->get_from_name()))/2.0)
			  *Variable_(S3()->get_name())
			  //(4*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*(sqrt(1-pow(costhetalb,2))*(pow(costhetalb,2)-1)/3.0-sqrt(1-pow(costhetala,2))*(pow(costhetala,2)-1)/3.0)*(sin(phib)-sin(phia)));
			  +4.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *(Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy())*(ctl_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctl_from_2->copy())*(ctl_from_2->copy()-1.0)/3.0)
			  *(Sin<kernelT,evalT>(phi()->get_to_name())-Sin<kernelT,evalT>(phi()->get_from_name()))
			  * Variable_(S4()->get_name())
			  //(2.0*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*((asin(costhetalb)+costhetalb*sqrt(1-pow(costhetalb,2)))/2.0-(asin(costhetala)+costhetala*sqrt(1-pow(costhetala,2)))/2.0)*(sin(phib)-sin(phia)));
			  +2.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *((Asin<kernelT,evalT>(ctl()->get_to_name())+Variable_(ctl()->get_to_name())*Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy()))/2.0-(Asin<kernelT,evalT>(ctl()->get_from_name())+Variable_(ctl()->get_from_name())*Sqrt<kernelT,evalT>(1.0- ctl_from_2->copy()))/2.0)
			  *(Sin<kernelT,evalT>(phi()->get_to_name())-Sin<kernelT,evalT>(phi()->get_from_name()))
			  * Variable_(S5()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*(pow(costhetalb,2)/2.0-pow(costhetala,2)/2.0)*(phib-phia));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *(ctl_to_2->copy()/2.0-ctl_from_2->copy()/2.0)
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  * 4.0/3.0*Variable_(Afb()->get_name())

			  //(2.0*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*((asin(costhetalb)+costhetalb*sqrt(1-pow(costhetalb,2)))/2.0-(asin(costhetala)+costhetala*sqrt(1-pow(costhetala,2)))/2.0)*(cos(phia)-cos(phib)));
			  +2.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *((Asin<kernelT,evalT>(ctl()->get_to_name())+Variable_(ctl()->get_to_name())*Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy()))/2.0-(Asin<kernelT,evalT>(ctl()->get_from_name())+Variable_(ctl()->get_from_name())*Sqrt<kernelT,evalT>(1.0- ctl_from_2->copy()))/2.0)
			  *(Cos<kernelT,evalT>(Variable_(phi()->get_from_name()))-Cos<kernelT,evalT>(Variable_(phi()->get_to_name())))
			  *Variable_(S7()->get_name())
			  //(4*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*(sqrt(1-pow(costhetalb,2))*(pow(costhetalb,2)-1)/3.0-sqrt(1-pow(costhetala,2))*(pow(costhetala,2)-1)/3.0)*(cos(phia)-cos(phib)));
			  +4.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *(Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy())*(ctl_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctl_from_2->copy())*(ctl_from_2->copy()-1.0)/3.0)
			  *(Cos<kernelT,evalT>(Variable_(phi()->get_from_name()))-Cos<kernelT,evalT>(Variable_(phi()->get_to_name())))
			  *Variable_(S8()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((pow(costhetala,3)-3.0*costhetala)/3.0-(pow(costhetalb,3)-3.0*costhetalb)/3.0)*(cos(2.0*phia)/2.0-cos(2.0*phib)/2.0));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((ctl_from_3->copy()/3.0 - Variable_(ctl()->get_from_name()))-(ctl_to_3->copy()/3.0 - Variable_(ctl()->get_to_name())))
			  *(Cos<kernelT,evalT>(2.0*Variable_(phi()->get_from_name()))/2.0-Cos<kernelT,evalT>(2.0*Variable_(phi()->get_to_name()))/2.0)
			  *Variable_(S9()->get_name())
			  );
    }
    dimension<evalT>* ctl() const
    {
      return this->dimensions_.at(0);
    }
    dimension<evalT>* ctk() const
    {
      return this->dimensions_.at(1);
    }
    dimension<evalT>* phi() const
    {
      return this->dimensions_.at(2);
    }
    parameter<evalT>* Fl() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* S3() const
    {
      return this->parameters_.at(1);
    }
    parameter<evalT>* S4() const
    {
      return this->parameters_.at(2);
    }
    parameter<evalT>* S5() const
    {
      return this->parameters_.at(3);
    }
    parameter<evalT>* Afb() const
    {
      return this->parameters_.at(4);
    }
    parameter<evalT>* S7() const
    {
      return this->parameters_.at(5);
    }
    parameter<evalT>* S8() const
    {
      return this->parameters_.at(6);
    }
    parameter<evalT>* S9() const
    {
      return this->parameters_.at(7);
    }
    virtual evalT get_max() const override
    {
      return 9.0/32.0/M_PI*(4.0 + fabs(S3()->get_value()) + fabs(S4()->get_value()) + fabs(S5()->get_value()) + 4.0/3.0*fabs(Afb()->get_value())
			    + fabs(S7()->get_value()) + fabs(S8()->get_value()) + fabs(S9()->get_value()));
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };


  template <typename kernelT=double, typename evalT=double> 
  class KstarmumuAngularPDFOnnxEps: public PDF<kernelT, evalT> {
  public:
    std::string filename_;
    KstarmumuAngularPDFOnnxEps(dimension<evalT>* ctl, dimension<evalT>* ctk, dimension<evalT>* phi,
			       parameter<evalT>* Fl, parameter<evalT>* S3, parameter<evalT>* S4, parameter<evalT>* S5, parameter<evalT>* Afb, 
			       parameter<evalT>* S7, parameter<evalT>* S8, parameter<evalT>* S9, std::string filename):
      filename_(filename)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({ctl, ctk, phi});
      this->parameters_ = std::vector<parameter<evalT>*>({Fl, S3, S4, S5, Afb, 
	  S7, S8, S9});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      
      Ptr c = Constant_(9.0/32.0/M_PI);
      Ptr costhetal = Variable_(ctl()->get_name());
      Ptr costhetak = Variable_(ctk()->get_name());
      Ptr costhetal2 = Variable_(ctl()->get_name())*Variable_(ctl()->get_name());
      Ptr costhetak2 = Variable_(ctk()->get_name())*Variable_(ctk()->get_name());
      Ptr cos2thetal = 2.0*costhetal2->copy() - 1.0;
      Ptr cos2thetak = 2.0*costhetak2->copy() - 1.0;
      Ptr sinthetal2 = 1.0 - costhetal2->copy();
      Ptr sinthetak2 = 1.0 - costhetak2->copy();
      Ptr sinthetal = Sqrt<kernelT,evalT>(sinthetal2->copy());
      Ptr sinthetak = Sqrt<kernelT,evalT>(sinthetak2->copy());
      Ptr sin2thetal = 2.0*sinthetal->copy()*Variable_(ctl()->get_name());
      Ptr sin2thetak = 2.0*sinthetak->copy()*Variable_(ctk()->get_name());
      Ptr cosphi = Cos<kernelT,evalT>(Variable_(phi()->get_name()));
      Ptr cos2phi = Cos<kernelT,evalT>(2.0*Variable_(phi()->get_name()));
      Ptr sinphi = Sin<kernelT,evalT>(Variable_(phi()->get_name()));
      Ptr sin2phi = Sin<kernelT,evalT>(2.0*Variable_(phi()->get_name()));
      
      return
	  c->copy() * sinthetak2->copy() * 3.0/4.0 * (1.0-Variable_(Fl()->get_name()))
	+ c->copy() * costhetak2->copy() * Variable_(Fl()->get_name())
	+ c->copy() * sinthetak2->copy() * cos2thetal->copy() * 1.0/4.0 * (1.0-Variable_(Fl()->get_name()))
	+ c->copy() * costhetak2->copy() * cos2thetal->copy() * (-Variable_(Fl()->get_name()))
	+ c->copy() * sinthetak2->copy() * sinthetal2->copy() * cos2phi->copy() * Variable_(S3()->get_name())
	+ c->copy() * sin2thetak->copy() * sin2thetal->copy() * cosphi->copy() * Variable_(S4()->get_name())
	+ c->copy() * sin2thetak->copy() * sinthetal->copy() * cosphi->copy() * Variable_(S5()->get_name())
	+ c->copy() * sinthetak2->copy() * costhetal->copy() * 4.0/3.0 * Variable_(Afb()->get_name())
	+ c->copy() * sin2thetak->copy() * sinthetal->copy() * sinphi->copy() * Variable_(S7()->get_name())
	+ c->copy() * sin2thetak->copy() * sin2thetal->copy() * sinphi->copy() * Variable_(S8()->get_name())
	+ c->copy() * sinthetak2->copy() * sinthetal2->copy() * sin2phi->copy() * Variable_(S9()->get_name())
	;
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      std::vector<std::string> inputs;
      inputs.push_back(this->Fl()->get_name());
      inputs.push_back(this->S3()->get_name());
      inputs.push_back(this->S4()->get_name());
      inputs.push_back(this->S5()->get_name());
      inputs.push_back(this->Afb()->get_name());
      inputs.push_back(this->S7()->get_name());
      inputs.push_back(this->S8()->get_name());
      inputs.push_back(this->S9()->get_name());
      return std::make_unique<OnnxNode<kernelT, evalT>>("OnnxNorm"+IDCreator::Instance()->get_name(), filename_, inputs);
    }
    //TODO, ONLY USED FOR PLOTTING
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      //auto Sin_ = [](std::unique_ptr<ComputeGraphNode<kernelT, evalT>> A) {return Sin<kernelT,evalT>(std::move(A));};
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;

      Ptr ctk_from_2 = Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name());
      Ptr ctl_from_2 = Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name());
      Ptr ctk_from_3 = Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name())*Variable_(ctk()->get_from_name());
      Ptr ctl_from_3 = Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name())*Variable_(ctl()->get_from_name());
      
      Ptr ctk_to_2 = Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name());
      Ptr ctl_to_2 = Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name());
      Ptr ctk_to_3 = Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name())*Variable_(ctk()->get_to_name());
      Ptr ctl_to_3 = Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name())*Variable_(ctl()->get_to_name());

      Ptr c = Constant_(9.0/32.0/M_PI);
      return c->copy() * (
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*(costhetalb-costhetala)*(phib-phia));
			  ((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name())) - ( ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *(Variable_(ctl()->get_to_name())-Variable_(ctl()->get_from_name()))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *3.0/4.0*(1.0-Variable_(Fl()->get_name()))
			  //((pow(costhetakb,3)/3.0-pow(costhetaka,3)/3.0)*(costhetalb-costhetala)*(phib-phia))
			  +(ctk_to_3->copy()/3.0-ctk_from_3->copy()/3.0)
			  *(Variable_(ctl()->get_to_name())-Variable_(ctl()->get_from_name()))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *Variable_(Fl()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((2.0*pow(costhetalb,3)-3.0*costhetalb)/3.0-(2.0*pow(costhetala,3)-3.0*costhetala)/3.0)*(phib-phia));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((2.0/3.0*ctl_to_3->copy()-Variable_(ctl()->get_to_name()))-(2.0/3.0*ctl_from_3->copy()-Variable_(ctl()->get_from_name())))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *1.0/4.0*(1.0-Variable_(Fl()->get_name()))
			  //((pow(costhetakb,3)/3.0-pow(costhetaka,3)/3.0)*((2.0*pow(costhetalb,3)-3.0*costhetalb)/3.0-(2.0*pow(costhetala,3)-3.0*costhetala)/3.0)*(phib-phia));
			  +(ctk_to_3->copy()/3.0-ctk_from_3->copy()/3.0)
			  *((2.0/3.0*ctl_to_3->copy()-Variable_(ctl()->get_to_name()))-(2.0/3.0*ctl_from_3->copy()-Variable_(ctl()->get_from_name())))
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  *(-Variable_(Fl()->get_name()))

			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((pow(costhetala,3)-3.0*costhetala)/3.0-(pow(costhetalb,3)-3.0*costhetalb)/3.0)*(sin(2.0*phib)/2.0-sin(2.0*phia)/2.0));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((ctl_from_3->copy()/3.0 - Variable_(ctl()->get_from_name()))-(ctl_to_3->copy()/3.0 - Variable_(ctl()->get_to_name())))
			  *(Sin<kernelT,evalT>(2.0*Variable_(phi()->get_to_name()))/2.0 - Sin<kernelT,evalT>(2.0*Variable_(phi()->get_from_name()))/2.0)
			  *Variable_(S3()->get_name())
			  //(4*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*(sqrt(1-pow(costhetalb,2))*(pow(costhetalb,2)-1)/3.0-sqrt(1-pow(costhetala,2))*(pow(costhetala,2)-1)/3.0)*(sin(phib)-sin(phia)));
			  +4.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *(Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy())*(ctl_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctl_from_2->copy())*(ctl_from_2->copy()-1.0)/3.0)
			  *(Sin<kernelT,evalT>(phi()->get_to_name())-Sin<kernelT,evalT>(phi()->get_from_name()))
			  * Variable_(S4()->get_name())
			  //(2.0*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*((asin(costhetalb)+costhetalb*sqrt(1-pow(costhetalb,2)))/2.0-(asin(costhetala)+costhetala*sqrt(1-pow(costhetala,2)))/2.0)*(sin(phib)-sin(phia)));
			  +2.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *((Asin<kernelT,evalT>(ctl()->get_to_name())+Variable_(ctl()->get_to_name())*Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy()))/2.0-(Asin<kernelT,evalT>(ctl()->get_from_name())+Variable_(ctl()->get_from_name())*Sqrt<kernelT,evalT>(1.0- ctl_from_2->copy()))/2.0)
			  *(Sin<kernelT,evalT>(phi()->get_to_name())-Sin<kernelT,evalT>(phi()->get_from_name()))
			  * Variable_(S5()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*(pow(costhetalb,2)/2.0-pow(costhetala,2)/2.0)*(phib-phia));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *(ctl_to_2->copy()/2.0-ctl_from_2->copy()/2.0)
			  *(Variable_(phi()->get_to_name())-Variable_(phi()->get_from_name()))
			  * 4.0/3.0*Variable_(Afb()->get_name())

			  //(2.0*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*((asin(costhetalb)+costhetalb*sqrt(1-pow(costhetalb,2)))/2.0-(asin(costhetala)+costhetala*sqrt(1-pow(costhetala,2)))/2.0)*(cos(phia)-cos(phib)));
			  +2.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *((Asin<kernelT,evalT>(ctl()->get_to_name())+Variable_(ctl()->get_to_name())*Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy()))/2.0-(Asin<kernelT,evalT>(ctl()->get_from_name())+Variable_(ctl()->get_from_name())*Sqrt<kernelT,evalT>(1.0- ctl_from_2->copy()))/2.0)
			  *(Cos<kernelT,evalT>(Variable_(phi()->get_from_name()))-Cos<kernelT,evalT>(Variable_(phi()->get_to_name())))
			  *Variable_(S7()->get_name())
			  //(4*(sqrt(1-pow(costhetakb,2))*(pow(costhetakb,2)-1)/3.0-sqrt(1-pow(costhetaka,2))*(pow(costhetaka,2)-1)/3.0)*(sqrt(1-pow(costhetalb,2))*(pow(costhetalb,2)-1)/3.0-sqrt(1-pow(costhetala,2))*(pow(costhetala,2)-1)/3.0)*(cos(phia)-cos(phib)));
			  +4.0
			  *(Sqrt<kernelT,evalT>(1.0-ctk_to_2->copy())*(ctk_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctk_from_2->copy())*(ctk_from_2->copy()-1.0)/3.0)
			  *(Sqrt<kernelT,evalT>(1.0-ctl_to_2->copy())*(ctl_to_2->copy()-1.0)/3.0 - Sqrt<kernelT,evalT>(1.0-ctl_from_2->copy())*(ctl_from_2->copy()-1.0)/3.0)
			  *(Cos<kernelT,evalT>(Variable_(phi()->get_from_name()))-Cos<kernelT,evalT>(Variable_(phi()->get_to_name())))
			  *Variable_(S8()->get_name())
			  //(((pow(costhetaka,3)-3.0*costhetaka)/3.0-(pow(costhetakb,3)-3.0*costhetakb)/3.0)*((pow(costhetala,3)-3.0*costhetala)/3.0-(pow(costhetalb,3)-3.0*costhetalb)/3.0)*(cos(2.0*phia)/2.0-cos(2.0*phib)/2.0));
			  +((ctk_from_3->copy()/3.0 - Variable_(ctk()->get_from_name()))-(ctk_to_3->copy()/3.0 - Variable_(ctk()->get_to_name())))
			  *((ctl_from_3->copy()/3.0 - Variable_(ctl()->get_from_name()))-(ctl_to_3->copy()/3.0 - Variable_(ctl()->get_to_name())))
			  *(Cos<kernelT,evalT>(2.0*Variable_(phi()->get_from_name()))/2.0-Cos<kernelT,evalT>(2.0*Variable_(phi()->get_to_name()))/2.0)
			  *Variable_(S9()->get_name())
			  );
    }
    dimension<evalT>* ctl() const
    {
      return this->dimensions_.at(0);
    }
    dimension<evalT>* ctk() const
    {
      return this->dimensions_.at(1);
    }
    dimension<evalT>* phi() const
    {
      return this->dimensions_.at(2);
    }
    parameter<evalT>* Fl() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* S3() const
    {
      return this->parameters_.at(1);
    }
    parameter<evalT>* S4() const
    {
      return this->parameters_.at(2);
    }
    parameter<evalT>* S5() const
    {
      return this->parameters_.at(3);
    }
    parameter<evalT>* Afb() const
    {
      return this->parameters_.at(4);
    }
    parameter<evalT>* S7() const
    {
      return this->parameters_.at(5);
    }
    parameter<evalT>* S8() const
    {
      return this->parameters_.at(6);
    }
    parameter<evalT>* S9() const
    {
      return this->parameters_.at(7);
    }
    virtual evalT get_max() const override
    {
      return 9.0/32.0/M_PI*(4.0 + fabs(S3()->get_value()) + fabs(S4()->get_value()) + fabs(S5()->get_value()) + 4.0/3.0*fabs(Afb()->get_value())
			    + fabs(S7()->get_value()) + fabs(S8()->get_value()) + fabs(S9()->get_value()));
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };


  template <typename kernelT=double, typename evalT=double> 
  class QuadraticPDF: public PDF<kernelT, evalT> {
  public:
    QuadraticPDF(dimension<evalT>* x, parameter<evalT>* c1, parameter<evalT>* c2)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({c1, c2});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_ = Variable_(x()->get_name());
      
      return Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name());
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      return Constant_(2.0/3.0)*Variable_(c2()->get_name()) + Constant_(2.0);
    }    
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_from = Variable_(x()->get_from_name());
      Ptr x_to = Variable_(x()->get_to_name());

      return x_to->copy() - x_from->copy()
	+ Constant_(1.0/2.0)*Variable_(c1()->get_name())*(x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy())
	+ Constant_(1.0/3.0)*Variable_(c2()->get_name())*(x_to->copy()*x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy()*x_from->copy());
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* c1() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* c2() const
    {
      return this->parameters_.at(1);
    }
    virtual evalT get_max() const override
    {
      return 1.0 + fabs(c1()->get_value()) + fabs(c2()->get_value());//assumes range -1...+1
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };

  template <typename kernelT=double, typename evalT=double> 
  class QuadraticPDFAnalyticEps: public PDF<kernelT, evalT> {
  public:
    QuadraticPDFAnalyticEps(dimension<evalT>* x, parameter<evalT>* c1, parameter<evalT>* c2)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({c1, c2});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_ = Variable_(x()->get_name());
      //constant term from 0.5+0.5sin(2pix) neglected?
      Ptr eps_ = Constant_(0.5) + Constant_(0.5)*Sin<kernelT,evalT>(Constant_(2.0*M_PI)*x_->copy());
      return eps_->copy() * (Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name()));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      //return Constant_(2.0/3.0)*Variable_(c2()->get_name()) + Constant_(2.0);
      return Constant_(1.0/3.0)*Variable_(c2()->get_name()) - Constant_(1.0/2.0/M_PI)*Variable_(c1()->get_name()) + Constant_(1.0);
    }    
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_from = Variable_(x()->get_from_name());
      Ptr x_to = Variable_(x()->get_to_name());
      /*      
      (c2*xto*sin(2*%pi*xto))/(4*%pi^2)
	+(c1*sin(2*%pi*xto))/(8*%pi^2)
	-(c2*xto^2*cos(2*%pi*xto))/(4*%pi)
	-(c1*xto*cos(2*%pi*xto))/(4*%pi)
	+(c2*cos(2*%pi*xto))/(8*%pi^3)
	-cos(2*%pi*xto)/(4*%pi)
	+(c2*xto^3)/6
	+(c1*xto^2)/4
	+xto/2
	-(c2*xfrom*sin(2*%pi*xfrom))/(4*%pi^2)
	-(c1*sin(2*%pi*xfrom))/(8*%pi^2)
	+(c2*xfrom^2*cos(2*%pi*xfrom))/(4*%pi)
	+(c1*xfrom*cos(2*%pi*xfrom))/(4*%pi)
	-(c2*cos(2*%pi*xfrom))/(8*%pi^3)
	+cos(2*%pi*xfrom)/(4*%pi)
	-(c2*xfrom^3)/6
	-(c1*xfrom^2)/4
	-xfrom/2;
      */
      return
	Constant_(1.0/4.0/M_PI/M_PI)*Variable_(c2()->get_name())*x_to->copy()*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	+Constant_(1.0/8.0/M_PI/M_PI)*Variable_(c1()->get_name())*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	-Constant_(1.0/4.0/M_PI)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	-Constant_(1.0/4.0/M_PI)*Variable_(c1()->get_name())*x_to->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	+Constant_(1.0/8.0/M_PI/M_PI/M_PI)*Variable_(c2()->get_name())*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	-Constant_(1.0/4.0/M_PI)*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	+Constant_(1.0/6.0)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()*x_to->copy()
	+Constant_(1.0/4.0)*Variable_(c1()->get_name())*x_to->copy()*x_to->copy()
	+Constant_(1.0/2.0)*x_to->copy()
	
	-Constant_(1.0/4.0/M_PI/M_PI)*Variable_(c2()->get_name())*x_from->copy()*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	-Constant_(1.0/8.0/M_PI/M_PI)*Variable_(c1()->get_name())*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	+Constant_(1.0/4.0/M_PI)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	+Constant_(1.0/4.0/M_PI)*Variable_(c1()->get_name())*x_from->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	-Constant_(1.0/8.0/M_PI/M_PI/M_PI)*Variable_(c2()->get_name())*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	+Constant_(1.0/4.0/M_PI)*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	-Constant_(1.0/6.0)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()*x_from->copy()
	-Constant_(1.0/4.0)*Variable_(c1()->get_name())*x_from->copy()*x_from->copy()
	-Constant_(1.0/2.0)*x_from->copy()
	;
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* c1() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* c2() const
    {
      return this->parameters_.at(1);
    }
    virtual evalT get_max() const override
    {
      return 1.0 + fabs(c1()->get_value()) + fabs(c2()->get_value());//always smaller...
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };

  template <typename kernelT=double, typename evalT=double> 
  class QuadraticPDFOnnxEps: public PDF<kernelT, evalT> {
  public:
    std::string filename_;
    QuadraticPDFOnnxEps(dimension<evalT>* x, parameter<evalT>* c1, parameter<evalT>* c2, std::string filename):
      filename_(filename)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({c1, c2});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_ = Variable_(x()->get_name());
      
      return Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name());
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      //return Constant_(2.0/3.0)*Variable_(c2()->get_name()) + Constant_(2.0);
      //return OnnxNode(std::string name, std::string filename, std::vector<std::string> inputs);
      std::vector<std::string> inputs;
      inputs.push_back(this->c1()->get_name());
      inputs.push_back(this->c2()->get_name());
      return std::make_unique<OnnxNode<kernelT, evalT>>("OnnxNorm"+IDCreator::Instance()->get_name(), filename_, inputs);
    }    
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_from = Variable_(x()->get_from_name());
      Ptr x_to = Variable_(x()->get_to_name());

      return x_to->copy() - x_from->copy()
	+ Constant_(1.0/2.0)*Variable_(c1()->get_name())*(x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy())
	+ Constant_(1.0/3.0)*Variable_(c2()->get_name())*(x_to->copy()*x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy()*x_from->copy());
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* c1() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* c2() const
    {
      return this->parameters_.at(1);
    }
    virtual evalT get_max() const override
    {
      return 1.0 + fabs(c1()->get_value()) + fabs(c2()->get_value());//assumes range -1...+1
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };




  template <typename kernelT=double, typename evalT=double> 
  class QuadraticPDFNormalised: public PDF<kernelT, evalT> {
  public:
    QuadraticPDFNormalised(dimension<evalT>* x, parameter<evalT>* c1, parameter<evalT>* c2)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({c1, c2});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_ = Variable_(x()->get_name());
      
      //return Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name());
      return (Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name()))/(Constant_(2.0) + Constant_(2.0/3.0)*Variable_(c2()->get_name()));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      //return Constant_(2.0/3.0)*Variable_(c2()->get_name()) + Constant_(2.0);
      return Constant_(1.0);
    }    
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_from = Variable_(x()->get_from_name());
      Ptr x_to = Variable_(x()->get_to_name());

      //return x_to->copy() - x_from->copy()
      //	+ Constant_(1.0/2.0)*Variable_(c1()->get_name())*(x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy())
      //	+ Constant_(1.0/3.0)*Variable_(c2()->get_name())*(x_to->copy()*x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy()*x_from->copy());

      //((xto-xfrom)*(2*c2*xto^2 + 2*c2*xfrom*xto + 3*c1*xto + 2*c2*xfrom^2 + 3*c1*xfrom + 6))/(4*(c2+3))
      return ((x_to->copy() - x_from->copy())*(Constant_(2.0)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()
					       + Constant_(2.0)*Variable_(c2()->get_name())*x_from->copy()*x_to->copy()
					       + Constant_(3.0)*Variable_(c1()->get_name())*x_to->copy()
					       + Constant_(2.0)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()
					       + Constant_(3.0)*Variable_(c1()->get_name())*x_from->copy()
					       + Constant_(6.0)
					       ))/(Constant_(4.0)*(Variable_(c2()->get_name()) + Constant_(3.0)));
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* c1() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* c2() const
    {
      return this->parameters_.at(1);
    }
    virtual evalT get_max() const override
    {
      return 1.0 + fabs(c1()->get_value()) + fabs(c2()->get_value());//assumes range -1...+1
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };

  template <typename kernelT=double, typename evalT=double> 
  class QuadraticPDFNormalisedAnalyticEps: public PDF<kernelT, evalT> {
  public:
    QuadraticPDFNormalisedAnalyticEps(dimension<evalT>* x, parameter<evalT>* c1, parameter<evalT>* c2)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({c1, c2});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_ = Variable_(x()->get_name());
      //constant term from 0.5+0.5sin(2pix) neglected?
      Ptr eps_ = Constant_(0.5) + Constant_(0.5)*Sin<kernelT,evalT>(Constant_(2.0*M_PI)*x_->copy());
      //return eps_->copy() * (Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name()));
      return eps_->copy() * (Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name()))/(Constant_(2.0) + Constant_(2.0/3.0)*Variable_(c2()->get_name()));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      //return Constant_(2.0/3.0)*Variable_(c2()->get_name()) + Constant_(2.0);
      //return Constant_(1.0/3.0)*Variable_(c2()->get_name()) - Constant_(1.0/2.0/M_PI)*Variable_(c1()->get_name()) + Constant_(1.0);
      //(2*%pi*c2-3*c1+6*%pi)/(4*%pi*(c2+3))
      return (Constant_(2.0*M_PI)*Variable_(c2()->get_name()) - Constant_(3.0)*Variable_(c1()->get_name()) + Constant_(6.0*M_PI))/(Constant_(4.0*M_PI)*(Variable_(c2()->get_name()) + Constant_(3.0)));
    }    
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_from = Variable_(x()->get_from_name());
      Ptr x_to = Variable_(x()->get_to_name());
      /*      
      (c2*xto*sin(2*%pi*xto))/(4*%pi^2)
	+(c1*sin(2*%pi*xto))/(8*%pi^2)
	-(c2*xto^2*cos(2*%pi*xto))/(4*%pi)
	-(c1*xto*cos(2*%pi*xto))/(4*%pi)
	+(c2*cos(2*%pi*xto))/(8*%pi^3)
	-cos(2*%pi*xto)/(4*%pi)
	+(c2*xto^3)/6
	+(c1*xto^2)/4
	+xto/2
	-(c2*xfrom*sin(2*%pi*xfrom))/(4*%pi^2)
	-(c1*sin(2*%pi*xfrom))/(8*%pi^2)
	+(c2*xfrom^2*cos(2*%pi*xfrom))/(4*%pi)
	+(c1*xfrom*cos(2*%pi*xfrom))/(4*%pi)
	-(c2*cos(2*%pi*xfrom))/(8*%pi^3)
	+cos(2*%pi*xfrom)/(4*%pi)
	-(c2*xfrom^3)/6
	-(c1*xfrom^2)/4
	-xfrom/2;
      */
      /*
      return
	Constant_(1.0/4.0/M_PI/M_PI)*Variable_(c2()->get_name())*x_to->copy()*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	+Constant_(1.0/8.0/M_PI/M_PI)*Variable_(c1()->get_name())*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	-Constant_(1.0/4.0/M_PI)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	-Constant_(1.0/4.0/M_PI)*Variable_(c1()->get_name())*x_to->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	+Constant_(1.0/8.0/M_PI/M_PI/M_PI)*Variable_(c2()->get_name())*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	-Constant_(1.0/4.0/M_PI)*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	+Constant_(1.0/6.0)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()*x_to->copy()
	+Constant_(1.0/4.0)*Variable_(c1()->get_name())*x_to->copy()*x_to->copy()
	+Constant_(1.0/2.0)*x_to->copy()
	
	-Constant_(1.0/4.0/M_PI/M_PI)*Variable_(c2()->get_name())*x_from->copy()*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	-Constant_(1.0/8.0/M_PI/M_PI)*Variable_(c1()->get_name())*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	+Constant_(1.0/4.0/M_PI)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	+Constant_(1.0/4.0/M_PI)*Variable_(c1()->get_name())*x_from->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	-Constant_(1.0/8.0/M_PI/M_PI/M_PI)*Variable_(c2()->get_name())*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	+Constant_(1.0/4.0/M_PI)*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	-Constant_(1.0/6.0)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()*x_from->copy()
	-Constant_(1.0/4.0)*Variable_(c1()->get_name())*x_from->copy()*x_from->copy()
	-Constant_(1.0/2.0)*x_from->copy()
	;
      */
      return (Constant_(6.0*M_PI)*Variable_(c2()->get_name())*x_to->copy()*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	      +Constant_(3.0*M_PI)*Variable_(c1()->get_name())*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	      -Constant_(6.0*M_PI*M_PI)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	      -Constant_(6.0*M_PI*M_PI)*Variable_(c1()->get_name())*x_to->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	      +Constant_(3.0)*Variable_(c2()->get_name())*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	      -Constant_(6.0*M_PI*M_PI)*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_to->copy())
	      +Constant_(4.0*M_PI*M_PI*M_PI)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()*x_to->copy()
	      +Constant_(6.0*M_PI*M_PI*M_PI)*Variable_(c1()->get_name())*x_to->copy()*x_to->copy()
	      +Constant_(12.0*M_PI*M_PI*M_PI)*x_to->copy()
	      
	      -Constant_(6.0*M_PI)*Variable_(c2()->get_name())*x_from->copy()*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	      -Constant_(3.0*M_PI)*Variable_(c1()->get_name())*Sin<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	      +Constant_(6.0*M_PI*M_PI)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	      +Constant_(6.0*M_PI*M_PI)*Variable_(c1()->get_name())*x_from->copy()*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	      -Constant_(3.0)*Variable_(c2()->get_name())*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	      +Constant_(6.0*M_PI*M_PI)*Cos<kernelT, evalT>(Constant_(2.0*M_PI)*x_from->copy())
	      -Constant_(4.0*M_PI*M_PI*M_PI)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()*x_from->copy()
	      -Constant_(6.0*M_PI*M_PI*M_PI)*Variable_(c1()->get_name())*x_from->copy()*x_from->copy()
	      -Constant_(12.0*M_PI*M_PI*M_PI)*x_from->copy()
	      )/(Constant_(16.0*M_PI*M_PI*M_PI)*(Constant_(3.0)+Variable_(c2()->get_name())));
	/*	
      (6*%pi*c2*xto*sin(2*%pi*xto)
       +3*%pi*c1*sin(2*%pi*xto)
       -6*%pi^2*c2*xto^2*cos(2*%pi*xto)
       -6*%pi^2*c1*xto*cos(2*%pi*xto)
       +3*c2*cos(2*%pi*xto)
       -6*%pi^2*cos(2*%pi*xto)
       +4*%pi^3*c2*xto^3
       +6*%pi^3*c1*xto^2
       +12*%pi^3*xto

       -6*%pi*c2*xfrom*sin(2*%pi*xfrom)
       -3*%pi*c1*sin(2*%pi*xfrom)
       +6*%pi^2*c2*xfrom^2*cos(2*%pi*xfrom)
       +6*%pi^2*c1*xfrom*cos(2*%pi*xfrom)
       -3*c2*cos(2*%pi*xfrom)
       +6*%pi^2*cos(2*%pi*xfrom)
       -4*%pi^3*c2*xfrom^3
       -6*%pi^3*c1*xfrom^2
       -12*%pi^3*xfrom)
	/(16*%pi^3*(c2+3))
	*/
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* c1() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* c2() const
    {
      return this->parameters_.at(1);
    }
    virtual evalT get_max() const override
    {
      return 1.0 + fabs(c1()->get_value()) + fabs(c2()->get_value());//always smaller...
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };

  template <typename kernelT=double, typename evalT=double> 
  class QuadraticPDFNormalisedOnnxEps: public PDF<kernelT, evalT> {
  public:
    std::string filename_;
    QuadraticPDFNormalisedOnnxEps(dimension<evalT>* x, parameter<evalT>* c1, parameter<evalT>* c2, std::string filename):
      filename_(filename)
    {           
      this->dimensions_ = std::vector<dimension<evalT>*>({x});
      this->parameters_ = std::vector<parameter<evalT>*>({c1, c2});
    }      
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> prob() const override
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_ = Variable_(x()->get_name());
      
      //return Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name());
      return (Constant_(1.0) + x_->copy()*Variable_(c1()->get_name()) + x_->copy()*x_->copy()*Variable_(c2()->get_name()))/(Constant_(2.0) + Constant_(2.0/3.0)*Variable_(c2()->get_name()));
    }    
    virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> norm() const override
    {
      constexpr auto Variable_ = &Variable<kernelT, evalT>;      
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      //return Constant_(2.0/3.0)*Variable_(c2()->get_name()) + Constant_(2.0);
      //return OnnxNode(std::string name, std::string filename, std::vector<std::string> inputs);
      std::vector<std::string> inputs;
      inputs.push_back(this->c1()->get_name());
      inputs.push_back(this->c2()->get_name());
      return std::make_unique<OnnxNode<kernelT, evalT>>("OnnxNorm"+IDCreator::Instance()->get_name(), filename_, inputs);
    }    
  virtual std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral() const override 
    {
      //to avoid typing template arguments
      constexpr auto Constant_ = &Constant<kernelT, evalT>;
      constexpr auto Variable_ = &Variable<kernelT, evalT>;
      typedef std::unique_ptr<ComputeGraphNode<kernelT,evalT>> Ptr;
      Ptr x_from = Variable_(x()->get_from_name());
      Ptr x_to = Variable_(x()->get_to_name());

      //return x_to->copy() - x_from->copy()
      //	+ Constant_(1.0/2.0)*Variable_(c1()->get_name())*(x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy())
      //	+ Constant_(1.0/3.0)*Variable_(c2()->get_name())*(x_to->copy()*x_to->copy()*x_to->copy() - x_from->copy()*x_from->copy()*x_from->copy());
      return ((x_to->copy() - x_from->copy())*(Constant_(2.0)*Variable_(c2()->get_name())*x_to->copy()*x_to->copy()
					       + Constant_(2.0)*Variable_(c2()->get_name())*x_from->copy()*x_to->copy()
					       + Constant_(3.0)*Variable_(c1()->get_name())*x_to->copy()
					       + Constant_(2.0)*Variable_(c2()->get_name())*x_from->copy()*x_from->copy()
					       + Constant_(3.0)*Variable_(c1()->get_name())*x_from->copy()
					       + Constant_(6.0)
					       ))/(Constant_(4.0)*(Variable_(c2()->get_name()) + Constant_(3.0)));
    }
    dimension<evalT>* x() const
    {
      return this->dimensions_.at(0);
    }
    parameter<evalT>* c1() const
    {
      return this->parameters_.at(0);
    }
    parameter<evalT>* c2() const
    {
      return this->parameters_.at(1);
    }
    virtual evalT get_max() const override
    {
      return 1.0 + fabs(c1()->get_value()) + fabs(c2()->get_value());//assumes range -1...+1
    }
    virtual bool provides_analytic_norm() const
    {
      return true;
      //return false;
    }
  };

};
#endif
