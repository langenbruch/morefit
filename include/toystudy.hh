/**
 * @file toystudy.hh
 * @author Christoph Langenbruch
 * @date 2024-12145
 *
 */

#ifndef TOYSTUDY_H
#define TOYSTUDY_H

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
#include "fitter.hh"
#include "generator.hh"

#ifdef WITH_ROOT
#include "TCanvas.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TStyle.h"
#endif

namespace morefit {

  template<typename kernelT, typename evalT, typename backendT, typename computeT, typename seedT=uint32_t>    
  class toystudy {
  private:
    fitter_options* fitter_opts_;
    generator_options* generator_opts_;
    backendT* backend_;
    RandomGenerator* rnd_;
  public:
    toystudy(fitter_options* fitter_opts, generator_options* generator_opts, backendT* backend, RandomGenerator* rnd):
      fitter_opts_(fitter_opts),
      generator_opts_(generator_opts),
      backend_(backend),
      rnd_(rnd)
    {
    }
    bool toy(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, std::vector<dimension<evalT>*> dimensions, unsigned int nruns, unsigned int nevents, bool draw_pulls=false)
    {
      generator_opts_->print();
      generator<kernelT, evalT, backendT, computeT > gen(generator_opts_, backend_, rnd_);

      fitter_opts_->print();	      
      fitter<kernelT, evalT, backendT, computeT> fit(fitter_opts_, backend_);
      
      std::vector<std::vector<double>> pulls(params.size(), std::vector<double>());
      for (unsigned int i=0; i<nruns; i++)
	{
	  std::cout << "toy no "<< i << std::endl;
	  for (unsigned int j = 0; j < params.size(); j++)
	    params.at(j)->set_value(params.at(j)->get_start_value());

	  unsigned int npoisson = 0;
	  if (pdf->is_extended())
	    npoisson = rnd_->poisson(nevents);

	  morefit::EventVector<kernelT, evalT> result(dimensions, pdf->is_extended() ? npoisson : nevents, backend_->required_padding() > 0 ? true : false, backend_->required_padding());
	  //only make kernel for the first iteration
	  gen.generate(pdf->is_extended() ? npoisson : nevents, pdf, params, result, i==0 ? false : true);
	  fit.fit(pdf, params, &result, i==0 ? false : true);
	  
	  for (unsigned int j=0; j<params.size(); j++)
	    if (!params.at(j)->is_constant())
	      pulls.at(j).push_back((params.at(j)->get_value()-params.at(j)->get_start_value())/params.at(j)->get_error());
	}
      
#ifdef WITH_ROOT
      if (draw_pulls)
	{
	  gROOT->SetStyle("Plain");
	  gStyle->SetOptFit(0);
	  gStyle->SetOptStat(2211);      
	  
	  TH1D* hists[params.size()];
	  for (unsigned int i=0; i<params.size(); i++)
	    {
	      hists[i] = new TH1D(("pull"+std::to_string(i)).c_str(), (";pull("+params.at(i)->get_name()+");").c_str(), 100, -5.0, 5.0);
	      for (unsigned int j=0; j<pulls.at(i).size(); j++)
		hists[i]->Fill(pulls.at(i).at(j));
	    }
	  TCanvas* c0 = new TCanvas("c0", "c0", 1600, 1200);
	  c0->Divide(3,3);
	  for (unsigned int j=0; j<params.size(); j++)
	    {
	      c0->cd(j+1);
	      hists[j]->Draw("hist");
	    }
	  c0->Print("pulls.eps", "eps");
	}
#endif

      
      return true;
    }
  };
}

#endif
