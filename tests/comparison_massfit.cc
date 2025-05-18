#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <chrono>

#include "TRandom3.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TStyle.h"


#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooFitResult.h"
#include "RooDataSet.h"
#include "RooAddPdf.h"
#include "RooMCStudy.h"
#include "RooPlot.h"

// OpenCL includes
#include <CL/cl.h>
#include "morefit.hh"
#include "random.hh"

using namespace RooFit;

template<typename returnT=double, typename vectorT=double>
void mean_rms(const std::vector<vectorT>& values, returnT& mean, returnT& rms)
{
  returnT sum = 0.0;
  returnT c = 0.0;
  for (vectorT value : values) {
    returnT y = value - c;
    volatile returnT t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  mean = sum/values.size();
  rms = 0.0;
  for (vectorT value : values) 
    rms += (value - mean)*(value - mean);
  rms /= values.size();
  rms = sqrt(rms);
  return ;
}

int main()
{
  RooRealVar roofit_m("m", "m", 5.0, 7.0);
  RooRealVar roofit_mb("mb", "mb", 5.28, 5.0, 6.0);
  RooRealVar roofit_sigma("sigma", "sigma", 0.06, 0.005, 0.130);
  RooRealVar roofit_fsig("fsig", "fsig", 0.3, 0.0, 1.0);
  RooRealVar roofit_alpha("alpha", "alpha", 0.0, -10.0, 10.0);
  
  // Build gaussian pdf in terms of x,mean and sigma
  RooGaussian roofit_gauss("gauss", "gauss", roofit_m, roofit_mb, roofit_sigma);
  RooExponential roofit_expo("expo", "expo", roofit_m, roofit_alpha);
  RooAddPdf roofit_model("model","model", RooArgList(roofit_gauss, roofit_expo), roofit_fsig);

  typedef double kernelT;
  typedef double evalT;
  morefit::compute_options compute_opts;
  compute_opts.opencl_platform = 0; compute_opts.opencl_device = 0; //gpu
  //compute_opts.opencl_platform = 1; compute_opts.opencl_device = 0; //cpu  
  compute_opts.print_kernel = true;
  compute_opts.llvm_nthreads = 1;
  compute_opts.llvm_print_intermediate = true;
  compute_opts.llvm_vectorization = true;
  compute_opts.llvm_vectorization_width = 4;
  //compute_opts.llvm_cpu = "x86-64-v4";
  //compute_opts.llvm_tunecpu = "x86-64";
  compute_opts.print();

  //typedef morefit::OpenCLBackend backendT;
  //typedef morefit::OpenCLBlock<kernelT, evalT> blockT;
  //morefit::OpenCLBackend backend(&compute_opts);
  typedef morefit::LLVMBackend backendT;
  typedef morefit::LLVMBlock<kernelT, evalT> blockT;
  morefit::LLVMBackend backend(&compute_opts);

  morefit::dimension<evalT> m("m", "m(K^{+}\\mu^{+}\\mu^{-})", 5.0, 7.0, false);
  morefit::parameter<evalT> mb("mb", "m(B^{+})", 5.28, 5.0, 6.0, 0.01, false);
  morefit::parameter<evalT> sigma("sigma", "\\sigma(B^{+})", 0.06, 0.005, 0.130, 0.001, false);
  morefit::parameter<evalT> fsig("fsig", "f_{\\mathrm{sig}}", 0.3, 0.0, 1.0, 0.01, false);
  morefit::parameter<evalT> alpha("alpha", "\\alpha_{\\mathrm{bkg}}", 0.0, -10.0, 10.0, 0.01, false);
  morefit::GaussianPDF<kernelT, evalT> gaus(&m, &mb, &sigma);
  morefit::ExponentialPDF<kernelT, evalT> exp(&m, &alpha);
  morefit::SumPDF<kernelT, evalT> sum(&gaus, &exp, &fsig);

  std::vector<morefit::parameter<evalT>*> params({&mb, &sigma, &fsig, &alpha});

  morefit::Xoshiro128pp rnd;
  rnd.setSeed(229387429ULL);
  
  if (true)
    {
      unsigned int nrepeats = 1;
      const unsigned int npoints = 1;
      unsigned int nstats[npoints] = {100000};
      //const unsigned int npoints = 4;      
      //unsigned int nstats[npoints] = {1000, 10000, 100000, 1000000};
      std::vector<double> means, rmss;
      for (unsigned int n=0; n<npoints; n++)
	{
	  std::vector<double> runtimes;
	  for (unsigned int q=0; q<nrepeats; q++)
	    {
	      auto t_before_toystudy = std::chrono::high_resolution_clock::now();
	      unsigned int ngen = nstats[n];
	      unsigned int nruns = 1000;
      
	      morefit::generator_options gen_opts;
	      gen_opts.rndtype = morefit::generator_options::randomization_type::on_accelerator;
	      //gen_opts.rndtype = morefit::generator_options::randomization_type::on_host;
      
	      morefit::generator<kernelT, evalT, backendT, blockT > gen(&gen_opts, &backend, &rnd);
	      std::vector<std::vector<double>> pulls(params.size(), std::vector<double>());
	      std::vector<std::vector<double>> diffs_values(params.size(), std::vector<double>());
	      std::vector<std::vector<double>> diffs_errors(params.size(), std::vector<double>());
	      for (unsigned int i=0; i<nruns; i++)
		{
		  std::cout << "toy no "<< i << std::endl;
		  for (unsigned int j = 0; j < params.size(); j++)
		    params.at(j)->set_value(params.at(j)->get_start_value());

	  
		  morefit::EventVector<kernelT, evalT> result({&m}, ngen, false);
		  gen.generate(ngen, &sum, params, result);

		  morefit::fitter_options opts;
		  opts.minuit_printlevel = 0;
		  opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
		  //opts.minimizer = morefit::fitter_options::minimizer_type::TMinuit;
		  opts.analytic_gradient = true;
		  opts.analytic_hessian = true;
		  //opts.analytic_fisher = true;
		  opts.optimize_parameters = true;
		  opts.optimize_dimensions = false;
		  opts.kahan_on_accelerator = false;
		  opts.print_level = 0;
		  morefit::fitter<kernelT, evalT, backendT, blockT> fit(&opts, &backend);
		  fit.fit(&sum, params, &result);
		  for (unsigned int j=0; j<params.size(); j++)
		    if (!params.at(j)->is_constant())
		      pulls.at(j).push_back((params.at(j)->get_value()-params.at(j)->get_start_value())/params.at(j)->get_error());		  

#ifdef WITH_ROOT
		  //copy toy to roofit and fit
		  RooDataSet roofit_data("roofit_data","roofit_data",RooArgSet(roofit_m));
		  for (unsigned int j=0; j<ngen; j++)
		    {
		      roofit_m = result(j,0);
		      roofit_data.add(RooArgSet(roofit_m));
		    }
		  roofit_mb.setVal(mb.get_start_value());
		  roofit_sigma.setVal(sigma.get_start_value());
		  roofit_fsig.setVal(fsig.get_start_value());
		  roofit_alpha.setVal(alpha.get_start_value());
		  roofit_mb.setError(mb.get_step_size());
		  roofit_sigma.setError(sigma.get_step_size());
		  roofit_fsig.setError(fsig.get_step_size());
		  roofit_alpha.setError(alpha.get_step_size());
		  RooFitResult* roofit_result = roofit_model.fitTo(roofit_data, Save(true));
		  const RooArgList& roofit_pars = roofit_result->floatParsFinal();
  
		  for (unsigned int j=0; j<params.size(); j++)
		    if (!params.at(j)->is_constant())
		      {
			int idx = -1;
			for (int k = 0; k < roofit_pars.getSize(); ++k)
			  {                   
			    auto & roofit_par = (RooRealVar &)roofit_pars[k];
			    if (roofit_par.GetName() == params.at(j)->get_name())
			      idx = k;
			  }
			assert(idx != -1);
			double v = ((RooRealVar &)roofit_pars[idx]).getVal();
			double e = ((RooRealVar &)roofit_pars[idx]).getError();
			diffs_values.at(j).push_back(params.at(j)->get_value() - v);
			diffs_errors.at(j).push_back(params.at(j)->get_error() - e);
		      }
		  
#endif		  
		}//ntoys
	      auto t_after_toystudy = std::chrono::high_resolution_clock::now();
	      std::cout << "toystudy takes " << std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count() << " ms in total" << std::endl;
	      runtimes.push_back(std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count());

#ifdef WITH_ROOT
	      if (true)
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
		  c0->Divide(2,2);
		  for (unsigned int j=0; j<params.size(); j++)
		    {
		      c0->cd(j+1);
		      hists[j]->Draw("hist");
		    }
		  c0->Print("pulls.eps", "eps");

		  TH1D* hists_values[params.size()];
		  for (unsigned int i=0; i<params.size(); i++)
		    {
		      hists_values[i] = new TH1D(("values"+std::to_string(i)).c_str(), (";#Delta values("+params.at(i)->get_name()+");").c_str(), 100, -1.0e-2, 1.0e-2);
		      for (unsigned int j=0; j<diffs_values.at(i).size(); j++)
			hists_values[i]->Fill(diffs_values.at(i).at(j));
		    }
		  TCanvas* c1 = new TCanvas("c1", "c1", 1600, 1200);
		  c1->Divide(2,2);
		  for (unsigned int j=0; j<params.size(); j++)
		    {
		      c1->cd(j+1);
		      hists_values[j]->Draw("hist");
		    }
		  c1->Print("diffs_values.eps", "eps");


		  TH1D* hists_errors[params.size()];
		  for (unsigned int i=0; i<params.size(); i++)
		    {
		      hists_errors[i] = new TH1D(("errors"+std::to_string(i)).c_str(), (";#Delta errors("+params.at(i)->get_name()+");").c_str(), 100, -1.0e-2, 1.0e-2);
		      for (unsigned int j=0; j<diffs_errors.at(i).size(); j++)
			hists_errors[i]->Fill(diffs_errors.at(i).at(j));
		    }
		  TCanvas* c2 = new TCanvas("c2", "c2", 1600, 1200);
		  c2->Divide(2,2);
		  for (unsigned int j=0; j<params.size(); j++)
		    {
		      c2->cd(j+1);
		      hists_errors[j]->Draw("hist");
		    }
		  c2->Print("diffs_errors.eps", "eps");



		}
#endif

	    }//end repeats
	  double mean, rms;
	  morefit::mean_rms<double, double>(runtimes, mean, rms);
	  std::cout << "Toy study with " << nstats[n] << " nevents, ms runtime mean " << mean << " rms " << rms << std::endl;
	  means.push_back(mean);
	  rmss.push_back(rms);
	}//end nstats
      std::cout << "_mean[" << npoints << "] = {";
      for (unsigned int i=0; i<npoints; i++)
	std::cout << means.at(i) << (i < npoints -1 ? ", " : "};");
      std::cout << std::endl;
      std::cout << "_rms[" << npoints << "] = {";
      for (unsigned int i=0; i<npoints; i++)
	std::cout << rmss.at(i) << (i < npoints -1 ? ", " : "};");
      std::cout << std::endl;
    }
  
  return 0;
}
