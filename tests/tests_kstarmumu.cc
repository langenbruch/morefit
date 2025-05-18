#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <chrono>

#include <CL/cl.h>

#include "morefit.hh"
#include "random.hh"

#ifdef WITH_ROOT
#include "TCanvas.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TStyle.h"
#endif


int main()
{

  typedef double kernelT;
  typedef double evalT;
  
  morefit::compute_options compute_opts;
  compute_opts.opencl_platform = 0; compute_opts.opencl_device = 0;  
  compute_opts.llvm_nthreads = 1;
  compute_opts.print_kernel = true;
  compute_opts.llvm_print_intermediate = false;
  compute_opts.print();
  
  //typedef morefit::OpenCLBackend backendT;
  //typedef morefit::OpenCLBlock<kernelT, evalT> blockT;
  //morefit::OpenCLBackend backend(&compute_opts);
  
  typedef morefit::LLVMBackend backendT;
  typedef morefit::LLVMBlock<kernelT, evalT> blockT;  
  morefit::LLVMBackend backend(&compute_opts);
  
  morefit::dimension<evalT> ctl("ctl", "cos(\\Theta_{l})", -1.0, 1.0, false);
  morefit::dimension<evalT> ctk("ctk", "cos(\\Theta_{K})", -1.0, 1.0, false);
  morefit::dimension<evalT> phi("phi", "\\phi", -M_PI, +M_PI, false);
  morefit::parameter<evalT> Fl("Fl", "F_{\\mathrm{L}}", 0.6, 0.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S3("S3", "S_{3}", 0.0, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S4("S4", "S_{4}", 0.0, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S5("S5", "S_{5}", 0.0, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> Afb("Afb", "A_{\\mathrm{FB}}", 0.0, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S7("S7", "S_{7}", 0.0, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S8("S8", "S_{8}", 0.0, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S9("S9", "S_{9}", 0.0, -1.0, 1.0, 0.01, false);

  morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
  std::vector<morefit::parameter<evalT>*> params({&Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9});
  
  morefit::Xoshiro128pp rnd;
  rnd.setSeed(int64_t(229387429));

  //produce graphs
  if (false)
    {
      kstarmumu.logprob()->draw("graph.tex");
      kstarmumu.logprob()->simplify()->draw("simplegraph.tex");

      std::vector<std::string> param_names;
      std::vector<evalT> param_values;
      for (auto param : params)
	{
	  param_names.push_back(param->get_name());
	  param_values.push_back(param->get_value());      
	}
      kstarmumu.prob_normalised()->substitute(param_names, param_values)->simplify()->draw("gen_graph.tex");      
      return 0;
    }
  
  //kernel output
  if (false)
    {
      std::cout << "FULL KERNEL " << kstarmumu.prob_normalised()->get_kernel() << std::endl;
      std::cout << "SIMPLIFIED KERNEL " << kstarmumu.prob_normalised()->simplify()->get_kernel() << std::endl;
      
      std::vector<std::string> param_names;
      std::vector<evalT> param_values;
      for (auto param : params)
	{
	  std::cout << "param name: " << param->get_name() << " param value: " << param->get_value() << std::endl;
	  param_names.push_back(param->get_name());
	  param_values.push_back(param->get_value());      
	}
    }

  //check plotting
  if (false)
    {
      unsigned int ngen = 100000;
      std::cout <<"generating" << std::endl;
      morefit::generator_options gen_opts;
      
      morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
      morefit::EventVector<kernelT, evalT> result({&ctl, &ctk, &phi}, ngen);  
      gen.generate(ngen, &kstarmumu, params, result);      
      
      std::cout <<"fitting" << std::endl;      
      morefit::fitter_options opts;
      opts.minuit_printlevel = 2;
      opts.analytic_gradient = true;
      opts.analytic_hessian = true;
      opts.print();
      morefit::fitter<kernelT, evalT, backendT, blockT > fit(&opts, &backend);
      fit.fit(&kstarmumu, params, &result);//TODO FIXME

      morefit::plotter_options plot_opts;
      //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
      plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
      plot_opts.print_level = 2;
      plot_opts.plot_pulls = true;
      //plot_opts.pull_fraction = 0.25;
      morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
      plot.plot(&kstarmumu, params, &result, "ctl", "plot_ctl.eps", "eps", 100);
      plot.plot(&kstarmumu, params, &result, "ctk", "plot_ctk.eps", "eps", 100);
      plot.plot(&kstarmumu, params, &result, "phi", "plot_phi.eps", "eps", 100);

      std::vector<std::string> param_names;
      for (unsigned int i=0; i<params.size(); i++)
	param_names.push_back(params.at(i)->get_name());
      std::vector<double> param_values;
      for (unsigned int i=0; i<params.size(); i++)
	param_values.push_back(params.at(i)->get_value());

      return 0;
    }
  
  //generate and fit unoptimized  
  if (false)
    {

      unsigned int nrepeats = 10;
      const unsigned int npoints = 4;
      unsigned int nstats[npoints] = {1000, 10000, 100000, 1000000};
      std::vector<double> means, rmss;
      for (unsigned int n=0; n<npoints; n++)
	{
	  std::vector<double> runtimes;
	  for (unsigned int q=0; q<nrepeats; q++)
	    {
	      auto t_before_toystudy = std::chrono::high_resolution_clock::now();

	      unsigned int ngen = nstats[n];
	      unsigned int nruns = 100;
	      morefit::generator_options gen_opts;
	      gen_opts.rndtype = morefit::generator_options::randomization_type::on_accelerator;
	      //gen_opts.rndtype = morefit::generator_options::randomization_type::on_host;
	      gen_opts.print();
	      
	      morefit::generator<kernelT, evalT, backendT, blockT > gen(&gen_opts, &backend, &rnd);
      
	      std::vector<std::vector<double>> pulls(params.size(), std::vector<double>());
	      for (unsigned int i=0; i<nruns; i++)
		{
		  std::cout << "toy no "<< i << std::endl;
		  for (unsigned int j = 0; j < params.size(); j++)
		    params.at(j)->set_value(params.at(j)->get_start_value());

	  
		  morefit::EventVector<kernelT, evalT> result({&ctl, &ctk, &phi}, ngen);	  
		  gen.generate(ngen, &kstarmumu, params, result);
	  
		  morefit::fitter_options opts;
		  opts.minuit_printlevel = 2;
		  
		  opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
		  opts.optimize_dimensions = true;
		  //opts.optimize_dimensions = false;
		  opts.optimize_parameters = true;
		  opts.analytic_gradient = true;
		  //opts.analytic_gradient = false;
		  opts.kahan_on_accelerator = true;
		  //opts.kahan_on_accelerator = false;
		  opts.print_level = 2;
		  opts.print();
		  
		  morefit::fitter<kernelT, evalT, backendT, blockT> fit(&opts, &backend);
		  fit.fit(&kstarmumu, params, &result);
		  for (unsigned int j=0; j<params.size(); j++)
		    if (!params.at(j)->is_constant())
		      pulls.at(j).push_back((params.at(j)->get_value()-params.at(j)->get_start_value())/params.at(j)->get_error());
		}
	      auto t_after_toystudy = std::chrono::high_resolution_clock::now();
	      std::cout << "toystudy takes " << std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count() << " ms in total" << std::endl;
	      runtimes.push_back(std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count());
      
#ifdef WITH_ROOT
	      if (true && q==0)
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
		  c0->Print(("pulls_"+std::to_string(n)+".eps").c_str(), "eps");
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
