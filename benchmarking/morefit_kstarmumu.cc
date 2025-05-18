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

  //benchmarking
  if (true)
    {
      unsigned int nrepeats = 10;
      const unsigned int npoints = 4;
      unsigned long int nstats[npoints] = {1000, 10000, 100000, 1000000};
      //const unsigned int npoints = 5;
      //unsigned int nstats[npoints] = {1000, 10000, 100000, 1000000, 10000000};
      std::vector<double> means, rmss;
      for (unsigned int n=0; n<npoints; n++)
	{
	  std::vector<double> runtimes;
	  for (unsigned int q=0; q<nrepeats; q++)
	    {
	      auto t_before_toystudy = std::chrono::high_resolution_clock::now();
	      morefit::generator_options gen_opts;
	      gen_opts.rndtype = morefit::generator_options::randomization_type::on_accelerator;
	      //gen_opts.rndtype = morefit::generator_options::randomization_type::on_host;
	      gen_opts.print_level = 0;//2
	      
	      gen_opts.print();
	      morefit::fitter_options fit_opts;
	      fit_opts.minuit_printlevel = 0;
	      
	      fit_opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
	      fit_opts.optimize_dimensions = true;
	      //fit_opts.optimize_dimensions = false;
	      fit_opts.optimize_parameters = true;
	      //fit_opts.analytic_gradient = false;
	      //fit_opts.analytic_hessian = false;

	      fit_opts.analytic_gradient = true;
	      fit_opts.analytic_hessian = true;

	      fit_opts.kahan_on_accelerator = true;
	      fit_opts.print_level = 0;//2
	      fit_opts.print();
	      
	      morefit::toystudy<kernelT, evalT, backendT, blockT > toy(&fit_opts, &gen_opts, &backend, &rnd);
	      toy.toy(&kstarmumu, params, {&ctl, &ctk, &phi}, 100, nstats[n], q==0);
	      
	      auto t_after_toystudy = std::chrono::high_resolution_clock::now();
	      std::cout << "toystudy takes " << std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count() << " ms in total" << std::endl;
	      runtimes.push_back(std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count());	      
	      
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
