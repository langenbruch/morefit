#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <chrono>

#include <CL/cl.h>

#include "morefit.hh"
#include "random.hh"

#ifdef WITH_ROOT
#include "TRandom3.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TLine.h"
#include "TStyle.h"
#endif

int main()
{

  typedef double kernelT;
  typedef double evalT;

  morefit::compute_options compute_opts;
  compute_opts.opencl_platform = 0; compute_opts.opencl_device = 0;  //gpu
  compute_opts.print_kernel = true;
  compute_opts.llvm_nthreads = 1;
  compute_opts.llvm_print_intermediate = false;
  compute_opts.llvm_vectorization = true;
  compute_opts.llvm_vectorization_width = 4;
  //compute_opts.llvm_cpu = "x86-64-v4";
  //compute_opts.llvm_tunecpu = "x86-64";
  compute_opts.print_level =2;
  compute_opts.print();
  
  //typedef morefit::OpenCLBackend backendT;
  //typedef morefit::OpenCLBlock<kernelT, evalT> blockT;
  //morefit::OpenCLBackend backend(&compute_opts);

  typedef morefit::LLVMBackend backendT;
  typedef morefit::LLVMBlock<kernelT, evalT> blockT;
  morefit::LLVMBackend backend(&compute_opts);

  morefit::dimension<evalT> m("m", "#it{m} [GeV/#it{c}^{2}]", 5.0, 7.0, false);
  morefit::parameter<evalT> mb("mb", "m(B^{+})", 5.28, 5.0, 6.0, 0.01, false);
  morefit::parameter<evalT> sigma("sigma", "\\sigma(B^{+})", 0.06, 0.005, 0.130, 0.001, false);
  morefit::parameter<evalT> fsig("fsig", "f_{\\mathrm{sig}}", 0.3, 0.0, 1.0, 0.01, false);  
  morefit::parameter<evalT> alpha("alpha", "\\alpha_{\\mathrm{bkg}}", -1.0, -10.0, 10.0, 0.01, false);
  
  morefit::GaussianPDF<kernelT, evalT> gaus(&m, &mb, &sigma);
  morefit::ExponentialPDF<kernelT, evalT> exp(&m, &alpha);
  morefit::SumPDF<kernelT, evalT> sum(&gaus, &exp, &fsig);
  std::vector<morefit::parameter<evalT>*> params({&mb, &sigma, &fsig, &alpha});

  morefit::Xoshiro128pp rnd;
  rnd.setSeed(229387429ULL);

  //benchmarking
  if (true)
    {
      unsigned int nrepeats = 10;
      const unsigned int npoints = 4;
      unsigned int nstats[npoints] = {1000, 10000, 100000, 1000000};      
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
	      fit_opts.minuit_printlevel = 0;//2
	      
	      fit_opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
	      //fit_opts.optimize_dimensions = true;
	      fit_opts.optimize_dimensions = false;
	      fit_opts.optimize_parameters = true;
	      //fit_opts.optimize_parameters = false;
	      fit_opts.analytic_gradient = false;
	      fit_opts.analytic_hessian = false;

	      //fit_opts.postrun_hesse = false;
	      //fit_opts.analytic_gradient = true;
	      //fit_opts.analytic_hessian = true;

	      fit_opts.kahan_on_accelerator = true;
	      //fit_opts.kahan_on_accelerator = false;
	      fit_opts.print_level = 0;//2
	      fit_opts.analytic_fisher = false;
	      fit_opts.print();

	      morefit::toystudy<kernelT, evalT, backendT, blockT > toy(&fit_opts, &gen_opts, &backend, &rnd);
	      toy.toy(&sum, params, {&m}, 100, nstats[n], q==0);
	      
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
      return 0;
    }
  return 0;
}
