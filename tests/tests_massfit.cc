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

#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooFitResult.h"
#include "RooDataSet.h"
#include "RooAddPdf.h"
#include "RooMCStudy.h"
#include "RooPlot.h"

#endif

int main()
{
  typedef double kernelT;
  typedef double evalT;

  morefit::compute_options compute_opts;
  compute_opts.opencl_platform = 0; compute_opts.opencl_device = 0; //gpu
  //compute_opts.opencl_platform = 1; compute_opts.opencl_device = 0; //cpu  
  compute_opts.print_kernel = true;
  compute_opts.llvm_nthreads = 1;
  compute_opts.llvm_print_intermediate = false;
  compute_opts.llvm_vectorization = true;
  compute_opts.llvm_vectorization_width = 4;
  //compute_opts.llvm_cpu = "x86-64-v4";
  //compute_opts.llvm_tunecpu = "x86-64";
  compute_opts.print_level =2;
  compute_opts.print();
  
  typedef morefit::OpenCLBackend backendT;
  typedef morefit::OpenCLBlock<kernelT, evalT> blockT;
  morefit::OpenCLBackend backend(&compute_opts);

  //typedef morefit::LLVMBackend backendT;
  //typedef morefit::LLVMBlock<kernelT, evalT> blockT;
  //morefit::LLVMBackend backend(&compute_opts);

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

  //check graphing
  if (false)
    {
      exp.prob_normalised()->draw("exp_prob_graph.tex");
      exp.prob_normalised()->diff("alpha")->draw("exp_diff_prob_graph.tex");
      exp.prob_normalised()->diff("alpha")->simplify()->draw("exp_diff_prob_graph_simplified.tex");

      gaus.prob_normalised()->draw("gaus_prob_graph.tex");
      gaus.prob_normalised()->diff("mb")->draw("gaus_diff_prob_graph.tex");
      gaus.prob_normalised()->diff("mb")->simplify()->draw("gaus_diff_prob_graph_simplified.tex");
      
      sum.prob()->draw("prob_graph.tex");
      sum.prob()->simplify()->draw("prob_simplegraph.tex");
      sum.logprob()->draw("logprob_graph.tex");
      sum.logprob()->simplify()->draw("logprob_simplegraph.tex");

      std::vector<std::string> param_names;
      std::vector<evalT> param_values;
      for (auto param : params)
	{
	  param_names.push_back(param->get_name());
	  param_values.push_back(param->get_value());      
	}
      sum.prob_normalised()->substitute(param_names, param_values)->simplify()->draw("gen_graph.tex");

      std::vector<morefit::parameter<evalT>*> params2({&alpha});
      morefit::generator_options gen_opts;
      gen_opts.rndtype = morefit::generator_options::randomization_type::on_accelerator;
      //gen_opts.rndtype = morefit::generator_options::randomization_type::on_host;
      gen_opts.print_level = 2;//2
      gen_opts.print();

      morefit::fitter_options fit_opts;
      fit_opts.minuit_printlevel = 2;//2
	      
      fit_opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
      //fit_opts.optimize_dimensions = true;
      fit_opts.optimize_dimensions = false;
      //fit_opts.optimize_parameters = true;
      fit_opts.optimize_parameters = false;
      fit_opts.analytic_gradient = false;
      fit_opts.analytic_hessian = false;
      
      //fit_opts.postrun_hesse = false;
      //fit_opts.analytic_gradient = true;
      //fit_opts.analytic_hessian = true;
      
      fit_opts.kahan_on_accelerator = true;
      //fit_opts.kahan_on_accelerator = false;
      fit_opts.print_level = 2;//2
      fit_opts.analytic_fisher = false;
      fit_opts.print();
      
      morefit::toystudy<kernelT, evalT, backendT, blockT > toy(&fit_opts, &gen_opts, &backend, &rnd);
      toy.toy(&exp, params2, {&m}, 10, 10000, false);

      return 0;
    }
  
  //check operator==
  if (false)
    {
      std::unique_ptr<morefit::ComputeGraphNode<kernelT,evalT>> c = morefit::Constant<kernelT,evalT>(9.0);
      std::unique_ptr<morefit::ComputeGraphNode<kernelT,evalT>> c2 = morefit::Constant<kernelT,evalT>(9.0);
      std::unique_ptr<morefit::ComputeGraphNode<kernelT,evalT>> x = morefit::Variable<kernelT,evalT>("varx");
      std::unique_ptr<morefit::ComputeGraphNode<kernelT,evalT>> proda = c2->copy()+x->copy();
      std::unique_ptr<morefit::ComputeGraphNode<kernelT,evalT>> prodb = x->copy()+c->copy();
      proda->print();
      prodb->print();
      if (proda == prodb)
	std::cout << "TRUE" << std::endl;
      else
	std::cout << "FALSE" << std::endl;	
      return 0;
    }
  
  //kernel output
  if (false)
    {
      std::cout << "FULL KERNEL " << sum.prob_normalised()->get_kernel() << std::endl;
      std::cout << "SIMPLIFIED KERNEL " << sum.prob_normalised()->simplify()->get_kernel() << std::endl;

      std::vector<std::string> param_names;
      std::vector<evalT> param_values;
      for (auto param : params)
	{
	  std::cout << "param name: " << param->get_name() << " param value: " << param->get_value() << std::endl;
	  param_names.push_back(param->get_name());
	  param_values.push_back(param->get_value());      
	}
      std::cout << "GEN KERNEL " << sum.prob_normalised()->substitute(param_names, param_values)->simplify()->get_kernel() << std::endl;
      std::cout << "KERNEL NORM " << sum.norm()->substitute(param_names, param_values)->simplify()->get_kernel() << std::endl;
    }
  
  //crosscheck random number genreator
  if (false)  
    {
#ifdef WITH_ROOT
      //TRandom3* rootRnd = new TRandom3(229387429);
      morefit::Random3 rootRnd(229387429);
      //rnd2->SetSeed(229387429);
      morefit::MersenneTwister mersenneRnd(229387429);
      morefit::Xoshiro256pp xoshiroRnd(229387429);
      morefit::PCG64DXSM pcgRnd(229387429);
      TH1D* h1 = new TH1D("h1", "TRandom3;r;", 1000, 0.0, 1.0);
      TH1D* h2 = new TH1D("h2", "MersenneTwister;r;", 1000, 0.0, 1.0);
      TH1D* h3 = new TH1D("h3", "XoshiroRnd++;r;", 1000, 0.0, 1.0);
      TH1D* h4 = new TH1D("h4", "pcgRnd;r;", 1000, 0.0, 1.0);
      // TH1D* h1 = new TH1D("h1", "TRandom3;r;", 1000, -5.0, 5.0);
      // TH1D* h2 = new TH1D("h2", "MersenneTwister;r;", 1000, -5.0, 5.0);
      // TH1D* h3 = new TH1D("h3", "XoshiroRnd++;r;", 1000, -5.0, 5.0);
      // TH1D* h4 = new TH1D("h4", "pcgRnd++;r;", 1000, -5.0, 5.0);
      unsigned int nnumbers = 10000000;
      for (unsigned int i=0; i<nnumbers; i++)
	{
	  double a = rootRnd.random();
	  double b = mersenneRnd.random();
	  double c = xoshiroRnd.random();
	  double d = pcgRnd.random();	  
	  // double a = rootRnd->Gaus();
	  // double b = mersenneRnd.gaus();
	  // double c = xoshiroRnd.gaus();
	  h1->Fill(a);
	  h2->Fill(b);
	  h3->Fill(c);
	  h4->Fill(d);
	  //std::cout << a << " " << b << " diff " << a-b << std::endl;
	}
      TCanvas* c0 = new TCanvas("c0", "c0", 1200, 1200);
      c0->Divide(2,2);
      c0->cd(1);
      h1->Draw();
      c0->cd(2);
      h2->Draw();
      c0->cd(3);
      h3->Draw();
      c0->cd(4);
      h4->Draw();
      c0->Print("rnd.eps", "eps");
#endif
      return 0;
    }
  
  if (false)
    {
      std::cout << "STD KERNEL " << sum.logprob_normalised()->simplify()->get_kernel() << std::endl;
      std::vector<std::string> buffernames;
      std::vector<std::unique_ptr<morefit::ComputeGraphNode<kernelT, evalT>>> bufferexpressions;
      std::vector<std::string> variables = {"mb", "sigma", "fsig", "alpha"};
      std::cout << "OPT KERNEL " << sum.logprob_normalised()->simplify()->optimize_buffering_constant_terms(buffernames, bufferexpressions, variables)->get_kernel() << std::endl;
      for (unsigned int i=0; i<buffernames.size(); i++)
	std::cout << "KERNEL " << buffernames.at(i) << ": " << bufferexpressions.at(i)->get_kernel() << std::endl;
      return 0;
    }
#ifdef WITH_ROOT
  //roofittoystudy for fit of multiple gaussians
  if (false)
    {
      using namespace RooFit;
      unsigned int ngaus = 20;
      unsigned int ngen = 100000;
      unsigned int nruns = 100;
      double mmin = 5.0;
      double mmax = 7.0;

      RooRealVar m("m", "m", mmin, mmax);
      std::vector<RooRealVar> mbs;
      std::vector<RooRealVar> sigmas;
      std::vector<RooRealVar> fsigs;
      for (unsigned int i=0; i<ngaus; i++)
	{
	  mbs.push_back(RooRealVar(("mb_"+std::to_string(i)).c_str(), "mb", mmin+(mmax-mmin)/double(ngaus)*(i+0.5), 5.0, 7.0));
	  sigmas.push_back(RooRealVar(("sigma_"+std::to_string(i)).c_str(), "sigma", 0.02, 0.005, 0.130));
	  fsigs.push_back(RooRealVar(("fsig_"+std::to_string(i)).c_str(), "fsig", 0.5/ngaus, 0.0, 1.0));
	}
      RooRealVar alpha("alpha", "alpha", -1.0, -10.0, 10.0);
      RooExponential expo("expo", "expo", m, alpha);

      std::vector<RooGaussian> gaussians;
      for (unsigned int i=0; i<ngaus; i++)
	gaussians.push_back(RooGaussian(("gauss_"+std::to_string(i)).c_str(), "gauss", m, mbs.at(i), sigmas.at(i)));

      RooArgList pdflist;
      for (unsigned int i=0; i<ngaus; i++)
	pdflist.add(gaussians.at(i));
      pdflist.add(expo);
      RooArgList fsiglist;
      for (unsigned int i=0; i<ngaus; i++)
	fsiglist.add(fsigs.at(i));
      RooAddPdf model("model","model", pdflist, fsiglist);

      auto t_before_toystudy = std::chrono::high_resolution_clock::now();
      RooMCStudy *mcstudy =
	new RooMCStudy(model, m, Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("cuda")));
      mcstudy->generateAndFit(nruns, ngen);
      auto t_after_toystudy = std::chrono::high_resolution_clock::now();
      std::cout << "toystudy takes " << std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count() << " ms in total" << std::endl;
    }
#endif  
  //toystudy for fit of multiple gaussians
  if (true)
    {
      compute_opts.llvm_vectorization = false;
      unsigned int ngaus = 20;
      unsigned int ngen = 100000;
      unsigned int nruns = 100;
      double mmin = 5.0;
      double mmax = 7.0;
      morefit::dimension<evalT> m("m", "#it{m} [GeV/#it{c}^{2}]", mmin, mmax, false);
      std::vector<morefit::parameter<evalT>*> mbs;
      std::vector<morefit::parameter<evalT>*> sigmas;
      std::vector<morefit::parameter<evalT>*> fsigs;
      for (unsigned int i=0; i<ngaus; i++)
	{
	  mbs.push_back(new morefit::parameter<evalT>("mb_"+std::to_string(i), "m(B^{+})", mmin+(mmax-mmin)/double(ngaus)*(i+0.5), 5.0, 7.0, 0.01, false));
	  sigmas.push_back(new morefit::parameter<evalT>("sigma_"+std::to_string(i), "\\sigma(B^{+})", 0.02, 0.005, 0.130, 0.001, false));
	  fsigs.push_back(new morefit::parameter<evalT>("fsig_"+std::to_string(i), "f_{\\mathrm{sig}}", 0.5/ngaus, 0.0, 1.0, 0.01, false));
	}
      morefit::parameter<evalT> alpha("alpha", "\\alpha_{\\mathrm{bkg}}", -1.0, -10.0, 10.0, 0.01, false);
      std::vector<morefit::PDF<kernelT, evalT>*> summands;
      for (unsigned int i=0; i<ngaus; i++)
	summands.push_back(new morefit::GaussianPDF<kernelT, evalT>(&m, mbs.at(i), sigmas.at(i)));
      morefit::ExponentialPDF<kernelT, evalT> exp(&m, &alpha);
      summands.push_back(&exp);
      
      morefit::SumPDF<kernelT, evalT> sums(summands, fsigs);
      std::vector<morefit::parameter<evalT>*> paramss;
      paramss.insert(paramss.end(), fsigs.begin(), fsigs.end());
      paramss.insert(paramss.end(), mbs.begin(), mbs.end());
      paramss.insert(paramss.end(), sigmas.begin(), sigmas.end());
      paramss.push_back(&alpha);
            
      morefit::generator_options gen_opts;
      gen_opts.rndtype = morefit::generator_options::randomization_type::on_accelerator;
      //gen_opts.rndtype = morefit::generator_options::randomization_type::on_host;
      gen_opts.print_level = 0;
      gen_opts.print();
      
      morefit::fitter_options fit_opts;
      fit_opts.minuit_maxiterations = 40000;
      fit_opts.minuit_maxcalls = 40000;
      fit_opts.minuit_printlevel = 2;
      fit_opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
      fit_opts.optimize_dimensions = false;
      fit_opts.optimize_parameters = true;
      fit_opts.analytic_gradient = true;
      fit_opts.analytic_hessian = true;      
      fit_opts.kahan_on_accelerator = true;
      fit_opts.print_level = 0;//2
      fit_opts.analytic_fisher = false;
      fit_opts.print();

      //first perform a single fit and plot
      if (false)
	{
	  morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
	  morefit::EventVector<kernelT, evalT> result({&m}, ngen);  
	  gen.generate(ngen, &sums, paramss, result);      
	  morefit::fitter<kernelT, evalT, backendT, blockT > fit(&fit_opts, &backend);
	  fit.fit(&sums, paramss, &result);
	  morefit::plotter_options plot_opts;
	  //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
	  plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
	  plot_opts.print_level = 0;
	  plot_opts.plot_pulls = true;
	  morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
	  plot.plot(&sums, paramss, &result, "m", "plot_m.eps", "eps", 200);
	}
      
      auto t_before_toystudy = std::chrono::high_resolution_clock::now();
      morefit::toystudy<kernelT, evalT, backendT, blockT > toy(&fit_opts, &gen_opts, &backend, &rnd);
      toy.toy(&sums, paramss, {&m}, nruns, ngen, true);
      auto t_after_toystudy = std::chrono::high_resolution_clock::now();
      std::cout << "toystudy takes " << std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count() << " ms in total" << std::endl;
    }
  
  //explicit generation followed by fit
  if (false)
    {
      unsigned int nrepeats = 1;
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
	      unsigned int nruns = 1000;
      
	      morefit::generator_options gen_opts;
	      gen_opts.rndtype = morefit::generator_options::randomization_type::on_accelerator;
	      //gen_opts.rndtype = morefit::generator_options::randomization_type::on_host;
      
	      morefit::generator<kernelT, evalT, backendT, blockT > gen(&gen_opts, &backend, &rnd);
	      std::vector<std::vector<double>> pulls(params.size(), std::vector<double>());
	      for (unsigned int i=0; i<nruns; i++)
		{
		  std::cout << "toy no "<< i << std::endl;
		  for (unsigned int j = 0; j < params.size(); j++)
		    params.at(j)->set_value(params.at(j)->get_start_value());

		  morefit::EventVector<kernelT, evalT> result({&m}, ngen, false);
		  gen.generate(ngen, &sum, params, result);
		  morefit::dimension<evalT> weight_dim("weight", -10.0, 10.0);
		  std::vector<morefit::dimension<evalT>*> dim_vector;
		  dim_vector.push_back(&weight_dim);
		  result.add_dimensions(dim_vector);
		  int idx = result.dim_index("weight");
		  for (unsigned int k = 0; k < result.nevents(); k++)
		    result(k,idx) = 2.0;
		  result.set_event_weight("weight");

		  morefit::fitter_options opts;
		  opts.minuit_printlevel = 0;
		  opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
		  //opts.minimizer = morefit::fitter_options::minimizer_type::TMinuit;
		  //opts.analytic_gradient = true;
		  //opts.analytic_hessian = true;
		  //opts.analytic_fisher = true;
		  opts.correct_weighted_fit = false;
		  opts.optimize_parameters = true;
		  opts.analytic_gradient = false;
		  opts.optimize_dimensions = false;
		  opts.kahan_on_accelerator = false;
		  opts.print_level = 0;
		  morefit::fitter<kernelT, evalT, backendT, blockT> fit(&opts, &backend);
		  fit.fit(&sum, params, &result);
	  
		  for (unsigned int j=0; j<params.size(); j++)
		    if (!params.at(j)->is_constant())
		      pulls.at(j).push_back((params.at(j)->get_value()-params.at(j)->get_start_value())/params.at(j)->get_error());		  
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
  
  //generation plots
  if (false)
    {
      unsigned int ngen = 10000;
      std::cout <<"generating" << std::endl;
      morefit::generator_options gen_opts;
      morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
      morefit::EventVector<kernelT, evalT> result({&m}, ngen);  
      gen.generate(ngen, &sum, params, result);      
      
      std::cout <<"fitting" << std::endl;      
      morefit::fitter_options opts;
      morefit::fitter<kernelT, evalT, backendT, blockT > fit(&opts, &backend);
      fit.fit(&sum, params, &result);

#ifdef WITH_ROOT
      std::cout << "plotting" <<std::endl;
      TCanvas* c0 = new TCanvas("c0", "c0", 1600, 1200);
      TH1D* masshist = new TH1D("masshist", ";m [GeV];", 100, m.get_min(), m.get_max());
      for (unsigned int l=0; l<result.nevents(); l++)
	masshist->Fill(result(l,0));
      TH1D* masspdf = new TH1D("masshist", ";m [GeV];", 100, m.get_min(), m.get_max());
      for (unsigned int i=0; i<100; i++)
	{
	  double sigma = 0.06;
	  double fsig = 0.3;
	  double alpha = -1.0;
	  double cur = m.get_min() + (i+0.5)*(m.get_max()-m.get_min())/100.0;
	  double mb = 5.28;
	  double dm = (m.get_max()-m.get_min())/100.0;
	  double p = fsig * 1.0/sqrt(2.0*TMath::Pi())/sigma* TMath::Exp(-0.5*(cur-mb)*(cur-mb)/sigma/sigma) / (0.5*TMath::Erf((mb-m.get_min())/(sqrt(2.0)*sigma))-0.5*TMath::Erf((mb-m.get_max())/(sqrt(2.0)*sigma)))
	    + (1.0-fsig) * TMath::Exp(alpha*cur)/(TMath::Exp(alpha*m.get_min())-TMath::Exp(alpha*m.get_max())); 	  
	    masspdf->SetBinContent(i+1, p * masshist->Integral() * dm);
	}
      double ymax = masspdf->GetMaximum();
      gROOT->SetStyle("Plain");
      gStyle->SetOptFit(0);
      gStyle->SetOptStat(0);      
      c0->cd();
      masshist->SetMinimum(0.0);
      masshist->Draw("e");
      masspdf->SetLineColor(2);
      masspdf->SetLineWidth(2);      
      masspdf->Draw("lcsame");
      TLine * l = new TLine();
      l->SetLineColor(2);
      l->DrawLine(m.get_min(), ymax, m.get_max(), ymax);
      l->DrawLine(5.6, 0.0, 5.6, ymax);
      c0->Print("mass.eps", "eps");
#endif
    }

  //plotting check
  if (false)
    {
      unsigned int ngen = 100000;
      std::cout <<"generating" << std::endl;
      morefit::generator_options gen_opts;
      gen_opts.print();
      morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
      morefit::EventVector<kernelT, evalT> result({&m}, ngen);  
      gen.generate(ngen, &sum, params, result);      
      
      std::cout <<"fitting" << std::endl;      
      morefit::fitter_options opts;
      opts.minuit_printlevel = 2;
      opts.analytic_gradient = true;
      opts.analytic_hessian = true;
      opts.print();
      morefit::fitter<kernelT, evalT, backendT, blockT > fit(&opts, &backend);
      fit.fit(&sum, params, &result);

      morefit::plotter_options plot_opts;
      //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
      plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
      plot_opts.print_level = 2;
      plot_opts.plot_pulls = true;
      morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
      //plot.plot(&sum, params, &result, "m", "plot_m.C", "C", 100);
      //plot.plot(&sum, params, &result, "m", "plot_m.py", "py", 100);
      plot.plot(&sum, params, &result, "m", "plot_m.eps", "eps", 100);

      std::vector<std::string> param_names;
      for (unsigned int i=0; i<params.size(); i++)
	param_names.push_back(params.at(i)->get_name());
      std::vector<double> param_values;
      for (unsigned int i=0; i<params.size(); i++)
	param_values.push_back(params.at(i)->get_value());

      return 0;
    }

  
  return 0;
}
