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
#include "TLegend.h"
#endif

#ifdef WITH_ONNX
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

void get_meanrms(std::vector<double> a, std::vector<double> b, double & mean, double & rms)
{
  assert(a.size() == b.size());
  std::vector<double> diff(a);
  for (unsigned int i=0; i<diff.size(); i++)
    diff.at(i) -= b.at(i);
  mean = 0.0;
  for (unsigned int i=0; i<diff.size(); i++)
    mean += diff.at(i)/diff.size();
  rms = 0.0;
  for (unsigned int i=0; i<diff.size(); i++)
    rms += (diff.at(i)-mean)*(diff.at(i)-mean)/diff.size();
  rms = sqrt(rms);
  //std::cout << prefix << " mean " << mean << " rms " << rms << std::endl;
  return;
}

void print_table_meanrms(std::vector<std::vector<double>> nominals, std::vector<std::vector<std::vector<double>>> values, std::vector<std::string> observables, std::vector<std::string> methods)
{
  assert(methods.size() == values.size());
  assert(observables.size() == nominals.size());
  std::cout << "\\begin{tabular}{l";
  for (unsigned int i=0; i<observables.size(); i++)
    std::cout << "rr";
  std::cout << "}\\hline" << std::endl;
  std::cout <<" &";
  for (unsigned int i=0; i<observables.size(); i++)
    std::cout << "\\multicolumn{2}{c}{" << observables.at(i) << "}" << (i<observables.size()-1 ? " & " : " ");
  std::cout << "\\\\" << std::endl;
  std::cout <<" &";
  for (unsigned int i=0; i<observables.size(); i++)
    std::cout << " $\\mu$ & $\\sigma$" << (i<observables.size()-1 ? " & " : " ");
  std::cout << "\\\\ \\hline\\hline" << std::endl;
  
  
  for (unsigned int i=0; i<methods.size(); i++)//i method idx
    {
      std::cout << methods.at(i) << " & ";
      for (unsigned int j=0; j<values.at(i).size(); j++)//j observable index
	{
	  std::string name = observables.at(j);
	  double mean, rms;
	  std::vector<double> v = values.at(i).at(j);
	  get_meanrms(v, nominals.at(j), mean, rms);
	  std::cout << std::fixed << std::setprecision(1) << mean/1.0e-3 << " & " << std::fixed << std::setprecision(1) << rms/1.0e-3 << (j<values.at(i).size() - 1 ? " & " : " ");
	}
      std::cout << "\\\\" << std::endl;
    }
  std::cout << "\\hline\\end{tabular}" << std::endl;
  std::cout << "\\caption{Mean $[10^{-3}]$ and RMS $[10^{-3}]$ of the distribution of the fitted angular observables of the specific methods to calculate the normalisation integral, subtracted by the analytic normalisation.}" << std::endl;
}

void print_table_mean(std::vector<std::vector<double>> nominals, std::vector<std::vector<std::vector<double>>> values, std::vector<std::string> observables, std::vector<std::string> methods)
{
  assert(methods.size() == values.size());
  assert(observables.size() == nominals.size());
  std::cout << "\\begin{tabular}{l";
  for (unsigned int i=0; i<observables.size(); i++)
    std::cout << "r";
  std::cout << "}\\hline" << std::endl;
  std::cout << " &";
  for (unsigned int i=0; i<observables.size(); i++)
    std::cout << observables.at(i) << (i<observables.size()-1 ? " & " : " ");
  std::cout << "\\\\ \\hline\\hline" << std::endl;
  for (unsigned int i=0; i<methods.size(); i++)//i method idx
    {
      std::cout << methods.at(i) << " & ";
      for (unsigned int j=0; j<values.at(i).size(); j++)//j observable index
	{
	  std::string name = observables.at(j);
	  double mean, rms;
	  std::vector<double> v = values.at(i).at(j);
	  get_meanrms(v, nominals.at(j), mean, rms);
	  std::cout << std::fixed << std::setprecision(1) << mean/1.0e-3 << (j<values.at(i).size() - 1 ? " & " : " ");
	}
      std::cout << "\\\\" << std::endl;
    }
  std::cout << "\\hline\\end{tabular}" << std::endl;
  std::cout << "\\caption{Mean $[10^{-3}]$ of the distribution of the fitted angular observables of the specific methods to calculate the normalisation integral, subtracted by the analytic normalisation.}" << std::endl;
}

void print_table_rms(std::vector<std::vector<double>> nominals, std::vector<std::vector<std::vector<double>>> values, std::vector<std::string> observables, std::vector<std::string> methods)
{
  assert(methods.size() == values.size());
  assert(observables.size() == nominals.size());
  std::cout << "\\begin{tabular}{l";
  for (unsigned int i=0; i<observables.size(); i++)
    std::cout << "r";
  std::cout << "}\\hline" << std::endl;
  std::cout <<" &";
  for (unsigned int i=0; i<observables.size(); i++)
    std::cout << observables.at(i) << (i<observables.size()-1 ? " & " : " ");
  std::cout << "\\\\ \\hline\\hline" << std::endl;
  for (unsigned int i=0; i<methods.size(); i++)//i method idx
    {
      std::cout << methods.at(i) << " & ";
      for (unsigned int j=0; j<values.at(i).size(); j++)//j observable index
	{
	  std::string name = observables.at(j);
	  double mean, rms;
	  std::vector<double> v = values.at(i).at(j);
	  get_meanrms(v, nominals.at(j), mean, rms);
	  std::cout << std::fixed << std::setprecision(1) << rms/1.0e-3 << (j<values.at(i).size() - 1 ? " & " : " ");
	}
      std::cout << "\\\\" << std::endl;
    }
  std::cout << "\\hline\\end{tabular}" << std::endl;
  std::cout << "\\caption{RMS $[10^{-3}]$ of the distribution of the fitted angular observables of the specific methods to calculate the normalisation integral, subtracted by the analytic normalisation.}" << std::endl;
}

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

  double genfl = 0.5;
  double gens3 = 0.0;
  double gens4 = 0.0;
  double gens5 = 0.0;
  double genafb = 0.0;
  double gens7 = 0.0;
  double gens8 = 0.0;
  double gens9 = 0.0;

  double startfl = 0.5;
  double starts3 = 0.0;
  double starts4 = 0.0;
  double starts5 = 0.0;
  double startafb = 0.0;
  double starts7 = 0.0;
  double starts8 = 0.0;
  double starts9 = 0.0;


  morefit::parameter<evalT> Fl("Fl", "F_{\\mathrm{L}}", genfl, 0.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S3("S3", "S_{3}", gens3, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S4("S4", "S_{4}", gens4, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S5("S5", "S_{5}", gens5, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> Afb("Afb", "A_{\\mathrm{FB}}", genafb, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S7("S7", "S_{7}", gens7, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S8("S8", "S_{8}", gens8, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> S9("S9", "S_{9}", gens9, -1.0, 1.0, 0.01, false);
  std::vector<morefit::parameter<evalT>*> params({&Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9});
  
  morefit::Xoshiro128pp rnd;
  rnd.setSeed(int64_t(229387429));

  //produce graphs
  if (false)
    {      
      morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
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
      morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
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
      morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
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
      morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
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

  //check plotting with efficiencies
  if (false)
    {
      morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);      
      morefit::EventVector<kernelT, evalT> eff;
      /*
      unsigned int nctlbins = 100;      
      unsigned int nctkbins = 100;      
      unsigned int nphibins = 100;      
      kstarmumu.set_acceptance_histo(eff, {nctlbins, nctkbins, nphibins});
      
      for (unsigned int i=0; i<nctlbins*nctkbins*nphibins; i++)
	{
	  double ctl_ = 0.5*(eff(i,1) + eff(i,2));
	  double ctk_ = 0.5*(eff(i,3) + eff(i,4));
	  double phi_ = 0.5*(eff(i,5) + eff(i,6));
	    
	  double eff_ = (1.0+0.2*sin(0.5*(ctl_+1.0)*2.0*M_PI*4)*0.5)/1.2
	    *(1.0-0.1*cos(0.5*(ctk_+1.0)*2.0*M_PI*6))/1.1
	    *(1.0+0.3*sin(phi_*10.0))/1.3;
	  
	  eff(i, 0) = eff_;
	}
      */
#ifdef WITH_ROOT
      TFile* bdt_file = new TFile("test_kstarmumu.root", "READ");
      TTree* tree = (TTree*)bdt_file->Get("xgboost_regression");
      unsigned int nnodes = tree->GetEntries();      
      kstarmumu.set_acceptance_bdt(eff, nnodes);
      double value, ctl_from, ctl_to, ctk_from, ctk_to, phi_from, phi_to;
      tree->SetBranchAddress("f0_from", &ctl_from);
      tree->SetBranchAddress("f0_to", &ctl_to);
      tree->SetBranchAddress("f1_from", &ctk_from);
      tree->SetBranchAddress("f1_to", &ctk_to);
      tree->SetBranchAddress("f2_from", &phi_from);
      tree->SetBranchAddress("f2_to", &phi_to);
      tree->SetBranchAddress("value", &value);
      for (unsigned int i=0; i<tree->GetEntries(); i++)
	{
	  tree->GetEntry(i);
	  eff(i,0) = value;
	  eff(i,1) = ctl_from;
	  eff(i,2) = ctl_to;
	  eff(i,3) = ctk_from;
	  eff(i,4) = ctk_to;
	  eff(i,5) = phi_from;
	  eff(i,6) = phi_to;
	}
      eff.print();
#endif      
      /*      
      morefit::EventVector<kernelT, evalT> montecarlo;
      kstarmumu.prepare_monte_carlo(montecarlo, 1000000);
      //montecarlo.print();
      */
      
      unsigned int ngen = 1000000;
      std::cout <<"generating" << std::endl;
      morefit::generator_options gen_opts;
      
      morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
      morefit::EventVector<kernelT, evalT> result({&ctl, &ctk, &phi}, ngen);  
      gen.generate(ngen, &kstarmumu, params, result);      
      
      std::cout <<"fitting" << std::endl;      
      morefit::fitter_options opts;
      opts.minuit_printlevel = 2;
      //opts.analytic_gradient = true;
      //opts.analytic_hessian = true;
      opts.analytic_gradient = false;
      opts.analytic_hessian = false;

      opts.parallelize_loops = true;
      //opts.parallelize_loops = false;
      opts.optimize_dimensions = true;
      
      opts.print();
      morefit::fitter<kernelT, evalT, backendT, blockT > fit(&opts, &backend);
      fit.fit(&kstarmumu, params, &result);
      
      
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
      
      /*
      std::vector<std::string> param_names;
      for (unsigned int i=0; i<params.size(); i++)
	param_names.push_back(params.at(i)->get_name());
      std::vector<double> param_values;
      for (unsigned int i=0; i<params.size(); i++)
	param_values.push_back(params.at(i)->get_value());
      */
      return 0;
    }

  //check different acceptance approaches
  if (false)
    {
      morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
      morefit::EventVector<kernelT, evalT> eff;
#ifdef WITH_ROOT
      TFile* bdt_file = new TFile("test_kstarmumu.root", "READ");
      TTree* tree = (TTree*)bdt_file->Get("xgboost_regression");
      unsigned int nnodes = tree->GetEntries();      
      kstarmumu.set_acceptance_bdt(eff, nnodes);
      double value, ctl_from, ctl_to, ctk_from, ctk_to, phi_from, phi_to;
      tree->SetBranchAddress("f0_from", &ctl_from);
      tree->SetBranchAddress("f0_to", &ctl_to);
      tree->SetBranchAddress("f1_from", &ctk_from);
      tree->SetBranchAddress("f1_to", &ctk_to);
      tree->SetBranchAddress("f2_from", &phi_from);
      tree->SetBranchAddress("f2_to", &phi_to);
      tree->SetBranchAddress("value", &value);
      for (unsigned int i=0; i<tree->GetEntries(); i++)
	{
	  tree->GetEntry(i);
	  eff(i,0) = value;
	  eff(i,1) = ctl_from;
	  eff(i,2) = ctl_to;
	  eff(i,3) = ctk_from;
	  eff(i,4) = ctk_to;
	  eff(i,5) = phi_from;
	  eff(i,6) = phi_to;
	}
      //eff.print();
#endif      

      //ONNX test
#ifdef WITH_ONNX
      
      Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx test");
      Ort::SessionOptions session_options;
      //Ort::Session session = Ort::Session(env, "torch_model_3D_efficiency_mse_grad_1.onnx", session_options);//normalisation
      //Ort::Session session = Ort::Session(env, "torch_model_3D_direct_grad_0.onnx", session_options);//normalisation      
      Ort::Session session = Ort::Session(env, "torch_model_3D_efficiency_mse_grad_0.onnx", session_options);//normalisation      
  
      using AllocatedStringPtr = std::unique_ptr<char, Ort::detail::AllocatedFree>;
      Ort::AllocatorWithDefaultOptions allocator;
      std::vector<AllocatedStringPtr> inputNodeNameAllocatedStrings;
      std::vector<AllocatedStringPtr> outputNodeNameAllocatedStrings;
      std::vector<const char*> input_names;
      std::vector<const char*> output_names;
      
      for (unsigned int i=0; i<session.GetInputCount(); i++)
	{
	  auto input_name_allocated = session.GetInputNameAllocated(i, allocator);
	  inputNodeNameAllocatedStrings.push_back(std::move(input_name_allocated));
	  input_names.emplace_back(inputNodeNameAllocatedStrings.back().get());
	  std::cout << "Input " << i << ": " << input_names.back() << std::endl;

	  auto input_type_info = session.GetInputTypeInfo(i);
	  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	  auto input_shape_ = input_tensor_info.GetShape();
	  for (unsigned int j=0; j <input_shape_.size(); j++)
	    std::cout << "Input " << i << " shape " << input_shape_.at(j) << std::endl; 

	}
      for (unsigned int i=0; i<session.GetOutputCount(); i++)
	{
	  auto output_name_allocated = session.GetOutputNameAllocated(i, allocator);
	  outputNodeNameAllocatedStrings.push_back(std::move(output_name_allocated));
	  output_names.emplace_back(outputNodeNameAllocatedStrings.back().get());
	  std::cout << "Output " << i << ": " << output_names.back() << std::endl;
	}
      
      std::cout << "InputCount " << session.GetInputCount() << " OutputCount " << session.GetOutputCount() << std::endl;
      assert(session.GetInputCount() == 1 && session.GetOutputCount() == 1);
      
      std::vector<int64_t> input_shape = {8};
      unsigned int npoints = 10000;
      std::vector<double> diffs;
      for (unsigned int i=0; i<npoints; i++)
	{
	  double delta = 0.25;
	  double afl = 0.5 + 2.0*delta*(rnd.random()-0.5);
	  double as3 = 2.0*delta*(rnd.random()-0.5);
	  double as4 = 2.0*delta*(rnd.random()-0.5);
	  double as5 = 2.0*delta*(rnd.random()-0.5);
	  double afb = 2.0*delta*(rnd.random()-0.5);
	  double as7 = 2.0*delta*(rnd.random()-0.5);
	  double as8 = 2.0*delta*(rnd.random()-0.5);
	  double as9 = 2.0*delta*(rnd.random()-0.5);
	  
	  std::vector<double> input_values = {afl, as3, as4, as5, afb, as7, as8, as9};
	  
	  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	  Ort::Value input_tensor = Ort::Value::CreateTensor<double>(memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());
	  std::vector<Ort::Value> input_tensors;
	  input_tensors.emplace_back(std::move(input_tensor));
	  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), 1, output_names.data(), 1);
	  const double* result = output_tensors[0].GetTensorMutableData<double>();

	  double j1s = 3.0/4.0*(1.0-afl);    
	  double j6s = 4.0/3.0*afb;
	  double j9 = as9;
	  double nominal = (63.0*j9)/625.0+(63.0*j6s)/625.0+(98.0*j1s)/1875.0+539.0/1250.0;
	  std::cout << result[0] << " " << nominal << " " << result[0]-nominal << std::endl;
	  diffs.push_back(result[0]-nominal);
	}
#ifdef WITH_ROOT
      TH1D* hdiffs = new TH1D("hdiff", ";modeled-true;", 100, -0.1, +0.1);
      for (unsigned int i=0; i<diffs.size(); i++)
	hdiffs->Fill(diffs.at(i));
      TCanvas* c0 = new TCanvas("c0", "c0", 1600, 1200);
      c0->cd();
      hdiffs->SetLineWidth(2.0);
      hdiffs->Draw("hist");
      c0->Print("diffs.eps", "eps");
#endif
#endif
      //#ifdef WITH_ONNX
#if FALSE
      Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx test");
      Ort::SessionOptions session_options;
      Ort::Session session = Ort::Session(env, "mlp_direct_accvsrej_mse_kstarmumu_19.onnx", session_options);//efficiency model

      using AllocatedStringPtr = std::unique_ptr<char, Ort::detail::AllocatedFree>;
      Ort::AllocatorWithDefaultOptions allocator;
      std::vector<AllocatedStringPtr> inputNodeNameAllocatedStrings;
      std::vector<AllocatedStringPtr> outputNodeNameAllocatedStrings;
      std::vector<const char*> input_names;
      std::vector<const char*> output_names;
      
      for (unsigned int i=0; i<session.GetInputCount(); i++)
	{
	  auto input_name_allocated = session.GetInputNameAllocated(i, allocator);
	  inputNodeNameAllocatedStrings.push_back(std::move(input_name_allocated));
	  input_names.emplace_back(inputNodeNameAllocatedStrings.back().get());
	  std::cout << "Input " << i << ": " << input_names.back() << std::endl;

	  auto input_type_info = session.GetInputTypeInfo(i);
	  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	  auto input_shape_ = input_tensor_info.GetShape();
	  for (unsigned int j=0; j <input_shape_.size(); j++)
	    std::cout << "Input " << i << " shape " << input_shape_.at(j) << std::endl; 

	}
      for (unsigned int i=0; i<session.GetOutputCount(); i++)
	{
	  auto output_name_allocated = session.GetOutputNameAllocated(i, allocator);
	  outputNodeNameAllocatedStrings.push_back(std::move(output_name_allocated));
	  output_names.emplace_back(outputNodeNameAllocatedStrings.back().get());
	  std::cout << "Output " << i << ": " << output_names.back() << std::endl;
	}
      
      std::cout << "InputCount " << session.GetInputCount() << " OutputCount " << session.GetOutputCount() << std::endl;
      assert(session.GetInputCount() == 1 && session.GetOutputCount() == 1);
      
      std::vector<int64_t> input_shape = {3};
      unsigned int npoints = 10000;
      std::vector<double> diffs;
      for (unsigned int i=0; i<npoints; i++)
	{
	  double actl = 2.0*rnd.random()-1.0;
	  double actk = 2.0*rnd.random()-1.0;
	  double aphi = (2.0*rnd.random()-1.0)*M_PI;
	  std::vector<double> input_values = {actl, actk, aphi};
	  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	  Ort::Value input_tensor = Ort::Value::CreateTensor<double>(memory_info, input_values.data(), input_values.size(), input_shape.data(), input_shape.size());
	  std::vector<Ort::Value> input_tensors;
	  input_tensors.emplace_back(std::move(input_tensor));
	  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), 1, output_names.data(), 1);
	  const double* result = output_tensors[0].GetTensorMutableData<double>();
	  double nominal = (0.7+0.3*actl)*(1.0-0.2*actk*actk)*(0.7+0.3*sin(2*aphi));
	  std::cout << result[0] << " " << nominal << " " << result[0]-nominal << std::endl;
	  diffs.push_back(result[0]-nominal);
	}
#ifdef WITH_ROOT
      TH1D* hdiffs = new TH1D("hdiff", ";modeled-true;", 100, -0.1, +0.1);
      for (unsigned int i=0; i<diffs.size(); i++)
	hdiffs->Fill(diffs.at(i));
      TCanvas* c0 = new TCanvas("c0", "c0", 1600, 1200);
      c0->cd();
      hdiffs->SetLineWidth(2.0);
      hdiffs->Draw("hist");
      c0->Print("diffs.eps", "eps");
#endif
#endif
      std::cout << "EXIT" << std::endl;
      return 0;
      unsigned int ngen = 1000000;
      std::cout <<"generating" << std::endl;
      morefit::generator_options gen_opts;
      
      morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
      morefit::EventVector<kernelT, evalT> result({&ctl, &ctk, &phi}, ngen);  
      gen.generate(ngen, &kstarmumu, params, result);      
      
      std::cout <<"fitting" << std::endl;      
      morefit::fitter_options opts;
      opts.minuit_printlevel = 2;
      //opts.analytic_gradient = true;
      //opts.analytic_hessian = true;
      opts.analytic_gradient = false;
      opts.analytic_hessian = false;

      opts.parallelize_loops = true;
      //opts.parallelize_loops = false;
      opts.optimize_dimensions = true;
      
      opts.print();
      morefit::fitter<kernelT, evalT, backendT, blockT > fit(&opts, &backend);
      fit.fit(&kstarmumu, params, &result);
      
      
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
      
      return 0;
    }

    //check different models
  if (true)
    {
#ifdef WITH_ONNX

      unsigned int nmodels = 100;
      unsigned int ngen = 100000;//00;

      //toy study
      unsigned int nruns = 1;
      std::vector<double> fl_values_analytic(nmodels*nruns, 0.0);      
      std::vector<double> s3_values_analytic(nmodels*nruns, 0.0);
      std::vector<double> s4_values_analytic(nmodels*nruns, 0.0);
      std::vector<double> s5_values_analytic(nmodels*nruns, 0.0);
      std::vector<double> afb_values_analytic(nmodels*nruns, 0.0);
      std::vector<double> s7_values_analytic(nmodels*nruns, 0.0);
      std::vector<double> s8_values_analytic(nmodels*nruns, 0.0);
      std::vector<double> s9_values_analytic(nmodels*nruns, 0.0);

      std::vector<double> fl_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_groundtruth(nmodels*nruns, 0.0);
      
      std::vector<double> fl_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);

      std::vector<double> fl_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_direct(nmodels*nruns, 0.0);
      
      std::vector<double> fl_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_direct_grad(nmodels*nruns, 0.0);

      std::vector<double> fl_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_mse(nmodels*nruns, 0.0);
      
      std::vector<double> fl_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_mse_grad(nmodels*nruns, 0.0);

      std::vector<double> fl_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_msemodified(nmodels*nruns, 0.0);
      
      std::vector<double> fl_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_msemodified_grad(nmodels*nruns, 0.0);

      std::vector<double> fl_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_bdt(nmodels*nruns, 0.0);
      
      std::vector<double> fl_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_bdt_grad(nmodels*nruns, 0.0);

      std::vector<double> fl_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      
      std::vector<double> fl_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s3_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s4_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s5_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> afb_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s7_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s8_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> s9_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);


      std::vector<double> fl_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> s3_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> s4_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> s5_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> afb_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> s7_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> s8_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> s9_values_bdt(nmodels*nruns, 0.0);

      for (unsigned int m=0; m<nmodels; m++)
	{
	  std::cout << "model " << m << std::endl;
	  morefit::KstarmumuAngularPDFAnalyticEps<kernelT, evalT> kstarmumu_analytic(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
	  
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_groundtruth(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_groundtruth_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_groundtruth_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_groundtruth_grad_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_direct(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_direct_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_direct_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_direct_grad_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_mse(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_mse_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_mse_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_mse_grad_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_msemodified(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_msemodified_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_msemodified_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_msemodified_grad_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_bdt(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_bdt_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_bdt_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_bdt_grad_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_efficiencytruth(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_truth_"+std::to_string(m)+".onnx").c_str());
	  morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_efficiencytruth_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_truth_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  //morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_bce_"+std::to_string(m)+".onnx").c_str());
	  
	  //morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_direct_grad_"+std::to_string(m)+".onnx").c_str());
	  //morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_groundtruth_grad_"+std::to_string(m)+".onnx").c_str());
	  //morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_mse_grad_"+std::to_string(m)+".onnx").c_str());
	  //morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_bdt_grad_"+std::to_string(m)+".onnx").c_str());
	  //morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_truth_grad_"+std::to_string(m)+".onnx").c_str());
	  //morefit::KstarmumuAngularPDFOnnxEps<kernelT, evalT> kstarmumu_onnx_grad(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9, ("weights/torch_model_3D_efficiency_bce_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu_bdt(&ctl, &ctk, &phi, &Fl, &S3, &S4, &S5, &Afb, &S7, &S8, &S9);
	  morefit::EventVector<kernelT, evalT> eff;
	  std::cout << "after pdfs" << std::endl;
#ifdef WITH_ROOT
	  TFile* bdt_file = new TFile(("weights/bdt_direct_accvsrej_mse_kstarmumu_"+std::to_string(m)+".root").c_str(), "READ");
	  TTree* tree = (TTree*)bdt_file->Get("xgboost_regression");
	  unsigned int nnodes = tree->GetEntries();      
	  kstarmumu_bdt.set_acceptance_bdt(eff, nnodes);
	  double value, ctl_from, ctl_to, ctk_from, ctk_to, phi_from, phi_to;
	  tree->SetBranchAddress("f0_from", &ctl_from);
	  tree->SetBranchAddress("f0_to", &ctl_to);
	  tree->SetBranchAddress("f1_from", &ctk_from);
	  tree->SetBranchAddress("f1_to", &ctk_to);
	  tree->SetBranchAddress("f2_from", &phi_from);
	  tree->SetBranchAddress("f2_to", &phi_to);
	  tree->SetBranchAddress("value", &value);
	  for (unsigned int i=0; i<tree->GetEntries(); i++)
	    {
	      tree->GetEntry(i);
	      eff(i,0) = value;
	      eff(i,1) = ctl_from;
	      eff(i,2) = ctl_to;
	      eff(i,3) = ctk_from;
	      eff(i,4) = ctk_to;
	      eff(i,5) = phi_from;
	      eff(i,6) = phi_to;
	    }
	  //eff.print();
#endif      
      
	  std::cout <<"generating" << std::endl;
	  morefit::generator_options gen_opts;
      
	  //morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
	  //morefit::EventVector<kernelT, evalT> result({&x}, ngen);  

	  morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
	  morefit::EventVector<kernelT, evalT> result({&ctl, &ctk, &phi}, ngen);  
	  //gen.generate(ngen, &kstarmumu, params, result);      

	  //gen.generate(ngen, &kstarmumu_analytic, params, result);      
	  //gen.generate(ngen, &kstarmumu, params, result);      
      
	  std::cout <<"fitting" << std::endl;      
	  morefit::fitter_options opts;
	  opts.minuit_printlevel = -1;
	  opts.analytic_gradient = false;
	  opts.analytic_hessian = false;
	  opts.print_level = -1;
	  opts.minuit_printlevel = 2;
	  opts.print_level = 2;

	  opts.optimize_dimensions = true;
	  opts.optimize_parameters = true;

	  opts.print();
	  morefit::fitter<kernelT, evalT, backendT, blockT > fit(&opts, &backend);
	  //fit.fit(&kstarmumu_analytic, params, &result);//TODO FIXME
	  //fit.fit(&kstarmumu, params, &result);//TODO FIXME

	  gen.generate(ngen, &kstarmumu_analytic, params, result);
	  morefit::plotter_options plot_opts;
	  //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
	  plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
	  plot_opts.print_level = 2;
	  plot_opts.plot_pulls = true;
	  //plot_opts.pull_fraction = 0.25;

	  //morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
	  //plot.plot(&kstarmumu_analytic, params, &result, "x", "plot_x.eps", "eps", 100);
	  //plot.plot(&kstarmumu_bdt, params, &result, "x", "plot_x.eps", "eps", 100);

	  for (unsigned int i=0; i<nruns; i++)
	    {
	      std::cout << "model no " << m << " run no " << i << std::endl;

	      //generate
	      Fl.init("Fl", "F_{\\mathrm{L}}", genfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", gens3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", gens4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", gens5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", genafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", gens7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", gens8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", gens9, -1.0, 1.0, 0.01, false);
	      std::cout << "generating" << std::endl;
	      gen.generate(ngen, &kstarmumu_analytic, params, result);
	  
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_groundtruth" << std::endl;
	      fit.fit(&kstarmumu_onnx_groundtruth, params, &result);	      
	      fl_values_onnx_groundtruth.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_groundtruth.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_groundtruth.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_groundtruth.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_groundtruth.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_groundtruth.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_groundtruth.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_groundtruth.at(m*nruns+i) = S9.get_value();
	  
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_groundtruth grad" << std::endl;
	      fit.fit(&kstarmumu_onnx_groundtruth_grad, params, &result);
	      fl_values_onnx_groundtruth_grad.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_groundtruth_grad.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_groundtruth_grad.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_groundtruth_grad.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_groundtruth_grad.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_groundtruth_grad.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_groundtruth_grad.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_groundtruth_grad.at(m*nruns+i) = S9.get_value();


	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_direct" << std::endl;
	      fit.fit(&kstarmumu_onnx_direct, params, &result);	      
	      fl_values_onnx_direct.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_direct.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_direct.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_direct.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_direct.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_direct.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_direct.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_direct.at(m*nruns+i) = S9.get_value();
	  
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_direct grad" << std::endl;
	      fit.fit(&kstarmumu_onnx_direct_grad, params, &result);
	      fl_values_onnx_direct_grad.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_direct_grad.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_direct_grad.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_direct_grad.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_direct_grad.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_direct_grad.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_direct_grad.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_direct_grad.at(m*nruns+i) = S9.get_value();


	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_mse" << std::endl;
	      fit.fit(&kstarmumu_onnx_mse, params, &result);	      
	      fl_values_onnx_mse.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_mse.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_mse.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_mse.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_mse.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_mse.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_mse.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_mse.at(m*nruns+i) = S9.get_value();
	  
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_mse grad" << std::endl;
	      fit.fit(&kstarmumu_onnx_mse_grad, params, &result);
	      fl_values_onnx_mse_grad.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_mse_grad.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_mse_grad.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_mse_grad.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_mse_grad.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_mse_grad.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_mse_grad.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_mse_grad.at(m*nruns+i) = S9.get_value();


	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_msemodified" << std::endl;
	      fit.fit(&kstarmumu_onnx_msemodified, params, &result);	      
	      fl_values_onnx_msemodified.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_msemodified.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_msemodified.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_msemodified.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_msemodified.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_msemodified.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_msemodified.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_msemodified.at(m*nruns+i) = S9.get_value();
	  
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_msemodified grad" << std::endl;
	      fit.fit(&kstarmumu_onnx_msemodified_grad, params, &result);
	      fl_values_onnx_msemodified_grad.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_msemodified_grad.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_msemodified_grad.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_msemodified_grad.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_msemodified_grad.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_msemodified_grad.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_msemodified_grad.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_msemodified_grad.at(m*nruns+i) = S9.get_value();


	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_bdt" << std::endl;
	      fit.fit(&kstarmumu_onnx_bdt, params, &result);	      
	      fl_values_onnx_bdt.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_bdt.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_bdt.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_bdt.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_bdt.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_bdt.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_bdt.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_bdt.at(m*nruns+i) = S9.get_value();
	  
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_bdt grad" << std::endl;
	      fit.fit(&kstarmumu_onnx_bdt_grad, params, &result);
	      fl_values_onnx_bdt_grad.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_bdt_grad.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_bdt_grad.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_bdt_grad.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_bdt_grad.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_bdt_grad.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_bdt_grad.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_bdt_grad.at(m*nruns+i) = S9.get_value();


	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_efficiencytruth" << std::endl;
	      fit.fit(&kstarmumu_onnx_efficiencytruth, params, &result);	      
	      fl_values_onnx_efficiencytruth.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_efficiencytruth.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_efficiencytruth.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_efficiencytruth.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_efficiencytruth.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_efficiencytruth.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_efficiencytruth.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_efficiencytruth.at(m*nruns+i) = S9.get_value();
	  
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting onnx_efficiencytruth grad" << std::endl;
	      fit.fit(&kstarmumu_onnx_efficiencytruth_grad, params, &result);
	      fl_values_onnx_efficiencytruth_grad.at(m*nruns+i) = Fl.get_value();
	      s3_values_onnx_efficiencytruth_grad.at(m*nruns+i) = S3.get_value();
	      s4_values_onnx_efficiencytruth_grad.at(m*nruns+i) = S4.get_value();
	      s5_values_onnx_efficiencytruth_grad.at(m*nruns+i) = S5.get_value();
	      afb_values_onnx_efficiencytruth_grad.at(m*nruns+i) = Afb.get_value();
	      s7_values_onnx_efficiencytruth_grad.at(m*nruns+i) = S7.get_value();
	      s8_values_onnx_efficiencytruth_grad.at(m*nruns+i) = S8.get_value();
	      s9_values_onnx_efficiencytruth_grad.at(m*nruns+i) = S9.get_value();

	      
	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting analytic" << std::endl;
	      fit.fit(&kstarmumu_analytic, params, &result);

	      fl_values_analytic.at(m*nruns+i) = Fl.get_value();
	      s3_values_analytic.at(m*nruns+i) = S3.get_value();
	      s4_values_analytic.at(m*nruns+i) = S4.get_value();
	      s5_values_analytic.at(m*nruns+i) = S5.get_value();
	      afb_values_analytic.at(m*nruns+i) = Afb.get_value();
	      s7_values_analytic.at(m*nruns+i) = S7.get_value();
	      s8_values_analytic.at(m*nruns+i) = S8.get_value();
	      s9_values_analytic.at(m*nruns+i) = S9.get_value();

	      Fl.init("Fl", "F_{\\mathrm{L}}", startfl, 0.0, 1.0, 0.01, false);
	      S3.init("S3", "S_{3}", starts3, -1.0, 1.0, 0.01, false);
	      S4.init("S4", "S_{4}", starts4, -1.0, 1.0, 0.01, false);
	      S5.init("S5", "S_{5}", starts5, -1.0, 1.0, 0.01, false);
	      Afb.init("Afb", "A_{\\mathrm{FB}}", startafb, -1.0, 1.0, 0.01, false);
	      S7.init("S7", "S_{7}", starts7, -1.0, 1.0, 0.01, false);
	      S8.init("S8", "S_{8}", starts8, -1.0, 1.0, 0.01, false);
	      S9.init("S9", "S_{9}", starts9, -1.0, 1.0, 0.01, false);	      
	      std::cout << "fitting bdt" << std::endl;
	      fit.fit(&kstarmumu_bdt, params, &result);

	      fl_values_bdt.at(m*nruns+i) = Fl.get_value();
	      s3_values_bdt.at(m*nruns+i) = S3.get_value();
	      s4_values_bdt.at(m*nruns+i) = S4.get_value();
	      s5_values_bdt.at(m*nruns+i) = S5.get_value();
	      afb_values_bdt.at(m*nruns+i) = Afb.get_value();
	      s7_values_bdt.at(m*nruns+i) = S7.get_value();
	      s8_values_bdt.at(m*nruns+i) = S8.get_value();
	      s9_values_bdt.at(m*nruns+i) = S9.get_value();

	    }
	}

      std::vector<std::string> methods = {"groundtruth", "groundtruth grad", "$\\epsilon$ truth", "$\\epsilon$ truth grad", "BDT", "modeled BDT", "modeled BDT grad", "direct", "direct grad", "mse", "mse grad", "msemodified", "msemodified grad"};
      std::vector<std::vector<double>> analytic_values = {fl_values_analytic, s3_values_analytic, s4_values_analytic, s5_values_analytic, afb_values_analytic, s7_values_analytic, s8_values_analytic, s9_values_analytic};
      std::vector<std::string> observables = {"$F_{\\mathrm{L}}$", "$S_{3}$", "$S_{4}$", "$S_{5}$", "$A_{\\mathrm{FB}}$", "$S_{7}$", "$S_{8}$", "$S_{9}$"};
      std::vector<std::vector<double>> values_onnx_groundtruth = {fl_values_onnx_groundtruth, s3_values_onnx_groundtruth, s4_values_onnx_groundtruth, s5_values_onnx_groundtruth, afb_values_onnx_groundtruth, s7_values_onnx_groundtruth, s8_values_onnx_groundtruth, s9_values_onnx_groundtruth};
      std::vector<std::vector<double>> values_onnx_groundtruth_grad = {fl_values_onnx_groundtruth_grad, s3_values_onnx_groundtruth_grad, s4_values_onnx_groundtruth_grad, s5_values_onnx_groundtruth_grad, afb_values_onnx_groundtruth_grad, s7_values_onnx_groundtruth_grad, s8_values_onnx_groundtruth_grad, s9_values_onnx_groundtruth_grad};
      std::vector<std::vector<double>> values_onnx_efficiencytruth = {fl_values_onnx_efficiencytruth, s3_values_onnx_efficiencytruth, s4_values_onnx_efficiencytruth, s5_values_onnx_efficiencytruth, afb_values_onnx_efficiencytruth, s7_values_onnx_efficiencytruth, s8_values_onnx_efficiencytruth, s9_values_onnx_efficiencytruth};
      std::vector<std::vector<double>> values_onnx_efficiencytruth_grad = {fl_values_onnx_efficiencytruth_grad, s3_values_onnx_efficiencytruth_grad, s4_values_onnx_efficiencytruth_grad, s5_values_onnx_efficiencytruth_grad, afb_values_onnx_efficiencytruth_grad, s7_values_onnx_efficiencytruth_grad, s8_values_onnx_efficiencytruth_grad, s9_values_onnx_efficiencytruth_grad};
      std::vector<std::vector<double>> values_bdt = {fl_values_bdt, s3_values_bdt, s4_values_bdt, s5_values_bdt, afb_values_bdt, s7_values_bdt, s8_values_bdt, s9_values_bdt};
      std::vector<std::vector<double>> values_onnx_bdt = {fl_values_onnx_bdt, s3_values_onnx_bdt, s4_values_onnx_bdt, s5_values_onnx_bdt, afb_values_onnx_bdt, s7_values_onnx_bdt, s8_values_onnx_bdt, s9_values_onnx_bdt};
      std::vector<std::vector<double>> values_onnx_bdt_grad = {fl_values_onnx_bdt_grad, s3_values_onnx_bdt_grad, s4_values_onnx_bdt_grad, s5_values_onnx_bdt_grad, afb_values_onnx_bdt_grad, s7_values_onnx_bdt_grad, s8_values_onnx_bdt_grad, s9_values_onnx_bdt_grad};
      std::vector<std::vector<double>> values_onnx_direct = {fl_values_onnx_direct, s3_values_onnx_direct, s4_values_onnx_direct, s5_values_onnx_direct, afb_values_onnx_direct, s7_values_onnx_direct, s8_values_onnx_direct, s9_values_onnx_direct};
      std::vector<std::vector<double>> values_onnx_direct_grad = {fl_values_onnx_direct_grad, s3_values_onnx_direct_grad, s4_values_onnx_direct_grad, s5_values_onnx_direct_grad, afb_values_onnx_direct_grad, s7_values_onnx_direct_grad, s8_values_onnx_direct_grad, s9_values_onnx_direct_grad};
      std::vector<std::vector<double>> values_onnx_mse = {fl_values_onnx_mse, s3_values_onnx_mse, s4_values_onnx_mse, s5_values_onnx_mse, afb_values_onnx_mse, s7_values_onnx_mse, s8_values_onnx_mse, s9_values_onnx_mse};
      std::vector<std::vector<double>> values_onnx_mse_grad = {fl_values_onnx_mse_grad, s3_values_onnx_mse_grad, s4_values_onnx_mse_grad, s5_values_onnx_mse_grad, afb_values_onnx_mse_grad, s7_values_onnx_mse_grad, s8_values_onnx_mse_grad, s9_values_onnx_mse_grad};
      std::vector<std::vector<double>> values_onnx_msemodified = {fl_values_onnx_msemodified, s3_values_onnx_msemodified, s4_values_onnx_msemodified, s5_values_onnx_msemodified, afb_values_onnx_msemodified, s7_values_onnx_msemodified, s8_values_onnx_msemodified, s9_values_onnx_msemodified};
      std::vector<std::vector<double>> values_onnx_msemodified_grad = {fl_values_onnx_msemodified_grad, s3_values_onnx_msemodified_grad, s4_values_onnx_msemodified_grad, s5_values_onnx_msemodified_grad, afb_values_onnx_msemodified_grad, s7_values_onnx_msemodified_grad, s8_values_onnx_msemodified_grad, s9_values_onnx_msemodified_grad};
      std::vector<std::vector<std::vector<double>>> values = {
	values_onnx_groundtruth, values_onnx_groundtruth_grad,
	values_onnx_efficiencytruth, values_onnx_efficiencytruth_grad,
	values_bdt,
	values_onnx_bdt, values_onnx_bdt_grad,
	values_onnx_direct, values_onnx_direct_grad,
	values_onnx_mse, values_onnx_mse_grad,
	values_onnx_msemodified, values_onnx_msemodified_grad
      };
      print_table_mean(analytic_values, values, observables, methods);
      print_table_rms(analytic_values, values, observables, methods);
      print_table_meanrms(analytic_values, values, observables, methods);
      
#ifdef WITH_ROOT
      gROOT->SetStyle("Plain");
      gStyle->SetOptFit(0);
      gStyle->SetOptStat(0);
      gStyle->SetTextFont(132);
      gStyle->SetTextSize(0.06);
      gStyle->SetTitleFont(132,"xyz");
      gStyle->SetLabelFont(132,"xyz");
      gStyle->SetLabelSize(0.05,"xyz");
      gStyle->SetTitleSize(0.06,"xyz");
      gStyle->SetLegendFont(132);
      
      double dx = 0.1;
      double dxdiff = 0.05;
      unsigned int nbins = 100;

      TH1D* hflanalytic = new TH1D("hflanalytic", ";F_{L};#entries", nbins, genfl-dx, genfl+dx);
      TH1D* hs3analytic = new TH1D("hs3analytic", ";S_{3};#entries", nbins, gens3-dx, gens3+dx);
      TH1D* hs4analytic = new TH1D("hs4analytic", ";S_{4};#entries", nbins, gens4-dx, gens4+dx);
      TH1D* hs5analytic = new TH1D("hs5analytic", ";S_{5};#entries", nbins, gens5-dx, gens5+dx);
      TH1D* hafbanalytic = new TH1D("hafbanalytic", ";A_{FB};#entries", nbins, genafb-dx, genafb+dx);
      TH1D* hs7analytic = new TH1D("hs7analytic", ";S_{7};#entries", nbins, gens7-dx, gens7+dx);
      TH1D* hs8analytic = new TH1D("hs8analytic", ";S_{8};#entries", nbins, gens8-dx, gens8+dx);
      TH1D* hs9analytic = new TH1D("hs9analytic", ";S_{9};#entries", nbins, gens9-dx, gens9+dx);
      
      TH1D* hfldiff_onnx_groundtruth = new TH1D("hfldiff_onnx_groundtruth", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_groundtruth = new TH1D("hs3diff_onnx_groundtruth", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_groundtruth = new TH1D("hs4diff_onnx_groundtruth", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_groundtruth = new TH1D("hs5diff_onnx_groundtruth", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_groundtruth = new TH1D("hafbdiff_onnx_groundtruth", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_groundtruth = new TH1D("hs7diff_onnx_groundtruth", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_groundtruth = new TH1D("hs8diff_onnx_groundtruth", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_groundtruth = new TH1D("hs9diff_onnx_groundtruth", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_groundtruth_grad = new TH1D("hfldiff_onnx_groundtruth_grad", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_groundtruth_grad = new TH1D("hs3diff_onnx_groundtruth_grad", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_groundtruth_grad = new TH1D("hs4diff_onnx_groundtruth_grad", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_groundtruth_grad = new TH1D("hs5diff_onnx_groundtruth_grad", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_groundtruth_grad = new TH1D("hafbdiff_onnx_groundtruth_grad", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_groundtruth_grad = new TH1D("hs7diff_onnx_groundtruth_grad", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_groundtruth_grad = new TH1D("hs8diff_onnx_groundtruth_grad", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_groundtruth_grad = new TH1D("hs9diff_onnx_groundtruth_grad", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_direct = new TH1D("hfldiff_onnx_direct", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_direct = new TH1D("hs3diff_onnx_direct", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_direct = new TH1D("hs4diff_onnx_direct", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_direct = new TH1D("hs5diff_onnx_direct", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_direct = new TH1D("hafbdiff_onnx_direct", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_direct = new TH1D("hs7diff_onnx_direct", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_direct = new TH1D("hs8diff_onnx_direct", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_direct = new TH1D("hs9diff_onnx_direct", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_direct_grad = new TH1D("hfldiff_onnx_direct_grad", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_direct_grad = new TH1D("hs3diff_onnx_direct_grad", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_direct_grad = new TH1D("hs4diff_onnx_direct_grad", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_direct_grad = new TH1D("hs5diff_onnx_direct_grad", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_direct_grad = new TH1D("hafbdiff_onnx_direct_grad", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_direct_grad = new TH1D("hs7diff_onnx_direct_grad", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_direct_grad = new TH1D("hs8diff_onnx_direct_grad", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_direct_grad = new TH1D("hs9diff_onnx_direct_grad", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_mse = new TH1D("hfldiff_onnx_mse", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_mse = new TH1D("hs3diff_onnx_mse", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_mse = new TH1D("hs4diff_onnx_mse", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_mse = new TH1D("hs5diff_onnx_mse", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_mse = new TH1D("hafbdiff_onnx_mse", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_mse = new TH1D("hs7diff_onnx_mse", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_mse = new TH1D("hs8diff_onnx_mse", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_mse = new TH1D("hs9diff_onnx_mse", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_mse_grad = new TH1D("hfldiff_onnx_mse_grad", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_mse_grad = new TH1D("hs3diff_onnx_mse_grad", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_mse_grad = new TH1D("hs4diff_onnx_mse_grad", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_mse_grad = new TH1D("hs5diff_onnx_mse_grad", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_mse_grad = new TH1D("hafbdiff_onnx_mse_grad", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_mse_grad = new TH1D("hs7diff_onnx_mse_grad", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_mse_grad = new TH1D("hs8diff_onnx_mse_grad", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_mse_grad = new TH1D("hs9diff_onnx_mse_grad", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_msemodified = new TH1D("hfldiff_onnx_msemodified", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_msemodified = new TH1D("hs3diff_onnx_msemodified", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_msemodified = new TH1D("hs4diff_onnx_msemodified", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_msemodified = new TH1D("hs5diff_onnx_msemodified", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_msemodified = new TH1D("hafbdiff_onnx_msemodified", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_msemodified = new TH1D("hs7diff_onnx_msemodified", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_msemodified = new TH1D("hs8diff_onnx_msemodified", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_msemodified = new TH1D("hs9diff_onnx_msemodified", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_msemodified_grad = new TH1D("hfldiff_onnx_msemodified_grad", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_msemodified_grad = new TH1D("hs3diff_onnx_msemodified_grad", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_msemodified_grad = new TH1D("hs4diff_onnx_msemodified_grad", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_msemodified_grad = new TH1D("hs5diff_onnx_msemodified_grad", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_msemodified_grad = new TH1D("hafbdiff_onnx_msemodified_grad", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_msemodified_grad = new TH1D("hs7diff_onnx_msemodified_grad", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_msemodified_grad = new TH1D("hs8diff_onnx_msemodified_grad", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_msemodified_grad = new TH1D("hs9diff_onnx_msemodified_grad", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_bdt = new TH1D("hfldiff_onnx_bdt", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_bdt = new TH1D("hs3diff_onnx_bdt", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_bdt = new TH1D("hs4diff_onnx_bdt", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_bdt = new TH1D("hs5diff_onnx_bdt", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_bdt = new TH1D("hafbdiff_onnx_bdt", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_bdt = new TH1D("hs7diff_onnx_bdt", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_bdt = new TH1D("hs8diff_onnx_bdt", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_bdt = new TH1D("hs9diff_onnx_bdt", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_bdt_grad = new TH1D("hfldiff_onnx_bdt_grad", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_bdt_grad = new TH1D("hs3diff_onnx_bdt_grad", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_bdt_grad = new TH1D("hs4diff_onnx_bdt_grad", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_bdt_grad = new TH1D("hs5diff_onnx_bdt_grad", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_bdt_grad = new TH1D("hafbdiff_onnx_bdt_grad", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_bdt_grad = new TH1D("hs7diff_onnx_bdt_grad", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_bdt_grad = new TH1D("hs8diff_onnx_bdt_grad", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_bdt_grad = new TH1D("hs9diff_onnx_bdt_grad", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_efficiencytruth = new TH1D("hfldiff_onnx_efficiencytruth", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_efficiencytruth = new TH1D("hs3diff_onnx_efficiencytruth", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_efficiencytruth = new TH1D("hs4diff_onnx_efficiencytruth", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_efficiencytruth = new TH1D("hs5diff_onnx_efficiencytruth", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_efficiencytruth = new TH1D("hafbdiff_onnx_efficiencytruth", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_efficiencytruth = new TH1D("hs7diff_onnx_efficiencytruth", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_efficiencytruth = new TH1D("hs8diff_onnx_efficiencytruth", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_efficiencytruth = new TH1D("hs9diff_onnx_efficiencytruth", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);

      TH1D* hfldiff_onnx_efficiencytruth_grad = new TH1D("hfldiff_onnx_efficiencytruth_grad", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_onnx_efficiencytruth_grad = new TH1D("hs3diff_onnx_efficiencytruth_grad", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_onnx_efficiencytruth_grad = new TH1D("hs4diff_onnx_efficiencytruth_grad", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_onnx_efficiencytruth_grad = new TH1D("hs5diff_onnx_efficiencytruth_grad", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_onnx_efficiencytruth_grad = new TH1D("hafbdiff_onnx_efficiencytruth_grad", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_onnx_efficiencytruth_grad = new TH1D("hs7diff_onnx_efficiencytruth_grad", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_onnx_efficiencytruth_grad = new TH1D("hs8diff_onnx_efficiencytruth_grad", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_onnx_efficiencytruth_grad = new TH1D("hs9diff_onnx_efficiencytruth_grad", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);


      TH1D* hfldiff_bdt = new TH1D("hfldiff_bdt", ";F_{L} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs3diff_bdt = new TH1D("hs3diff_bdt", ";S_{3} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs4diff_bdt = new TH1D("hs4diff_bdt", ";S_{4} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs5diff_bdt = new TH1D("hs5diff_bdt", ";S_{5} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hafbdiff_bdt = new TH1D("hafbdiff_bdt", ";A_{FB} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs7diff_bdt = new TH1D("hs7diff_bdt", ";S_{7} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs8diff_bdt = new TH1D("hs8diff_bdt", ";S_{8} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hs9diff_bdt = new TH1D("hs9diff_bdt", ";S_{9} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);


      for (unsigned int i=0; i<nruns*nmodels; i++)
	{
	  hflanalytic->Fill(fl_values_analytic.at(i));
	  hs3analytic->Fill(s3_values_analytic.at(i));
	  hs4analytic->Fill(s4_values_analytic.at(i));
	  hs5analytic->Fill(s5_values_analytic.at(i));
	  hafbanalytic->Fill(afb_values_analytic.at(i));
	  hs7analytic->Fill(s7_values_analytic.at(i));
	  hs8analytic->Fill(s8_values_analytic.at(i));
	  hs9analytic->Fill(s9_values_analytic.at(i));

	  hfldiff_onnx_groundtruth->Fill(fl_values_onnx_groundtruth.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_groundtruth->Fill(s3_values_onnx_groundtruth.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_groundtruth->Fill(s4_values_onnx_groundtruth.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_groundtruth->Fill(s5_values_onnx_groundtruth.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_groundtruth->Fill(afb_values_onnx_groundtruth.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_groundtruth->Fill(s7_values_onnx_groundtruth.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_groundtruth->Fill(s8_values_onnx_groundtruth.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_groundtruth->Fill(s9_values_onnx_groundtruth.at(i)-s9_values_analytic.at(i));
	  
	  hfldiff_onnx_groundtruth_grad->Fill(fl_values_onnx_groundtruth_grad.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_groundtruth_grad->Fill(s3_values_onnx_groundtruth_grad.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_groundtruth_grad->Fill(s4_values_onnx_groundtruth_grad.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_groundtruth_grad->Fill(s5_values_onnx_groundtruth_grad.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_groundtruth_grad->Fill(afb_values_onnx_groundtruth_grad.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_groundtruth_grad->Fill(s7_values_onnx_groundtruth_grad.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_groundtruth_grad->Fill(s8_values_onnx_groundtruth_grad.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_groundtruth_grad->Fill(s9_values_onnx_groundtruth_grad.at(i)-s9_values_analytic.at(i));	  

	  hfldiff_onnx_direct->Fill(fl_values_onnx_direct.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_direct->Fill(s3_values_onnx_direct.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_direct->Fill(s4_values_onnx_direct.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_direct->Fill(s5_values_onnx_direct.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_direct->Fill(afb_values_onnx_direct.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_direct->Fill(s7_values_onnx_direct.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_direct->Fill(s8_values_onnx_direct.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_direct->Fill(s9_values_onnx_direct.at(i)-s9_values_analytic.at(i));
	  
	  hfldiff_onnx_direct_grad->Fill(fl_values_onnx_direct_grad.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_direct_grad->Fill(s3_values_onnx_direct_grad.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_direct_grad->Fill(s4_values_onnx_direct_grad.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_direct_grad->Fill(s5_values_onnx_direct_grad.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_direct_grad->Fill(afb_values_onnx_direct_grad.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_direct_grad->Fill(s7_values_onnx_direct_grad.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_direct_grad->Fill(s8_values_onnx_direct_grad.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_direct_grad->Fill(s9_values_onnx_direct_grad.at(i)-s9_values_analytic.at(i));	  

	  hfldiff_onnx_mse->Fill(fl_values_onnx_mse.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_mse->Fill(s3_values_onnx_mse.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_mse->Fill(s4_values_onnx_mse.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_mse->Fill(s5_values_onnx_mse.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_mse->Fill(afb_values_onnx_mse.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_mse->Fill(s7_values_onnx_mse.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_mse->Fill(s8_values_onnx_mse.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_mse->Fill(s9_values_onnx_mse.at(i)-s9_values_analytic.at(i));
	  
	  hfldiff_onnx_mse_grad->Fill(fl_values_onnx_mse_grad.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_mse_grad->Fill(s3_values_onnx_mse_grad.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_mse_grad->Fill(s4_values_onnx_mse_grad.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_mse_grad->Fill(s5_values_onnx_mse_grad.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_mse_grad->Fill(afb_values_onnx_mse_grad.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_mse_grad->Fill(s7_values_onnx_mse_grad.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_mse_grad->Fill(s8_values_onnx_mse_grad.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_mse_grad->Fill(s9_values_onnx_mse_grad.at(i)-s9_values_analytic.at(i));	  

	  hfldiff_onnx_msemodified->Fill(fl_values_onnx_msemodified.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_msemodified->Fill(s3_values_onnx_msemodified.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_msemodified->Fill(s4_values_onnx_msemodified.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_msemodified->Fill(s5_values_onnx_msemodified.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_msemodified->Fill(afb_values_onnx_msemodified.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_msemodified->Fill(s7_values_onnx_msemodified.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_msemodified->Fill(s8_values_onnx_msemodified.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_msemodified->Fill(s9_values_onnx_msemodified.at(i)-s9_values_analytic.at(i));
	  
	  hfldiff_onnx_msemodified_grad->Fill(fl_values_onnx_msemodified_grad.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_msemodified_grad->Fill(s3_values_onnx_msemodified_grad.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_msemodified_grad->Fill(s4_values_onnx_msemodified_grad.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_msemodified_grad->Fill(s5_values_onnx_msemodified_grad.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_msemodified_grad->Fill(afb_values_onnx_msemodified_grad.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_msemodified_grad->Fill(s7_values_onnx_msemodified_grad.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_msemodified_grad->Fill(s8_values_onnx_msemodified_grad.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_msemodified_grad->Fill(s9_values_onnx_msemodified_grad.at(i)-s9_values_analytic.at(i));	  

	  hfldiff_onnx_bdt->Fill(fl_values_onnx_bdt.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_bdt->Fill(s3_values_onnx_bdt.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_bdt->Fill(s4_values_onnx_bdt.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_bdt->Fill(s5_values_onnx_bdt.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_bdt->Fill(afb_values_onnx_bdt.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_bdt->Fill(s7_values_onnx_bdt.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_bdt->Fill(s8_values_onnx_bdt.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_bdt->Fill(s9_values_onnx_bdt.at(i)-s9_values_analytic.at(i));
	  
	  hfldiff_onnx_bdt_grad->Fill(fl_values_onnx_bdt_grad.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_bdt_grad->Fill(s3_values_onnx_bdt_grad.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_bdt_grad->Fill(s4_values_onnx_bdt_grad.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_bdt_grad->Fill(s5_values_onnx_bdt_grad.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_bdt_grad->Fill(afb_values_onnx_bdt_grad.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_bdt_grad->Fill(s7_values_onnx_bdt_grad.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_bdt_grad->Fill(s8_values_onnx_bdt_grad.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_bdt_grad->Fill(s9_values_onnx_bdt_grad.at(i)-s9_values_analytic.at(i));	  

	  hfldiff_onnx_efficiencytruth->Fill(fl_values_onnx_efficiencytruth.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_efficiencytruth->Fill(s3_values_onnx_efficiencytruth.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_efficiencytruth->Fill(s4_values_onnx_efficiencytruth.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_efficiencytruth->Fill(s5_values_onnx_efficiencytruth.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_efficiencytruth->Fill(afb_values_onnx_efficiencytruth.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_efficiencytruth->Fill(s7_values_onnx_efficiencytruth.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_efficiencytruth->Fill(s8_values_onnx_efficiencytruth.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_efficiencytruth->Fill(s9_values_onnx_efficiencytruth.at(i)-s9_values_analytic.at(i));
	  
	  hfldiff_onnx_efficiencytruth_grad->Fill(fl_values_onnx_efficiencytruth_grad.at(i)-fl_values_analytic.at(i));
	  hs3diff_onnx_efficiencytruth_grad->Fill(s3_values_onnx_efficiencytruth_grad.at(i)-s3_values_analytic.at(i));
	  hs4diff_onnx_efficiencytruth_grad->Fill(s4_values_onnx_efficiencytruth_grad.at(i)-s4_values_analytic.at(i));
	  hs5diff_onnx_efficiencytruth_grad->Fill(s5_values_onnx_efficiencytruth_grad.at(i)-s5_values_analytic.at(i));
	  hafbdiff_onnx_efficiencytruth_grad->Fill(afb_values_onnx_efficiencytruth_grad.at(i)-afb_values_analytic.at(i));
	  hs7diff_onnx_efficiencytruth_grad->Fill(s7_values_onnx_efficiencytruth_grad.at(i)-s7_values_analytic.at(i));
	  hs8diff_onnx_efficiencytruth_grad->Fill(s8_values_onnx_efficiencytruth_grad.at(i)-s8_values_analytic.at(i));
	  hs9diff_onnx_efficiencytruth_grad->Fill(s9_values_onnx_efficiencytruth_grad.at(i)-s9_values_analytic.at(i));	  
	  
	  hfldiff_bdt->Fill(fl_values_bdt.at(i)-fl_values_analytic.at(i));
	  hs3diff_bdt->Fill(s3_values_bdt.at(i)-s3_values_analytic.at(i));
	  hs4diff_bdt->Fill(s4_values_bdt.at(i)-s4_values_analytic.at(i));
	  hs5diff_bdt->Fill(s5_values_bdt.at(i)-s5_values_analytic.at(i));
	  hafbdiff_bdt->Fill(afb_values_bdt.at(i)-afb_values_analytic.at(i));
	  hs7diff_bdt->Fill(s7_values_bdt.at(i)-s7_values_analytic.at(i));
	  hs8diff_bdt->Fill(s8_values_bdt.at(i)-s8_values_analytic.at(i));
	  hs9diff_bdt->Fill(s9_values_bdt.at(i)-s9_values_analytic.at(i));
	}

      TCanvas* c0_ = new TCanvas("c0", "c0", 3*1200, 3*800);
      c0_->Divide(3,3);
      c0_->cd(1)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hflanalytic->SetLineWidth(1.0);
      hflanalytic->Draw("hist");
      c0_->cd(2)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs3analytic->SetLineWidth(1.0);
      hs3analytic->Draw("hist");
      c0_->cd(3)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs4analytic->SetLineWidth(1.0);
      hs4analytic->Draw("hist");
      c0_->cd(4)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs5analytic->SetLineWidth(1.0);
      hs5analytic->Draw("hist");
      c0_->cd(5)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hafbanalytic->SetLineWidth(1.0);
      hafbanalytic->Draw("hist");
      c0_->cd(6)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs7analytic->SetLineWidth(1.0);
      hs7analytic->Draw("hist");
      c0_->cd(7)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs8analytic->SetLineWidth(1.0);
      hs8analytic->Draw("hist");
      c0_->cd(8)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs9analytic->SetLineWidth(1.0);
      hs9analytic->Draw("hist");
      
      c0_->Print("values_kstarmumu.eps", "eps");
      c0_->Print("values_kstarmumu.root", "root");
      
      TCanvas* c1_ = new TCanvas("c1", "c1", 3*1600, 3*800);
      c1_->Divide(3,3);
      c1_->cd(1)->SetMargin(0.125, 0.05, 0.125, 0.05);

      TLegend* leg = new TLegend(0.6, 0.5, 0.95, 0.95);
      //leg->AddEntry(hc1analytic,"analytic truth","l");
      leg->AddEntry(hfldiff_bdt,"BDT modeling #epsilon","l");
      leg->AddEntry(hfldiff_onnx_groundtruth,"ONNX groundtruth","l");
      leg->AddEntry(hfldiff_onnx_groundtruth_grad,"ONNX groundtruth grad.","l");
      leg->AddEntry(hfldiff_onnx_efficiencytruth,"ONNX efficiencytruth","l");
      leg->AddEntry(hfldiff_onnx_efficiencytruth_grad,"ONNX efficiencytruth grad.","l");
      leg->AddEntry(hfldiff_onnx_direct,"ONNX direct","l");
      leg->AddEntry(hfldiff_onnx_direct_grad,"ONNX direct grad.","l");
      leg->AddEntry(hfldiff_onnx_mse,"ONNX mse","l");
      leg->AddEntry(hfldiff_onnx_mse_grad,"ONNX mse grad.","l");
      leg->AddEntry(hfldiff_onnx_msemodified,"ONNX msemodified","l");
      leg->AddEntry(hfldiff_onnx_msemodified_grad,"ONNX msemodified grad.","l");
      //leg->AddEntry(hfldiff_onnx_bdt,"ONNX bdt","l");
      //leg->AddEntry(hfldiff_onnx_bdt_grad,"ONNX bdt grad.","l");
      
      hfldiff_onnx_groundtruth->SetLineWidth(1.0);
      hfldiff_onnx_groundtruth->SetLineColor(2);
      hfldiff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hfldiff_onnx_groundtruth_grad->SetLineColor(2);
      hfldiff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hfldiff_onnx_direct->SetLineWidth(1.0);
      hfldiff_onnx_direct->SetLineColor(kYellow+2);
      hfldiff_onnx_direct_grad->SetLineWidth(1.0);
      hfldiff_onnx_direct_grad->SetLineColor(kYellow+2);
      hfldiff_onnx_direct_grad->SetLineStyle(kDashed);
      hfldiff_onnx_mse->SetLineWidth(1.0);
      hfldiff_onnx_mse->SetLineColor(4);
      hfldiff_onnx_mse_grad->SetLineWidth(1.0);
      hfldiff_onnx_mse_grad->SetLineColor(4);
      hfldiff_onnx_mse_grad->SetLineStyle(kDashed);
      hfldiff_onnx_msemodified->SetLineWidth(1.0);
      hfldiff_onnx_msemodified->SetLineColor(kMagenta+1);
      hfldiff_onnx_msemodified_grad->SetLineWidth(1.0);
      hfldiff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hfldiff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hfldiff_onnx_bdt->SetLineWidth(1.0);
      hfldiff_onnx_bdt->SetLineColor(kOrange);
      hfldiff_onnx_bdt_grad->SetLineWidth(1.0);
      hfldiff_onnx_bdt_grad->SetLineColor(kOrange);
      hfldiff_onnx_bdt_grad->SetLineStyle(kDashed);
      hfldiff_onnx_efficiencytruth->SetLineWidth(1.0);
      hfldiff_onnx_efficiencytruth->SetLineColor(7);
      hfldiff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hfldiff_onnx_efficiencytruth_grad->SetLineColor(7);
      hfldiff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);  
      hfldiff_bdt->SetLineWidth(1.0);
      hfldiff_bdt->SetLineColor(kGreen+2);
      
      hfldiff_onnx_groundtruth_grad->SetMaximum(hfldiff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hfldiff_onnx_groundtruth_grad->Draw("hist");
      hfldiff_onnx_groundtruth->Draw("histsame");
      hfldiff_onnx_efficiencytruth->Draw("histsame");
      hfldiff_onnx_efficiencytruth_grad->Draw("histsame");
      hfldiff_bdt->Draw("histsame");      
      hfldiff_onnx_direct->Draw("histsame");
      hfldiff_onnx_direct_grad->Draw("histsame");
      hfldiff_onnx_mse->Draw("histsame");
      hfldiff_onnx_mse_grad->Draw("histsame");
      hfldiff_onnx_msemodified->Draw("histsame");
      hfldiff_onnx_msemodified_grad->Draw("histsame");
      //hfldiff_onnx_bdt->Draw("histsame");
      //hfldiff_onnx_bdt_grad->Draw("histsame");
      leg->Draw();
      c1_->cd(1)->Print("diffs_kstarmumu_fl.eps", "eps");
      c1_->cd(1)->Print("diffs_kstarmumu_fl.root", "root");

      c1_->cd(2)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs3diff_onnx_groundtruth->SetLineWidth(1.0);
      hs3diff_onnx_groundtruth->SetLineColor(2);
      hs3diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hs3diff_onnx_groundtruth_grad->SetLineColor(2);
      hs3diff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hs3diff_onnx_direct->SetLineWidth(1.0);
      hs3diff_onnx_direct->SetLineColor(kYellow+2);
      hs3diff_onnx_direct_grad->SetLineWidth(1.0);
      hs3diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hs3diff_onnx_direct_grad->SetLineStyle(kDashed);
      hs3diff_onnx_mse->SetLineWidth(1.0);
      hs3diff_onnx_mse->SetLineColor(4);
      hs3diff_onnx_mse_grad->SetLineWidth(1.0);
      hs3diff_onnx_mse_grad->SetLineColor(4);
      hs3diff_onnx_mse_grad->SetLineStyle(kDashed);
      hs3diff_onnx_msemodified->SetLineWidth(1.0);
      hs3diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hs3diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hs3diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hs3diff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hs3diff_onnx_bdt->SetLineWidth(1.0);
      hs3diff_onnx_bdt->SetLineColor(kOrange);
      hs3diff_onnx_bdt_grad->SetLineWidth(1.0);
      hs3diff_onnx_bdt_grad->SetLineColor(kOrange);
      hs3diff_onnx_bdt_grad->SetLineStyle(kDashed);
      hs3diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hs3diff_onnx_efficiencytruth->SetLineColor(7);
      hs3diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hs3diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hs3diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);
      hs3diff_bdt->SetLineWidth(1.0);
      hs3diff_bdt->SetLineColor(kGreen+2);

      hs3diff_onnx_groundtruth_grad->SetMaximum(hs3diff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hs3diff_onnx_groundtruth_grad->Draw("hist");
      hs3diff_onnx_groundtruth->Draw("histsame");
      hs3diff_onnx_efficiencytruth->Draw("histsame");
      hs3diff_onnx_efficiencytruth_grad->Draw("histsame");
      hs3diff_bdt->Draw("histsame");
      hs3diff_onnx_direct->Draw("histsame");
      hs3diff_onnx_direct_grad->Draw("histsame");
      hs3diff_onnx_mse->Draw("histsame");
      hs3diff_onnx_mse_grad->Draw("histsame");
      hs3diff_onnx_msemodified->Draw("histsame");
      hs3diff_onnx_msemodified_grad->Draw("histsame");
      //hs3diff_onnx_bdt->Draw("histsame");
      //hs3diff_onnx_bdt_grad->Draw("histsame");
      c1_->cd(2)->Print("diffs_kstarmumu_s3.eps", "eps");
      c1_->cd(2)->Print("diffs_kstarmumu_s3.root", "root");
      
      c1_->cd(3)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs4diff_onnx_groundtruth->SetLineWidth(1.0);
      hs4diff_onnx_groundtruth->SetLineColor(2);
      hs4diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hs4diff_onnx_groundtruth_grad->SetLineColor(2);
      hs4diff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hs4diff_onnx_direct->SetLineWidth(1.0);
      hs4diff_onnx_direct->SetLineColor(kYellow+2);
      hs4diff_onnx_direct_grad->SetLineWidth(1.0);
      hs4diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hs4diff_onnx_direct_grad->SetLineStyle(kDashed);
      hs4diff_onnx_mse->SetLineWidth(1.0);
      hs4diff_onnx_mse->SetLineColor(4);
      hs4diff_onnx_mse_grad->SetLineWidth(1.0);
      hs4diff_onnx_mse_grad->SetLineColor(4);
      hs4diff_onnx_mse_grad->SetLineStyle(kDashed);
      hs4diff_onnx_msemodified->SetLineWidth(1.0);
      hs4diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hs4diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hs4diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hs4diff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hs4diff_onnx_bdt->SetLineWidth(1.0);
      hs4diff_onnx_bdt->SetLineColor(kOrange);
      hs4diff_onnx_bdt_grad->SetLineWidth(1.0);
      hs4diff_onnx_bdt_grad->SetLineColor(kOrange);
      hs4diff_onnx_bdt_grad->SetLineStyle(kDashed);
      hs4diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hs4diff_onnx_efficiencytruth->SetLineColor(7);
      hs4diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hs4diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hs4diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);
      hs4diff_bdt->SetLineWidth(1.0);
      hs4diff_bdt->SetLineColor(kGreen+2);

      hs4diff_onnx_groundtruth_grad->SetMaximum(hs4diff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hs4diff_onnx_groundtruth_grad->Draw("hist");
      hs4diff_onnx_groundtruth->Draw("histsame");
      hs4diff_onnx_efficiencytruth->Draw("histsame");
      hs4diff_onnx_efficiencytruth_grad->Draw("histsame");
      hs4diff_bdt->Draw("histsame");
      hs4diff_onnx_direct->Draw("histsame");
      hs4diff_onnx_direct_grad->Draw("histsame");
      hs4diff_onnx_mse->Draw("histsame");
      hs4diff_onnx_mse_grad->Draw("histsame");
      hs4diff_onnx_msemodified->Draw("histsame");
      hs4diff_onnx_msemodified_grad->Draw("histsame");
      //hs4diff_onnx_bdt->Draw("histsame");
      //hs4diff_onnx_bdt_grad->Draw("histsame");
      c1_->cd(3)->Print("diffs_kstarmumu_s4.eps", "eps");
      c1_->cd(3)->Print("diffs_kstarmumu_s4.root", "root");

      c1_->cd(4)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs5diff_onnx_groundtruth->SetLineWidth(1.0);
      hs5diff_onnx_groundtruth->SetLineColor(2);
      hs5diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hs5diff_onnx_groundtruth_grad->SetLineColor(2);
      hs5diff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hs5diff_onnx_direct->SetLineWidth(1.0);
      hs5diff_onnx_direct->SetLineColor(kYellow+2);
      hs5diff_onnx_direct_grad->SetLineWidth(1.0);
      hs5diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hs5diff_onnx_direct_grad->SetLineStyle(kDashed);
      hs5diff_onnx_mse->SetLineWidth(1.0);
      hs5diff_onnx_mse->SetLineColor(4);
      hs5diff_onnx_mse_grad->SetLineWidth(1.0);
      hs5diff_onnx_mse_grad->SetLineColor(4);
      hs5diff_onnx_mse_grad->SetLineStyle(kDashed);
      hs5diff_onnx_msemodified->SetLineWidth(1.0);
      hs5diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hs5diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hs5diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hs5diff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hs5diff_onnx_bdt->SetLineWidth(1.0);
      hs5diff_onnx_bdt->SetLineColor(kOrange);
      hs5diff_onnx_bdt_grad->SetLineWidth(1.0);
      hs5diff_onnx_bdt_grad->SetLineColor(kOrange);
      hs5diff_onnx_bdt_grad->SetLineStyle(kDashed);
      hs5diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hs5diff_onnx_efficiencytruth->SetLineColor(7);
      hs5diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hs5diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hs5diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);
      hs5diff_bdt->SetLineWidth(1.0);
      hs5diff_bdt->SetLineColor(kGreen+2);

      hs5diff_onnx_groundtruth_grad->SetMaximum(hs5diff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hs5diff_onnx_groundtruth_grad->Draw("hist");
      hs5diff_onnx_groundtruth->Draw("histsame");
      hs5diff_onnx_efficiencytruth->Draw("histsame");
      hs5diff_onnx_efficiencytruth_grad->Draw("histsame");
      hs5diff_bdt->Draw("histsame");
      hs5diff_onnx_direct->Draw("histsame");
      hs5diff_onnx_direct_grad->Draw("histsame");
      hs5diff_onnx_mse->Draw("histsame");
      hs5diff_onnx_mse_grad->Draw("histsame");
      hs5diff_onnx_msemodified->Draw("histsame");
      hs5diff_onnx_msemodified_grad->Draw("histsame");
      //hs5diff_onnx_bdt->Draw("histsame");
      //hs5diff_onnx_bdt_grad->Draw("histsame");
      c1_->cd(4)->Print("diffs_kstarmumu_s5.eps", "eps");
      c1_->cd(4)->Print("diffs_kstarmumu_s5.root", "root");
      
      c1_->cd(5)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hafbdiff_onnx_groundtruth->SetLineWidth(1.0);
      hafbdiff_onnx_groundtruth->SetLineColor(2);
      hafbdiff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hafbdiff_onnx_groundtruth_grad->SetLineColor(2);
      hafbdiff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hafbdiff_onnx_direct->SetLineWidth(1.0);
      hafbdiff_onnx_direct->SetLineColor(kYellow+2);
      hafbdiff_onnx_direct_grad->SetLineWidth(1.0);
      hafbdiff_onnx_direct_grad->SetLineColor(kYellow+2);
      hafbdiff_onnx_direct_grad->SetLineStyle(kDashed);
      hafbdiff_onnx_mse->SetLineWidth(1.0);
      hafbdiff_onnx_mse->SetLineColor(4);
      hafbdiff_onnx_mse_grad->SetLineWidth(1.0);
      hafbdiff_onnx_mse_grad->SetLineColor(4);
      hafbdiff_onnx_mse_grad->SetLineStyle(kDashed);
      hafbdiff_onnx_msemodified->SetLineWidth(1.0);
      hafbdiff_onnx_msemodified->SetLineColor(kMagenta+1);
      hafbdiff_onnx_msemodified_grad->SetLineWidth(1.0);
      hafbdiff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hafbdiff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hafbdiff_onnx_bdt->SetLineWidth(1.0);
      hafbdiff_onnx_bdt->SetLineColor(kOrange);
      hafbdiff_onnx_bdt_grad->SetLineWidth(1.0);
      hafbdiff_onnx_bdt_grad->SetLineColor(kOrange);
      hafbdiff_onnx_bdt_grad->SetLineStyle(kDashed);
      hafbdiff_onnx_efficiencytruth->SetLineWidth(1.0);
      hafbdiff_onnx_efficiencytruth->SetLineColor(7);
      hafbdiff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hafbdiff_onnx_efficiencytruth_grad->SetLineColor(7);
      hafbdiff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);
      hafbdiff_bdt->SetLineWidth(1.0);
      hafbdiff_bdt->SetLineColor(kGreen+2);

      hafbdiff_onnx_groundtruth_grad->SetMaximum(hafbdiff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hafbdiff_onnx_groundtruth_grad->Draw("hist");
      hafbdiff_onnx_groundtruth->Draw("histsame");
      hafbdiff_onnx_efficiencytruth->Draw("histsame");
      hafbdiff_onnx_efficiencytruth_grad->Draw("histsame");
      hafbdiff_bdt->Draw("histsame");
      hafbdiff_onnx_direct->Draw("histsame");
      hafbdiff_onnx_direct_grad->Draw("histsame");
      hafbdiff_onnx_mse->Draw("histsame");
      hafbdiff_onnx_mse_grad->Draw("histsame");
      hafbdiff_onnx_msemodified->Draw("histsame");
      hafbdiff_onnx_msemodified_grad->Draw("histsame");
      //hafbdiff_onnx_bdt->Draw("histsame");
      //hafbdiff_onnx_bdt_grad->Draw("histsame");
      leg->Draw();
      c1_->cd(5)->Print("diffs_kstarmumu_afb.eps", "eps");
      c1_->cd(5)->Print("diffs_kstarmumu_afb.root", "root");
      
      c1_->cd(6)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs7diff_onnx_groundtruth->SetLineWidth(1.0);
      hs7diff_onnx_groundtruth->SetLineColor(2);
      hs7diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hs7diff_onnx_groundtruth_grad->SetLineColor(2);
      hs7diff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hs7diff_onnx_direct->SetLineWidth(1.0);
      hs7diff_onnx_direct->SetLineColor(kYellow+2);
      hs7diff_onnx_direct_grad->SetLineWidth(1.0);
      hs7diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hs7diff_onnx_direct_grad->SetLineStyle(kDashed);
      hs7diff_onnx_mse->SetLineWidth(1.0);
      hs7diff_onnx_mse->SetLineColor(4);
      hs7diff_onnx_mse_grad->SetLineWidth(1.0);
      hs7diff_onnx_mse_grad->SetLineColor(4);
      hs7diff_onnx_mse_grad->SetLineStyle(kDashed);
      hs7diff_onnx_msemodified->SetLineWidth(1.0);
      hs7diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hs7diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hs7diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hs7diff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hs7diff_onnx_bdt->SetLineWidth(1.0);
      hs7diff_onnx_bdt->SetLineColor(kOrange);
      hs7diff_onnx_bdt_grad->SetLineWidth(1.0);
      hs7diff_onnx_bdt_grad->SetLineColor(kOrange);
      hs7diff_onnx_bdt_grad->SetLineStyle(kDashed);
      hs7diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hs7diff_onnx_efficiencytruth->SetLineColor(7);
      hs7diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hs7diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hs7diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);
      hs7diff_bdt->SetLineWidth(1.0);
      hs7diff_bdt->SetLineColor(kGreen+2);

      hs7diff_onnx_groundtruth_grad->SetMaximum(hs7diff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hs7diff_onnx_groundtruth_grad->Draw("hist");
      hs7diff_onnx_groundtruth->Draw("histsame");
      hs7diff_onnx_efficiencytruth->Draw("histsame");
      hs7diff_onnx_efficiencytruth_grad->Draw("histsame");
      hs7diff_bdt->Draw("histsame");
      hs7diff_onnx_direct->Draw("histsame");
      hs7diff_onnx_direct_grad->Draw("histsame");
      hs7diff_onnx_mse->Draw("histsame");
      hs7diff_onnx_mse_grad->Draw("histsame");
      hs7diff_onnx_msemodified->Draw("histsame");
      hs7diff_onnx_msemodified_grad->Draw("histsame");
      //hs7diff_onnx_bdt->Draw("histsame");
      //hs7diff_onnx_bdt_grad->Draw("histsame");
      c1_->cd(6)->Print("diffs_kstarmumu_s7.eps", "eps");
      c1_->cd(6)->Print("diffs_kstarmumu_s7.root", "root");
      
      c1_->cd(7)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs8diff_onnx_groundtruth->SetLineWidth(1.0);
      hs8diff_onnx_groundtruth->SetLineColor(2);
      hs8diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hs8diff_onnx_groundtruth_grad->SetLineColor(2);
      hs8diff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hs8diff_onnx_direct->SetLineWidth(1.0);
      hs8diff_onnx_direct->SetLineColor(kYellow+2);
      hs8diff_onnx_direct_grad->SetLineWidth(1.0);
      hs8diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hs8diff_onnx_direct_grad->SetLineStyle(kDashed);
      hs8diff_onnx_mse->SetLineWidth(1.0);
      hs8diff_onnx_mse->SetLineColor(4);
      hs8diff_onnx_mse_grad->SetLineWidth(1.0);
      hs8diff_onnx_mse_grad->SetLineColor(4);
      hs8diff_onnx_mse_grad->SetLineStyle(kDashed);
      hs8diff_onnx_msemodified->SetLineWidth(1.0);
      hs8diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hs8diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hs8diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hs8diff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hs8diff_onnx_bdt->SetLineWidth(1.0);
      hs8diff_onnx_bdt->SetLineColor(kOrange);
      hs8diff_onnx_bdt_grad->SetLineWidth(1.0);
      hs8diff_onnx_bdt_grad->SetLineColor(kOrange);
      hs8diff_onnx_bdt_grad->SetLineStyle(kDashed);
      hs8diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hs8diff_onnx_efficiencytruth->SetLineColor(7);
      hs8diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hs8diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hs8diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);
      hs8diff_bdt->SetLineWidth(1.0);
      hs8diff_bdt->SetLineColor(kGreen+2);

      hs8diff_onnx_groundtruth_grad->SetMaximum(hs8diff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hs8diff_onnx_groundtruth_grad->Draw("hist");
      hs8diff_onnx_groundtruth->Draw("histsame");
      hs8diff_onnx_efficiencytruth->Draw("histsame");
      hs8diff_onnx_efficiencytruth_grad->Draw("histsame");
      hs8diff_bdt->Draw("histsame");
      hs8diff_onnx_direct->Draw("histsame");
      hs8diff_onnx_direct_grad->Draw("histsame");
      hs8diff_onnx_mse->Draw("histsame");
      hs8diff_onnx_mse_grad->Draw("histsame");
      hs8diff_onnx_msemodified->Draw("histsame");
      hs8diff_onnx_msemodified_grad->Draw("histsame");
      //hs8diff_onnx_bdt->Draw("histsame");
      //hs8diff_onnx_bdt_grad->Draw("histsame");
      c1_->cd(7)->Print("diffs_kstarmumu_s8.eps", "eps");
      c1_->cd(7)->Print("diffs_kstarmumu_s8.root", "root");

      c1_->cd(8)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hs9diff_onnx_groundtruth->SetLineWidth(1.0);
      hs9diff_onnx_groundtruth->SetLineColor(2);
      hs9diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hs9diff_onnx_groundtruth_grad->SetLineColor(2);
      hs9diff_onnx_groundtruth_grad->SetLineStyle(kDashed);
      hs9diff_onnx_direct->SetLineWidth(1.0);
      hs9diff_onnx_direct->SetLineColor(kYellow+2);
      hs9diff_onnx_direct_grad->SetLineWidth(1.0);
      hs9diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hs9diff_onnx_direct_grad->SetLineStyle(kDashed);
      hs9diff_onnx_mse->SetLineWidth(1.0);
      hs9diff_onnx_mse->SetLineColor(4);
      hs9diff_onnx_mse_grad->SetLineWidth(1.0);
      hs9diff_onnx_mse_grad->SetLineColor(4);
      hs9diff_onnx_mse_grad->SetLineStyle(kDashed);
      hs9diff_onnx_msemodified->SetLineWidth(1.0);
      hs9diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hs9diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hs9diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hs9diff_onnx_msemodified_grad->SetLineStyle(kDashed);
      hs9diff_onnx_bdt->SetLineWidth(1.0);
      hs9diff_onnx_bdt->SetLineColor(kOrange);
      hs9diff_onnx_bdt_grad->SetLineWidth(1.0);
      hs9diff_onnx_bdt_grad->SetLineColor(kOrange);
      hs9diff_onnx_bdt_grad->SetLineStyle(kDashed);
      hs9diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hs9diff_onnx_efficiencytruth->SetLineColor(7);
      hs9diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hs9diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hs9diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);
      hs9diff_bdt->SetLineWidth(1.0);
      hs9diff_bdt->SetLineColor(kGreen+2);

      hs9diff_onnx_groundtruth_grad->SetMaximum(hs9diff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hs9diff_onnx_groundtruth_grad->Draw("hist");
      hs9diff_onnx_groundtruth->Draw("histsame");
      hs9diff_onnx_efficiencytruth->Draw("histsame");
      hs9diff_onnx_efficiencytruth_grad->Draw("histsame");
      hs9diff_bdt->Draw("histsame");
      hs9diff_onnx_direct->Draw("histsame");
      hs9diff_onnx_direct_grad->Draw("histsame");
      hs9diff_onnx_mse->Draw("histsame");
      hs9diff_onnx_mse_grad->Draw("histsame");
      hs9diff_onnx_msemodified->Draw("histsame");
      hs9diff_onnx_msemodified_grad->Draw("histsame");
      //hs9diff_onnx_bdt->Draw("histsame");
      //hs9diff_onnx_bdt_grad->Draw("histsame");
      c1_->cd(8)->Print("diffs_kstarmumu_s9.eps", "eps");
      c1_->cd(8)->Print("diffs_kstarmumu_s9.root", "root");
      
      c1_->cd(9)->SetMargin(0.125, 0.05, 0.125, 0.05);
      /*
      TLegend* leg2 = new TLegend(0.15, 0.15, 0.95, 0.95);
      //leg2->AddEntry(hc1analytic,"analytic truth","l");
      leg2->AddEntry(hfldiff_bdt,"BDT modeling #epsilon","l");
      leg2->AddEntry(hfldiff_onnx_groundtruth,"ONNX groundtruth","l");
      leg2->AddEntry(hfldiff_onnx_groundtruth_grad,"ONNX groundtruth grad.","l");
      leg2->AddEntry(hfldiff_onnx_efficiencytruth,"ONNX efficiencytruth","l");
      leg2->AddEntry(hfldiff_onnx_efficiencytruth_grad,"ONNX efficiencytruth grad.","l");
      leg2->AddEntry(hfldiff_onnx_direct,"ONNX direct","l");
      leg2->AddEntry(hfldiff_onnx_direct_grad,"ONNX direct grad.","l");
      leg2->AddEntry(hfldiff_onnx_mse,"ONNX mse","l");
      leg2->AddEntry(hfldiff_onnx_mse_grad,"ONNX mse grad.","l");
      leg2->AddEntry(hfldiff_onnx_msemodified,"ONNX msemodified","l");
      leg2->AddEntry(hfldiff_onnx_msemodified_grad,"ONNX msemodified grad.","l");
      //leg2->AddEntry(hfldiff_onnx_bdt,"ONNX bdt","l");
      //leg2->AddEntry(hfldiff_onnx_bdt_grad,"ONNX bdt grad.","l");
      leg2->Draw();
      */
      leg->Draw();
      c1_->cd(9)->Print("diffs_kstarmumu_legend.eps", "eps");
      c1_->cd(9)->Print("diffs_kstarmumu_legend.root", "root");

      c1_->Print("diffs_kstarmumu.eps", "eps");
      c1_->Print("diffs_kstarmumu.root", "root");
#endif

      return 0;
#endif
    }

  return 0;
}
