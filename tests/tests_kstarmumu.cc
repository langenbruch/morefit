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

#ifdef WITH_ONNX
#include <onnxruntime/onnxruntime_cxx_api.h>
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

  //check plotting with efficiencies
  if (false)
    {
      
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
  if (true)
    {
      
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

  return 0;
}
