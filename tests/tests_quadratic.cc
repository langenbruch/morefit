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
#include "TLatex.h"
#include "TLine.h"
#include "TLegend.h"
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
  compute_opts.print_kernel = false;
  compute_opts.llvm_print_intermediate = false;
  compute_opts.print();
  
  //typedef morefit::OpenCLBackend backendT;
  //typedef morefit::OpenCLBlock<kernelT, evalT> blockT;
  //morefit::OpenCLBackend backend(&compute_opts);
  
  typedef morefit::LLVMBackend backendT;
  typedef morefit::LLVMBlock<kernelT, evalT> blockT;  
  morefit::LLVMBackend backend(&compute_opts);


  double genc1 = 0.0;
  double genc2 = 0.0;
  //genc1 = 0.0;
  //genc2 = 0.0;
  double startc1 = 0.0;
  double startc2 = 0.0;
  morefit::dimension<evalT> x("x", "x", -1.0, 1.0, false);
  morefit::parameter<evalT> c1("c1", "c_{1}", genc1, -1.0, 1.0, 0.01, false);
  morefit::parameter<evalT> c2("c2", "c_{2}", genc2, -1.0, 1.0, 0.01, false);

  //morefit::QuadraticPDF<kernelT, evalT> quadratic(&x, &c1, &c2);
  std::vector<morefit::parameter<evalT>*> params({&c1, &c2});
  
  morefit::Xoshiro128pp rnd;
  rnd.setSeed(int64_t(229387429));

  morefit::QuadraticPDFAnalyticEps<kernelT, evalT> quadratic_analytic(&x, &c1, &c2);

  //morefit::QuadraticPDFOnnxEps<kernelT, evalT> quadratic_onnx(&x, &c1, &c2, "torch_model_direct_grad_0.onnx");
  morefit::QuadraticPDFOnnxEps<kernelT, evalT> quadratic_onnx(&x, &c1, &c2, "torch_model_direct_4.onnx");

  morefit::QuadraticPDFOnnxEps<kernelT, evalT> quadratic_onnx_grad(&x, &c1, &c2, "torch_model_direct_grad_4.onnx");

  morefit::QuadraticPDF<kernelT, evalT> quadratic_bdt(&x, &c1, &c2);
  morefit::EventVector<kernelT, evalT> eff;
#ifdef WITH_ROOT
  TFile* bdt_file = new TFile("bdt_direct_accvsrej_mse_0.root", "READ");
  TTree* tree = (TTree*)bdt_file->Get("xgboost_regression");
  unsigned int nnodes = tree->GetEntries();      
  quadratic_bdt.set_acceptance_bdt(eff, nnodes);
  double value, x_from, x_to;
  tree->SetBranchAddress("f0_from", &x_from);
  tree->SetBranchAddress("f0_to", &x_to);
  tree->SetBranchAddress("value", &value);
  for (unsigned int i=0; i<tree->GetEntries(); i++)
    {
      tree->GetEntry(i);
      eff(i,0) = value;
      eff(i,1) = x_from;
      eff(i,2) = x_to;
    }
  eff.print();
#endif      

  /*
  //produce graphs
  if (false)
    {
      quadratic.logprob()->draw("graph.tex");
      quadratic.logprob()->simplify()->draw("simplegraph.tex");

      std::vector<std::string> param_names;
      std::vector<evalT> param_values;
      for (auto param : params)
	{
	  param_names.push_back(param->get_name());
	  param_values.push_back(param->get_value());      
	}
      quadratic.prob_normalised()->substitute(param_names, param_values)->simplify()->draw("gen_graph.tex");      
      return 0;
    }
  
  //kernel output
  if (false)
    {
      std::cout << "FULL KERNEL " << quadratic.prob_normalised()->get_kernel() << std::endl;
      std::cout << "SIMPLIFIED KERNEL " << quadratic.prob_normalised()->simplify()->get_kernel() << std::endl;
      
      std::vector<std::string> param_names;
      std::vector<evalT> param_values;
      for (auto param : params)
	{
	  std::cout << "param name: " << param->get_name() << " param value: " << param->get_value() << std::endl;
	  param_names.push_back(param->get_name());
	  param_values.push_back(param->get_value());      
	}
    }
  */
  //check plotting
  if (true)
    {
      unsigned int ngen = 2000;//00;
      std::cout <<"generating" << std::endl;
      morefit::generator_options gen_opts;
      
      morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
      morefit::EventVector<kernelT, evalT> result({&x}, ngen);  
      //gen.generate(ngen, &quadratic_analytic, params, result);      
      //gen.generate(ngen, &quadratic, params, result);      
      
      std::cout <<"fitting" << std::endl;      
      morefit::fitter_options opts;
      opts.minuit_printlevel = -1;
      opts.analytic_gradient = false;
      opts.analytic_hessian = false;
      opts.print_level = -1;
      opts.print();
      morefit::fitter<kernelT, evalT, backendT, blockT > fit(&opts, &backend);
      //fit.fit(&quadratic_analytic, params, &result);//TODO FIXME
      //fit.fit(&quadratic, params, &result);//TODO FIXME

      gen.generate(ngen, &quadratic_analytic, params, result);
      morefit::plotter_options plot_opts;
      //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
      plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
      plot_opts.print_level = 2;
      plot_opts.plot_pulls = true;
      //plot_opts.pull_fraction = 0.25;

      morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
      plot.plot(&quadratic_analytic, params, &result, "x", "plot_x.eps", "eps", 100);
      //plot.plot(&quadratic_bdt, params, &result, "x", "plot_x.eps", "eps", 100);

      //toy study
      unsigned int nruns = 1000;
      std::vector<double> c1_values_analytic(nruns, 0.0);
      std::vector<double> c2_values_analytic(nruns, 0.0);
      std::vector<double> c1_values_onnx(nruns, 0.0);
      std::vector<double> c2_values_onnx(nruns, 0.0);
      std::vector<double> c1_values_onnx_grad(nruns, 0.0);
      std::vector<double> c2_values_onnx_grad(nruns, 0.0);
      std::vector<double> c1_values_bdt(nruns, 0.0);
      std::vector<double> c2_values_bdt(nruns, 0.0);
      for (unsigned int i=0; i<nruns; i++)
	{
	  std::cout << "run no " << i << std::endl;
	  //generate
	  c1.init("c1", "c_{1}", genc1, -1.0, 1.0, 0.01, false);
	  c2.init("c2", "c_{2}", genc2, -1.0, 1.0, 0.01, false);
	  gen.generate(ngen, &quadratic_analytic, params, result);
	  
	  c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	  c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	  fit.fit(&quadratic_onnx, params, &result);//TODO FIXME
	  std::cout << "ONNX RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	  std::cout << "ONNX RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	  c1_values_onnx.at(i) = c1.get_value();
	  c2_values_onnx.at(i) = c2.get_value();
	  
	  c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	  c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	  fit.fit(&quadratic_onnx_grad, params, &result);//TODO FIXME
	  std::cout << "ONNX GRAD RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	  std::cout << "ONNX GRAD RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	  c1_values_onnx_grad.at(i) = c1.get_value();
	  c2_values_onnx_grad.at(i) = c2.get_value();
	  
	  c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	  c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	  fit.fit(&quadratic_analytic, params, &result);//TODO FIXME
	  std::cout << "ANALYTIC RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	  std::cout << "ANALYTIC RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	  c1_values_analytic.at(i) = c1.get_value();
	  c2_values_analytic.at(i) = c2.get_value();

	  c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	  c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	  fit.fit(&quadratic_bdt, params, &result);//TODO FIXME
	  std::cout << "BDT RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	  std::cout << "BDT RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	  c1_values_bdt.at(i) = c1.get_value();
	  c2_values_bdt.at(i) = c2.get_value();

	}
#ifdef WITH_ROOT
      double dx = 0.5;
      double dxdiff = 0.1;
      unsigned int nbins = 50;
      TH1D* hc1analytic = new TH1D("hc1analytic", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2analytic = new TH1D("hc2analytic", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnx = new TH1D("hc1onnx", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx = new TH1D("hc2onnx", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnxgrad = new TH1D("hc1onnxgrad", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnxgrad = new TH1D("hc2onnxgrad", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1bdt = new TH1D("hc1bdt", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2bdt = new TH1D("hc2bdt", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1diff = new TH1D("hc1diff", ";modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff = new TH1D("hc2diff", ";modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff2 = new TH1D("hc1diff2", ";modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff2 = new TH1D("hc2diff2", ";modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff3 = new TH1D("hc1diff3", ";modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff3 = new TH1D("hc2diff3", ";modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      for (unsigned int i=0; i<nruns; i++)
	{
	  hc1diff->Fill(c1_values_onnx.at(i)-c1_values_analytic.at(i));
	  hc2diff->Fill(c2_values_onnx.at(i)-c2_values_analytic.at(i));
	  hc1diff2->Fill(c1_values_onnx_grad.at(i)-c1_values_analytic.at(i));
	  hc2diff2->Fill(c2_values_onnx_grad.at(i)-c2_values_analytic.at(i));
	  hc1diff3->Fill(c1_values_bdt.at(i)-c1_values_analytic.at(i));
	  hc2diff3->Fill(c2_values_bdt.at(i)-c2_values_analytic.at(i));
	  hc1analytic->Fill(c1_values_analytic.at(i));
	  hc2analytic->Fill(c2_values_analytic.at(i));
	  hc1onnx->Fill(c1_values_onnx.at(i));
	  hc2onnx->Fill(c2_values_onnx.at(i));
	  hc1onnxgrad->Fill(c1_values_onnx_grad.at(i));
	  hc2onnxgrad->Fill(c2_values_onnx_grad.at(i));
	  hc1bdt->Fill(c1_values_bdt.at(i));
	  hc2bdt->Fill(c2_values_bdt.at(i));
	}
      double maxy = hc1analytic->GetMaximum();
      if (hc1bdt->GetMaximum() > maxy)
	maxy = hc1bdt->GetMaximum();
      if (hc1onnx->GetMaximum() > maxy)
	maxy = hc1onnx->GetMaximum();
      if (hc1onnxgrad->GetMaximum() > maxy)
	maxy = hc1onnxgrad->GetMaximum();
      hc1analytic->SetMaximum(maxy*1.1);      
      maxy = hc2analytic->GetMaximum();
      if (hc2bdt->GetMaximum() > maxy)
	maxy = hc2bdt->GetMaximum();
      if (hc2onnx->GetMaximum() > maxy)
	maxy = hc2onnx->GetMaximum();
      if (hc2onnxgrad->GetMaximum() > maxy)
	maxy = hc2onnxgrad->GetMaximum();
      hc2analytic->SetMaximum(maxy*1.1);
      maxy = hc1diff->GetMaximum();
      if (hc1diff2->GetMaximum() > maxy)
	maxy = hc1diff2->GetMaximum();
      if (hc1diff3->GetMaximum() > maxy)
	maxy = hc1diff3->GetMaximum();
      hc1diff->SetMaximum(maxy*1.1);
      maxy = hc2diff->GetMaximum();
      if (hc2diff2->GetMaximum() > maxy)
	maxy = hc2diff2->GetMaximum();
      if (hc2diff3->GetMaximum() > maxy)
	maxy = hc2diff3->GetMaximum();
      hc2diff->SetMaximum(maxy*1.1);
      
      TCanvas* c0_ = new TCanvas("c0", "c0", 1600, 800);
      c0_->Divide(2,1);
      c0_->cd(1)->SetMargin(0.15, 0.05, 0.15, 0.05);
      hc1analytic->SetLineWidth(2.0);
      hc1onnx->SetLineWidth(2.0);
      hc1onnx->SetLineColor(2);
      hc1onnxgrad->SetLineWidth(2.0);
      hc1onnxgrad->SetLineColor(kOrange);
      hc1bdt->SetLineWidth(2.0);
      hc1bdt->SetLineColor(4);
      hc1analytic->Draw("hist");
      hc1onnx->Draw("histsame");
      hc1onnxgrad->Draw("histsame");      
      hc1bdt->Draw("histsame");
      TLine* line = new TLine();
      line->SetLineStyle(kDashed);
      line->DrawLine(genc1, 0.0, genc1, hc1analytic->GetMaximum());
      TLatex* tex = new TLatex();
      tex->SetTextAlign(32);
      tex->DrawLatex(genc1-0.05, 0.8*hc1analytic->GetMaximum(), "c_{1} generated");
      TLegend* leg = new TLegend(0.6, 0.7, 0.95, 0.95);
      leg->AddEntry(hc1analytic,"analytic truth","l");
      leg->AddEntry(hc1bdt,"BDT modeling #epsilon","l");
      leg->AddEntry(hc1onnx,"NN modeling N(c_{1},c_{2})","l");
      leg->AddEntry(hc1onnxgrad,"differential NN","l");
      leg->Draw();
      c0_->cd(2)->SetMargin(0.15, 0.05, 0.15, 0.05);
      hc2analytic->SetLineWidth(2.0);
      hc2onnx->SetLineWidth(2.0);
      hc2onnx->SetLineColor(2);
      hc2onnxgrad->SetLineWidth(2.0);
      hc2onnxgrad->SetLineColor(kOrange);
      hc2bdt->SetLineWidth(2.0);
      hc2bdt->SetLineColor(4);
      hc2analytic->Draw("hist");
      hc2onnx->Draw("histsame");      
      hc2onnxgrad->Draw("histsame");      
      hc2bdt->Draw("histsame");
      line->DrawLine(genc2, 0.0, genc2, hc2analytic->GetMaximum());
      tex->DrawLatex(genc2-0.05, 0.8*hc2analytic->GetMaximum(), "c_{2} generated");      
      leg->Draw();
      c0_->Print("diffs.eps", "eps");
      
      TCanvas* c1_ = new TCanvas("c1", "c1", 1600, 800);
      c1_->Divide(2,1);
      c1_->cd(1)->SetMargin(0.15, 0.05, 0.15, 0.05);
      hc1diff->SetLineWidth(2.0);
      hc1diff->SetLineColor(2);
      hc1diff2->SetLineWidth(2.0);
      hc1diff2->SetLineColor(kOrange);
      hc1diff3->SetLineWidth(2.0);
      hc1diff3->SetLineColor(4);
      hc1diff->Draw("hist");
      hc1diff2->Draw("histsame");
      hc1diff3->Draw("histsame");
      leg->Draw();
      c1_->cd(2)->SetMargin(0.15, 0.05, 0.15, 0.05);
      hc2diff->SetLineWidth(2.0);      
      hc2diff->SetLineColor(2);
      hc2diff2->SetLineWidth(2.0);      
      hc2diff2->SetLineColor(kOrange);
      hc2diff3->SetLineWidth(2.0);      
      hc2diff3->SetLineColor(4);
      hc2diff->Draw("hist");
      hc2diff2->Draw("histsame");
      hc2diff3->Draw("histsame");
      leg->Draw();
      c1_->Print("diffs2.eps", "eps");
#endif


      /*
      morefit::plotter_options plot_opts;
      //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
      plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
      plot_opts.print_level = 2;
      plot_opts.plot_pulls = true;
      //plot_opts.pull_fraction = 0.25;
      morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
      plot.plot(&quadratic_analytic, params, &result, "x", "plot_x.eps", "eps", 100);
      //plot.plot(&quadratic, params, &result, "x", "plot_x.eps", "eps", 100);
      */
      /*
      std::vector<std::string> param_names;
      for (unsigned int i=0; i<params.size(); i++)
	param_names.push_back(params.at(i)->get_name());
      std::vector<double> param_values;
      for (unsigned int i=0; i<params.size(); i++)
	param_values.push_back(params.at(i)->get_value());
      std::vector<double> param_errors;
      for (unsigned int i=0; i<params.size(); i++)
	param_errors.push_back(params.at(i)->get_error());
      for (unsigned int i=0; i<params.size(); i++)
	std::cout << param_names.at(i) << " = " << param_values.at(i) << " +- " << param_errors.at(i) << std::endl;
      */
      return 0;
    }
  /*  
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
		  gen.generate(ngen, &quadratic, params, result);
	  
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
		  fit.fit(&quadratic, params, &result);
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
#ifdef WITH_ROOT
      TFile* bdt_file = new TFile("test_quadratic.root", "READ");
      TTree* tree = (TTree*)bdt_file->Get("xgboost_regression");
      unsigned int nnodes = tree->GetEntries();      
      quadratic.set_acceptance_bdt(eff, nnodes);
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
      
      unsigned int ngen = 1000000;
      std::cout <<"generating" << std::endl;
      morefit::generator_options gen_opts;
      
      morefit::generator<kernelT, evalT, backendT, blockT> gen(&gen_opts, &backend, &rnd);
      morefit::EventVector<kernelT, evalT> result({&ctl, &ctk, &phi}, ngen);  
      gen.generate(ngen, &quadratic, params, result);      
      
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
      fit.fit(&quadratic, params, &result);
      
      
      morefit::plotter_options plot_opts;
      //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
      plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
      plot_opts.print_level = 2;
      plot_opts.plot_pulls = true;
      //plot_opts.pull_fraction = 0.25;
      morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
      plot.plot(&quadratic, params, &result, "ctl", "plot_ctl.eps", "eps", 100);
      plot.plot(&quadratic, params, &result, "ctk", "plot_ctk.eps", "eps", 100);
      plot.plot(&quadratic, params, &result, "phi", "plot_phi.eps", "eps", 100);
      
      return 0;
    }

  //check different acceptance approaches
  if (true)
    {
      
      morefit::EventVector<kernelT, evalT> eff;
#ifdef WITH_ROOT
      TFile* bdt_file = new TFile("test_quadratic.root", "READ");
      TTree* tree = (TTree*)bdt_file->Get("xgboost_regression");
      unsigned int nnodes = tree->GetEntries();      
      quadratic.set_acceptance_bdt(eff, nnodes);
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
      Ort::Session session = Ort::Session(env, "mlp_direct_accvsrej_mse_quadratic_19.onnx", session_options);//efficiency model

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
      gen.generate(ngen, &quadratic, params, result);      
      
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
      fit.fit(&quadratic, params, &result);
      
      
      morefit::plotter_options plot_opts;
      //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
      plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
      plot_opts.print_level = 2;
      plot_opts.plot_pulls = true;
      //plot_opts.pull_fraction = 0.25;
      morefit::plotter<kernelT,evalT, backendT, blockT> plot(&plot_opts, &backend);
      plot.plot(&quadratic, params, &result, "ctl", "plot_ctl.eps", "eps", 100);
      plot.plot(&quadratic, params, &result, "ctk", "plot_ctk.eps", "eps", 100);
      plot.plot(&quadratic, params, &result, "phi", "plot_phi.eps", "eps", 100);
      
      return 0;
    }
  */
  return 0;
}
