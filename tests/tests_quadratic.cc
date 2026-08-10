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
  if (false)
    {

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
      c0_->cd(1)->SetMargin(0.125, 0.05, 0.125, 0.05);
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
      c0_->cd(2)->SetMargin(0.125, 0.05, 0.125, 0.05);
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
      c1_->cd(1)->SetMargin(0.125, 0.05, 0.125, 0.05);
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
      c1_->cd(2)->SetMargin(0.125, 0.05, 0.125, 0.05);
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


  //check different models
  if (true)
    {

      unsigned int nmodels = 100;

      //toy study
      unsigned int nruns = 1;
      std::vector<double> c1_values_analytic(nmodels*nruns, 0.0);
      std::vector<double> c2_values_analytic(nmodels*nruns, 0.0);
      
      std::vector<double> c1_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_groundtruth(nmodels*nruns, 0.0);
      std::vector<double> c1_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_groundtruth_grad(nmodels*nruns, 0.0);

      std::vector<double> c1_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_direct(nmodels*nruns, 0.0);
      std::vector<double> c1_values_onnx_direct_grad(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_direct_grad(nmodels*nruns, 0.0);

      std::vector<double> c1_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_mse(nmodels*nruns, 0.0);
      std::vector<double> c1_values_onnx_mse_grad(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_mse_grad(nmodels*nruns, 0.0);

      std::vector<double> c1_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_msemodified(nmodels*nruns, 0.0);
      std::vector<double> c1_values_onnx_msemodified_grad(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_msemodified_grad(nmodels*nruns, 0.0);

      std::vector<double> c1_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_bdt(nmodels*nruns, 0.0);
      std::vector<double> c1_values_onnx_bdt_grad(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_bdt_grad(nmodels*nruns, 0.0);

      std::vector<double> c1_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_efficiencytruth(nmodels*nruns, 0.0);
      std::vector<double> c1_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);
      std::vector<double> c2_values_onnx_efficiencytruth_grad(nmodels*nruns, 0.0);


      std::vector<double> c1_values_bdt(nmodels*nruns, 0.0);
      std::vector<double> c2_values_bdt(nmodels*nruns, 0.0);

      for (unsigned int m=0; m<nmodels; m++)
	{
	  
	  morefit::QuadraticPDFNormalisedAnalyticEps<kernelT, evalT> quadratic_analytic(&x, &c1, &c2);

	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_groundtruth(&x, &c1, &c2, ("weights/torch_model_groundtruth_"+std::to_string(m)+".onnx").c_str());	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_groundtruth_grad(&x, &c1, &c2, ("weights/torch_model_groundtruth_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_direct(&x, &c1, &c2, ("weights/torch_model_direct_"+std::to_string(m)+".onnx").c_str());	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_direct_grad(&x, &c1, &c2, ("weights/torch_model_direct_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_mse(&x, &c1, &c2, ("weights/torch_model_efficiency_mse_"+std::to_string(m)+".onnx").c_str());	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_mse_grad(&x, &c1, &c2, ("weights/torch_model_efficiency_mse_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_msemodified(&x, &c1, &c2, ("weights/torch_model_efficiency_msemodified_"+std::to_string(m)+".onnx").c_str());	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_msemodified_grad(&x, &c1, &c2, ("weights/torch_model_efficiency_msemodified_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_bdt(&x, &c1, &c2, ("weights/torch_model_bdt_"+std::to_string(m)+".onnx").c_str());	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_bdt_grad(&x, &c1, &c2, ("weights/torch_model_bdt_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_efficiencytruth(&x, &c1, &c2, ("weights/torch_model_efficiency_truth_"+std::to_string(m)+".onnx").c_str());	  
	  morefit::QuadraticPDFNormalisedOnnxEps<kernelT, evalT> quadratic_onnx_efficiencytruth_grad(&x, &c1, &c2, ("weights/torch_model_efficiency_truth_grad_"+std::to_string(m)+".onnx").c_str());
	  
	  morefit::QuadraticPDFNormalised<kernelT, evalT> quadratic_bdt(&x, &c1, &c2);
	  
	  morefit::EventVector<kernelT, evalT> eff;
#ifdef WITH_ROOT
	  TFile* bdt_file = new TFile(("weights/bdt_direct_accvsrej_mse_"+std::to_string(m)+".root").c_str(), "READ");
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
	  //eff.print();
#endif      
      
	  unsigned int ngen = 100000;//00;
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

	  gen.generate(ngen, &quadratic_analytic, params, result);
	  morefit::plotter_options plot_opts;
	  //plot_opts.plotter = morefit::plotter_options::plotter_type::MatPlotLib;
	  plot_opts.plotter = morefit::plotter_options::plotter_type::Root;
	  plot_opts.print_level = 2;
	  plot_opts.plot_pulls = true;
	  //plot_opts.pull_fraction = 0.25;

	  for (unsigned int i=0; i<nruns; i++)
	    {
	      std::cout << "model no " << m << " run no " << i << std::endl;
	      //generate
	      c1.init("c1", "c_{1}", genc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", genc2, -1.0, 1.0, 0.01, false);
	      gen.generate(ngen, &quadratic_analytic, params, result);
	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_groundtruth, params, &result);
	      std::cout << "ONNX_GROUNDTRUTH RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_GROUNDTRUTH RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_groundtruth.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_groundtruth.at(m*nruns+i) = c2.get_value();	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_groundtruth_grad, params, &result);
	      std::cout << "ONNX_GROUNDTRUTH GRAD RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_GROUNDTRUTH GRAD RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_groundtruth_grad.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_groundtruth_grad.at(m*nruns+i) = c2.get_value();
	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_direct, params, &result);
	      std::cout << "ONNX_DIRECT RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_DIRECT RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_direct.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_direct.at(m*nruns+i) = c2.get_value();	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_direct_grad, params, &result);
	      std::cout << "ONNX_DIRECT GRAD RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_DIRECT GRAD RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_direct_grad.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_direct_grad.at(m*nruns+i) = c2.get_value();
	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_mse, params, &result);
	      std::cout << "ONNX_MSE RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_MSE RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_mse.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_mse.at(m*nruns+i) = c2.get_value();	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_mse_grad, params, &result);
	      std::cout << "ONNX_MSE GRAD RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_MSE GRAD RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_mse_grad.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_mse_grad.at(m*nruns+i) = c2.get_value();
	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_msemodified, params, &result);
	      std::cout << "ONNX_MSEMODIFIED RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_MSEMODIFIED RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_msemodified.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_msemodified.at(m*nruns+i) = c2.get_value();	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_msemodified_grad, params, &result);
	      std::cout << "ONNX_MSEMODIFIED GRAD RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_MSEMODIFIED GRAD RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_msemodified_grad.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_msemodified_grad.at(m*nruns+i) = c2.get_value();
	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_bdt, params, &result);
	      std::cout << "ONNX_BDT RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_BDT RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_bdt.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_bdt.at(m*nruns+i) = c2.get_value();	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_bdt_grad, params, &result);
	      std::cout << "ONNX_BDT GRAD RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_BDT GRAD RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_bdt_grad.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_bdt_grad.at(m*nruns+i) = c2.get_value();
	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_efficiencytruth, params, &result);
	      std::cout << "ONNX_EFFICIENCYTRUTH RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_EFFICIENCYTRUTH RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_efficiencytruth.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_efficiencytruth.at(m*nruns+i) = c2.get_value();	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_onnx_efficiencytruth_grad, params, &result);
	      std::cout << "ONNX_EFFICIENCYTRUTH GRAD RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ONNX_EFFICIENCYTRUTH GRAD RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_onnx_efficiencytruth_grad.at(m*nruns+i) = c1.get_value();
	      c2_values_onnx_efficiencytruth_grad.at(m*nruns+i) = c2.get_value();
	  
	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_analytic, params, &result);
	      std::cout << "ANALYTIC RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "ANALYTIC RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_analytic.at(m*nruns+i) = c1.get_value();
	      c2_values_analytic.at(m*nruns+i) = c2.get_value();

	      c1.init("c1", "c_{1}", startc1, -1.0, 1.0, 0.01, false);
	      c2.init("c2", "c_{2}", startc2, -1.0, 1.0, 0.01, false);
	      fit.fit(&quadratic_bdt, params, &result);
	      std::cout << "BDT RESULT c1 = " << c1.get_value() << "+-" << c1.get_error() << std::endl;
	      std::cout << "BDT RESULT c2 = " << c2.get_value() << "+-" << c2.get_error() << std::endl;
	      c1_values_bdt.at(m*nruns+i) = c1.get_value();
	      c2_values_bdt.at(m*nruns+i) = c2.get_value();

	    }
	}
      std::vector<std::string> methods = {"groundtruth", "groundtruth grad", "$\\epsilon$ truth", "$\\epsilon$ truth grad", "BDT", "modeled BDT", "modeled BDT grad", "direct", "direct grad", "mse", "mse grad", "msemodified", "msemodified grad"};
      std::vector<std::vector<double>> analytic_values = {c1_values_analytic, c2_values_analytic};
      std::vector<std::string> observables = {"$c_{1}$", "$c_{2}$"};
      std::vector<std::vector<double>> values_onnx_groundtruth = {c1_values_onnx_groundtruth, c2_values_onnx_groundtruth};
      std::vector<std::vector<double>> values_onnx_groundtruth_grad = {c1_values_onnx_groundtruth_grad, c2_values_onnx_groundtruth_grad};
      std::vector<std::vector<double>> values_onnx_efficiencytruth = {c1_values_onnx_efficiencytruth, c2_values_onnx_efficiencytruth};
      std::vector<std::vector<double>> values_onnx_efficiencytruth_grad = {c1_values_onnx_efficiencytruth_grad, c2_values_onnx_efficiencytruth_grad};
      std::vector<std::vector<double>> values_bdt = {c1_values_bdt, c2_values_bdt};
      std::vector<std::vector<double>> values_onnx_bdt = {c1_values_onnx_bdt, c2_values_onnx_bdt};
      std::vector<std::vector<double>> values_onnx_bdt_grad = {c1_values_onnx_bdt_grad, c2_values_onnx_bdt_grad};
      std::vector<std::vector<double>> values_onnx_direct = {c1_values_onnx_direct, c2_values_onnx_direct};
      std::vector<std::vector<double>> values_onnx_direct_grad = {c1_values_onnx_direct_grad, c2_values_onnx_direct_grad};
      std::vector<std::vector<double>> values_onnx_mse = {c1_values_onnx_mse, c2_values_onnx_mse};
      std::vector<std::vector<double>> values_onnx_mse_grad = {c1_values_onnx_mse_grad, c2_values_onnx_mse_grad};
      std::vector<std::vector<double>> values_onnx_msemodified = {c1_values_onnx_msemodified, c2_values_onnx_msemodified};
      std::vector<std::vector<double>> values_onnx_msemodified_grad = {c1_values_onnx_msemodified_grad, c2_values_onnx_msemodified_grad};
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
      
      double dx = 0.2;
      double dxdiff = 0.1;
      unsigned int nbins = 50;
      TH1D* hc1analytic = new TH1D("hc1analytic", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2analytic = new TH1D("hc2analytic", ";c_{2};#entries", nbins, -dx, +dx);
      
      TH1D* hc1onnx_groundtruth = new TH1D("hc1onnx_groundtruth", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_groundtruth = new TH1D("hc2onnx_groundtruth", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnx_groundtruth_grad = new TH1D("hc1onnx_groundtruth_grad", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_groundtruth_grad = new TH1D("hc2onnx_groundtruth_grad", ";c_{2};#entries", nbins, -dx, +dx);
      
      TH1D* hc1onnx_direct = new TH1D("hc1onnx_direct", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_direct = new TH1D("hc2onnx_direct", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnx_direct_grad = new TH1D("hc1onnx_direct_grad", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_direct_grad = new TH1D("hc2onnx_direct_grad", ";c_{2};#entries", nbins, -dx, +dx);
      
      TH1D* hc1onnx_mse = new TH1D("hc1onnx_mse", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_mse = new TH1D("hc2onnx_mse", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnx_mse_grad = new TH1D("hc1onnx_mse_grad", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_mse_grad = new TH1D("hc2onnx_mse_grad", ";c_{2};#entries", nbins, -dx, +dx);
      
      TH1D* hc1onnx_msemodified = new TH1D("hc1onnx_msemodified", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_msemodified = new TH1D("hc2onnx_msemodified", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnx_msemodified_grad = new TH1D("hc1onnx_msemodified_grad", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_msemodified_grad = new TH1D("hc2onnx_msemodified_grad", ";c_{2};#entries", nbins, -dx, +dx);
      
      TH1D* hc1onnx_bdt = new TH1D("hc1onnx_bdt", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_bdt = new TH1D("hc2onnx_bdt", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnx_bdt_grad = new TH1D("hc1onnx_bdt_grad", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_bdt_grad = new TH1D("hc2onnx_bdt_grad", ";c_{2};#entries", nbins, -dx, +dx);
      
      TH1D* hc1onnx_efficiencytruth = new TH1D("hc1onnx_efficiencytruth", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_efficiencytruth = new TH1D("hc2onnx_efficiencytruth", ";c_{2};#entries", nbins, -dx, +dx);
      TH1D* hc1onnx_efficiencytruth_grad = new TH1D("hc1onnx_efficiencytruth_grad", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2onnx_efficiencytruth_grad = new TH1D("hc2onnx_efficiencytruth_grad", ";c_{2};#entries", nbins, -dx, +dx);
      
      TH1D* hc1bdt = new TH1D("hc1bdt", ";c_{1};#entries", nbins, -dx, +dx);
      TH1D* hc2bdt = new TH1D("hc2bdt", ";c_{2};#entries", nbins, -dx, +dx);


      TH1D* hc1diff_onnx_groundtruth = new TH1D("hc1diff_onnx_groundtruth", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_groundtruth = new TH1D("hc2diff_onnx_groundtruth", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff_onnx_groundtruth_grad = new TH1D("hc1diff_onnx_groundtruth_grad", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_groundtruth_grad = new TH1D("hc2diff_onnx_groundtruth_grad", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      
      TH1D* hc1diff_onnx_direct = new TH1D("hc1diff_onnx_direct", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_direct = new TH1D("hc2diff_onnx_direct", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff_onnx_direct_grad = new TH1D("hc1diff_onnx_direct_grad", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_direct_grad = new TH1D("hc2diff_onnx_direct_grad", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      
      TH1D* hc1diff_onnx_mse = new TH1D("hc1diff_onnx_mse", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_mse = new TH1D("hc2diff_onnx_mse", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff_onnx_mse_grad = new TH1D("hc1diff_onnx_mse_grad", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_mse_grad = new TH1D("hc2diff_onnx_mse_grad", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      
      TH1D* hc1diff_onnx_msemodified = new TH1D("hc1diff_onnx_msemodified", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_msemodified = new TH1D("hc2diff_onnx_msemodified", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff_onnx_msemodified_grad = new TH1D("hc1diff_onnx_msemodified_grad", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_msemodified_grad = new TH1D("hc2diff_onnx_msemodified_grad", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      
      TH1D* hc1diff_onnx_bdt = new TH1D("hc1diff_onnx_bdt", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_bdt = new TH1D("hc2diff_onnx_bdt", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff_onnx_bdt_grad = new TH1D("hc1diff_onnx_bdt_grad", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_bdt_grad = new TH1D("hc2diff_onnx_bdt_grad", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      
      TH1D* hc1diff_onnx_efficiencytruth = new TH1D("hc1diff_onnx_efficiencytruth", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_efficiencytruth = new TH1D("hc2diff_onnx_efficiencytruth", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc1diff_onnx_efficiencytruth_grad = new TH1D("hc1diff_onnx_efficiencytruth_grad", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_onnx_efficiencytruth_grad = new TH1D("hc2diff_onnx_efficiencytruth_grad", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      
      TH1D* hc1diff_bdt = new TH1D("hc1diff_bdt", ";c_{1} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      TH1D* hc2diff_bdt = new TH1D("hc2diff_bdt", ";c_{2} modeled-analytic;#entries", nbins, -dxdiff, +dxdiff);
      for (unsigned int i=0; i<nruns*nmodels; i++)
	{
	  //differences to analytic
	  hc1diff_onnx_groundtruth->Fill(c1_values_onnx_groundtruth.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_groundtruth->Fill(c2_values_onnx_groundtruth.at(i)-c2_values_analytic.at(i));
	  hc1diff_onnx_groundtruth_grad->Fill(c1_values_onnx_groundtruth_grad.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_groundtruth_grad->Fill(c2_values_onnx_groundtruth_grad.at(i)-c2_values_analytic.at(i));
	  
	  hc1diff_onnx_direct->Fill(c1_values_onnx_direct.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_direct->Fill(c2_values_onnx_direct.at(i)-c2_values_analytic.at(i));
	  hc1diff_onnx_direct_grad->Fill(c1_values_onnx_direct_grad.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_direct_grad->Fill(c2_values_onnx_direct_grad.at(i)-c2_values_analytic.at(i));
	  
	  hc1diff_onnx_mse->Fill(c1_values_onnx_mse.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_mse->Fill(c2_values_onnx_mse.at(i)-c2_values_analytic.at(i));
	  hc1diff_onnx_mse_grad->Fill(c1_values_onnx_mse_grad.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_mse_grad->Fill(c2_values_onnx_mse_grad.at(i)-c2_values_analytic.at(i));
	  
	  hc1diff_onnx_msemodified->Fill(c1_values_onnx_msemodified.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_msemodified->Fill(c2_values_onnx_msemodified.at(i)-c2_values_analytic.at(i));
	  hc1diff_onnx_msemodified_grad->Fill(c1_values_onnx_msemodified_grad.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_msemodified_grad->Fill(c2_values_onnx_msemodified_grad.at(i)-c2_values_analytic.at(i));
	  
	  hc1diff_onnx_bdt->Fill(c1_values_onnx_bdt.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_bdt->Fill(c2_values_onnx_bdt.at(i)-c2_values_analytic.at(i));
	  hc1diff_onnx_bdt_grad->Fill(c1_values_onnx_bdt_grad.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_bdt_grad->Fill(c2_values_onnx_bdt_grad.at(i)-c2_values_analytic.at(i));
	  
	  hc1diff_onnx_efficiencytruth->Fill(c1_values_onnx_efficiencytruth.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_efficiencytruth->Fill(c2_values_onnx_efficiencytruth.at(i)-c2_values_analytic.at(i));
	  hc1diff_onnx_efficiencytruth_grad->Fill(c1_values_onnx_efficiencytruth_grad.at(i)-c1_values_analytic.at(i));
	  hc2diff_onnx_efficiencytruth_grad->Fill(c2_values_onnx_efficiencytruth_grad.at(i)-c2_values_analytic.at(i));
	  	  
	  hc1diff_bdt->Fill(c1_values_bdt.at(i)-c1_values_analytic.at(i));
	  hc2diff_bdt->Fill(c2_values_bdt.at(i)-c2_values_analytic.at(i));
	  //values
	  hc1analytic->Fill(c1_values_analytic.at(i));
	  hc2analytic->Fill(c2_values_analytic.at(i));

	  hc1onnx_groundtruth->Fill(c1_values_onnx_groundtruth.at(i));
	  hc2onnx_groundtruth->Fill(c2_values_onnx_groundtruth.at(i));
	  hc1onnx_groundtruth_grad->Fill(c1_values_onnx_groundtruth_grad.at(i));
	  hc2onnx_groundtruth_grad->Fill(c2_values_onnx_groundtruth_grad.at(i));

	  hc1onnx_direct->Fill(c1_values_onnx_direct.at(i));
	  hc2onnx_direct->Fill(c2_values_onnx_direct.at(i));
	  hc1onnx_direct_grad->Fill(c1_values_onnx_direct_grad.at(i));
	  hc2onnx_direct_grad->Fill(c2_values_onnx_direct_grad.at(i));

	  hc1onnx_mse->Fill(c1_values_onnx_mse.at(i));
	  hc2onnx_mse->Fill(c2_values_onnx_mse.at(i));
	  hc1onnx_mse_grad->Fill(c1_values_onnx_mse_grad.at(i));
	  hc2onnx_mse_grad->Fill(c2_values_onnx_mse_grad.at(i));

	  hc1onnx_msemodified->Fill(c1_values_onnx_msemodified.at(i));
	  hc2onnx_msemodified->Fill(c2_values_onnx_msemodified.at(i));
	  hc1onnx_msemodified_grad->Fill(c1_values_onnx_msemodified_grad.at(i));
	  hc2onnx_msemodified_grad->Fill(c2_values_onnx_msemodified_grad.at(i));

	  hc1onnx_bdt->Fill(c1_values_onnx_bdt.at(i));
	  hc2onnx_bdt->Fill(c2_values_onnx_bdt.at(i));
	  hc1onnx_bdt_grad->Fill(c1_values_onnx_bdt_grad.at(i));
	  hc2onnx_bdt_grad->Fill(c2_values_onnx_bdt_grad.at(i));

	  hc1onnx_efficiencytruth->Fill(c1_values_onnx_efficiencytruth.at(i));
	  hc2onnx_efficiencytruth->Fill(c2_values_onnx_efficiencytruth.at(i));
	  hc1onnx_efficiencytruth_grad->Fill(c1_values_onnx_efficiencytruth_grad.at(i));
	  hc2onnx_efficiencytruth_grad->Fill(c2_values_onnx_efficiencytruth_grad.at(i));

	  hc1bdt->Fill(c1_values_bdt.at(i));
	  hc2bdt->Fill(c2_values_bdt.at(i));
	}
      hc1analytic->SetMaximum(hc1analytic->GetMaximum()*1.25);
      hc2analytic->SetMaximum(hc2analytic->GetMaximum()*1.25);
      hc1diff_onnx_groundtruth_grad->SetMaximum(hc1diff_onnx_groundtruth_grad->GetMaximum()*1.25);
      hc2diff_onnx_groundtruth_grad->SetMaximum(hc2diff_onnx_groundtruth_grad->GetMaximum()*1.25);
 
      TCanvas* c0_ = new TCanvas("c0", "c0", 1600, 800);
      c0_->Divide(2,1);
      c0_->cd(1)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hc1analytic->SetLineWidth(1.0);

      hc1onnx_groundtruth->SetLineWidth(1.0);
      hc1onnx_groundtruth->SetLineColor(2);
      hc1onnx_groundtruth_grad->SetLineWidth(1.0);
      hc1onnx_groundtruth_grad->SetLineColor(2);
      hc1onnx_groundtruth_grad->SetLineStyle(kDashed);	
      hc1onnx_direct->SetLineWidth(1.0);
      hc1onnx_direct->SetLineColor(kYellow+2);
      hc1onnx_direct_grad->SetLineWidth(1.0);
      hc1onnx_direct_grad->SetLineColor(kYellow+2);
      hc1onnx_direct_grad->SetLineStyle(kDashed);	
      hc1onnx_mse->SetLineWidth(1.0);
      hc1onnx_mse->SetLineColor(4);
      hc1onnx_mse_grad->SetLineWidth(1.0);
      hc1onnx_mse_grad->SetLineColor(4);
      hc1onnx_mse_grad->SetLineStyle(kDashed);      
      hc1onnx_msemodified->SetLineWidth(1.0);
      hc1onnx_msemodified->SetLineColor(kMagenta+1);
      hc1onnx_msemodified_grad->SetLineWidth(1.0);
      hc1onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hc1onnx_msemodified_grad->SetLineStyle(kDashed);      
      hc1onnx_bdt->SetLineWidth(1.0);
      hc1onnx_bdt->SetLineColor(kOrange);
      hc1onnx_bdt_grad->SetLineWidth(1.0);
      hc1onnx_bdt_grad->SetLineColor(kOrange);
      hc1onnx_bdt_grad->SetLineStyle(kDashed);      
      hc1onnx_efficiencytruth->SetLineWidth(1.0);
      hc1onnx_efficiencytruth->SetLineColor(7);
      hc1onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hc1onnx_efficiencytruth_grad->SetLineColor(7);
      hc1onnx_efficiencytruth_grad->SetLineStyle(kDashed);      
      hc1bdt->SetLineWidth(1.0);
      hc1bdt->SetLineColor(kGreen+2);
      
      hc1analytic->Draw("hist");      
      hc1onnx_groundtruth->Draw("histsame");
      hc1onnx_groundtruth_grad->Draw("histsame");      
      hc1onnx_efficiencytruth->Draw("histsame");
      hc1onnx_efficiencytruth_grad->Draw("histsame");
      hc1onnx_direct->Draw("histsame");
      hc1onnx_direct_grad->Draw("histsame");
      hc1onnx_mse->Draw("histsame");
      hc1onnx_mse_grad->Draw("histsame");      
      hc1onnx_msemodified->Draw("histsame");
      hc1onnx_msemodified_grad->Draw("histsame");      
      //hc1onnx_bdt->Draw("histsame");
      //hc1onnx_bdt_grad->Draw("histsame");      
      hc1bdt->Draw("histsame");

      TLine* line = new TLine();
      line->SetLineStyle(kDashed);
      line->DrawLine(genc1, 0.0, genc1, hc1analytic->GetMaximum());
      TLatex* tex = new TLatex();
      tex->SetTextAlign(32);
      tex->DrawLatex(genc1-0.05, 0.8*hc1analytic->GetMaximum(), "c_{1} generated");
      TLegend* leg = new TLegend(0.6, 0.5, 0.95, 0.95);
      leg->AddEntry(hc1analytic,"analytic truth","l");
      leg->AddEntry(hc1bdt,"BDT modeling #epsilon","l");
      leg->AddEntry(hc1onnx_groundtruth,"ONNX groundtruth","l");
      leg->AddEntry(hc1onnx_groundtruth_grad,"ONNX groundtruth grad.","l");
      leg->AddEntry(hc1onnx_efficiencytruth,"ONNX efficiencytruth","l");
      leg->AddEntry(hc1onnx_efficiencytruth_grad,"ONNX efficiencytruth grad.","l");
      leg->AddEntry(hc1onnx_direct,"ONNX direct","l");
      leg->AddEntry(hc1onnx_direct_grad,"ONNX direct grad.","l");
      leg->AddEntry(hc1onnx_mse,"ONNX mse","l");
      leg->AddEntry(hc1onnx_mse_grad,"ONNX mse grad.","l");
      leg->AddEntry(hc1onnx_msemodified,"ONNX msemodified","l");
      leg->AddEntry(hc1onnx_msemodified_grad,"ONNX msemodified grad.","l");
      //leg->AddEntry(hc1onnx_bdt,"ONNX bdt","l");
      //leg->AddEntry(hc1onnx_bdt_grad,"ONNX bdt grad.","l");
      leg->Draw();
      c0_->cd(1)->Print("values_c1.eps", "eps");
      c0_->cd(1)->Print("values_c1.root", "root");
      
      c0_->cd(2)->SetMargin(0.125, 0.05, 0.125, 0.05);
      hc2analytic->SetLineWidth(1.0);

      hc2onnx_groundtruth->SetLineWidth(1.0);
      hc2onnx_groundtruth->SetLineColor(2);
      hc2onnx_groundtruth_grad->SetLineWidth(1.0);
      hc2onnx_groundtruth_grad->SetLineColor(2);
      hc2onnx_groundtruth_grad->SetLineStyle(kDashed);	
      hc2onnx_direct->SetLineWidth(1.0);
      hc2onnx_direct->SetLineColor(kYellow+2);
      hc2onnx_direct_grad->SetLineWidth(1.0);
      hc2onnx_direct_grad->SetLineColor(kYellow+2);
      hc2onnx_direct_grad->SetLineStyle(kDashed);	
      hc2onnx_mse->SetLineWidth(1.0);
      hc2onnx_mse->SetLineColor(4);
      hc2onnx_mse_grad->SetLineWidth(1.0);
      hc2onnx_mse_grad->SetLineColor(4);
      hc2onnx_mse_grad->SetLineStyle(kDashed);      
      hc2onnx_msemodified->SetLineWidth(1.0);
      hc2onnx_msemodified->SetLineColor(kMagenta+1);
      hc2onnx_msemodified_grad->SetLineWidth(1.0);
      hc2onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hc2onnx_msemodified_grad->SetLineStyle(kDashed);      
      hc2onnx_bdt->SetLineWidth(1.0);
      hc2onnx_bdt->SetLineColor(kOrange);
      hc2onnx_bdt_grad->SetLineWidth(1.0);
      hc2onnx_bdt_grad->SetLineColor(kOrange);
      hc2onnx_bdt_grad->SetLineStyle(kDashed);      
      hc2onnx_efficiencytruth->SetLineWidth(1.0);
      hc2onnx_efficiencytruth->SetLineColor(7);
      hc2onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hc2onnx_efficiencytruth_grad->SetLineColor(7);
      hc2onnx_efficiencytruth_grad->SetLineStyle(kDashed);      
      hc2bdt->SetLineWidth(1.0);
      hc2bdt->SetLineColor(kGreen+2);

      hc2analytic->Draw("hist");      
      hc2onnx_groundtruth->Draw("histsame");
      hc2onnx_groundtruth_grad->Draw("histsame");      
      hc2onnx_efficiencytruth->Draw("histsame");
      hc2onnx_efficiencytruth_grad->Draw("histsame");      
      hc2onnx_direct->Draw("histsame");
      hc2onnx_direct_grad->Draw("histsame");      
      hc2onnx_mse->Draw("histsame");
      hc2onnx_mse_grad->Draw("histsame");      
      hc2onnx_msemodified->Draw("histsame");
      hc2onnx_msemodified_grad->Draw("histsame");      
      //hc2onnx_bdt->Draw("histsame");
      //hc2onnx_bdt_grad->Draw("histsame");      
      hc2bdt->Draw("histsame");

      line->DrawLine(genc2, 0.0, genc2, hc2analytic->GetMaximum());
      tex->DrawLatex(genc2-0.05, 0.8*hc2analytic->GetMaximum(), "c_{2} generated");      
      leg->Draw();
      c0_->cd(2)->Print("values_c2.eps", "eps");
      c0_->cd(2)->Print("values_c2.root", "root");
      c0_->Print("values.eps", "eps");
      c0_->Print("values.root", "root");
      
      TCanvas* c1_ = new TCanvas("c1", "c1", 1600, 800);
      c1_->Divide(2,1);
      c1_->cd(1)->SetMargin(0.125, 0.05, 0.125, 0.05);

      hc1diff_onnx_groundtruth->SetLineWidth(1.0);
      hc1diff_onnx_groundtruth->SetLineColor(2);
      hc1diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hc1diff_onnx_groundtruth_grad->SetLineColor(2);
      hc1diff_onnx_groundtruth_grad->SetLineStyle(kDashed);      
      hc1diff_onnx_direct->SetLineWidth(1.0);
      hc1diff_onnx_direct->SetLineColor(kYellow+2);
      hc1diff_onnx_direct_grad->SetLineWidth(1.0);
      hc1diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hc1diff_onnx_direct_grad->SetLineStyle(kDashed);	
      hc1diff_onnx_mse->SetLineWidth(1.0);
      hc1diff_onnx_mse->SetLineColor(4);
      hc1diff_onnx_mse_grad->SetLineWidth(1.0);
      hc1diff_onnx_mse_grad->SetLineColor(4);
      hc1diff_onnx_mse_grad->SetLineStyle(kDashed);      
      hc1diff_onnx_msemodified->SetLineWidth(1.0);
      hc1diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hc1diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hc1diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hc1diff_onnx_msemodified_grad->SetLineStyle(kDashed);      
      hc1diff_onnx_bdt->SetLineWidth(1.0);
      hc1diff_onnx_bdt->SetLineColor(kOrange);
      hc1diff_onnx_bdt_grad->SetLineWidth(1.0);
      hc1diff_onnx_bdt_grad->SetLineColor(kOrange);
      hc1diff_onnx_bdt_grad->SetLineStyle(kDashed);      
      hc1diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hc1diff_onnx_efficiencytruth->SetLineColor(7);
      hc1diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hc1diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hc1diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);      
      hc1diff_bdt->SetLineWidth(1.0);
      hc1diff_bdt->SetLineColor(kGreen+2);
      
      hc1diff_onnx_groundtruth_grad->Draw("hist");      
      hc1diff_onnx_groundtruth->Draw("histsame");
      hc1diff_onnx_efficiencytruth->Draw("histsame");
      hc1diff_onnx_efficiencytruth_grad->Draw("histsame");      
      hc1diff_onnx_direct->Draw("histsame");
      hc1diff_onnx_direct_grad->Draw("histsame");      
      hc1diff_onnx_mse->Draw("histsame");
      hc1diff_onnx_mse_grad->Draw("histsame");      
      hc1diff_onnx_msemodified->Draw("histsame");
      hc1diff_onnx_msemodified_grad->Draw("histsame");      
      //hc1diff_onnx_bdt->Draw("histsame");
      //hc1diff_onnx_bdt_grad->Draw("histsame");      
      hc1diff_bdt->Draw("histsame");
      
      //hc1diff_onnx_groundtruth->Draw("histsame");
      //hc1diff_onnx_groundtruth_grad->Draw("histsame");      

      TLegend* leg2 = new TLegend(0.6, 0.5, 0.95, 0.95);
      //leg2->AddEntry(hc1analytic,"analytic truth","l");
      leg2->AddEntry(hc1bdt,"BDT modeling #epsilon","l");
      leg2->AddEntry(hc1onnx_groundtruth,"ONNX groundtruth","l");
      leg2->AddEntry(hc1onnx_groundtruth_grad,"ONNX groundtruth grad.","l");
      leg2->AddEntry(hc1onnx_direct,"ONNX direct","l");
      leg2->AddEntry(hc1onnx_direct_grad,"ONNX direct grad.","l");
      leg2->AddEntry(hc1onnx_mse,"ONNX mse","l");
      leg2->AddEntry(hc1onnx_mse_grad,"ONNX mse grad.","l");
      leg2->AddEntry(hc1onnx_msemodified,"ONNX msemodified","l");
      leg2->AddEntry(hc1onnx_msemodified_grad,"ONNX msemodified grad.","l");
      //leg2->AddEntry(hc1onnx_bdt,"ONNX bdt","l");
      //leg2->AddEntry(hc1onnx_bdt_grad,"ONNX bdt grad.","l");
      leg2->AddEntry(hc1onnx_efficiencytruth,"ONNX efficiencytruth","l");
      leg2->AddEntry(hc1onnx_efficiencytruth_grad,"ONNX efficiencytruth grad.","l");
      leg2->Draw();
      c1_->cd(1)->Print("diffs_c1.eps", "eps");
      c1_->cd(1)->Print("diffs_c1.root", "root");
      
      c1_->cd(2)->SetMargin(0.125, 0.05, 0.125, 0.05);

      hc2diff_onnx_groundtruth->SetLineWidth(1.0);
      hc2diff_onnx_groundtruth->SetLineColor(2);
      hc2diff_onnx_groundtruth_grad->SetLineWidth(1.0);
      hc2diff_onnx_groundtruth_grad->SetLineColor(2);
      hc2diff_onnx_groundtruth_grad->SetLineStyle(kDashed);	
      hc2diff_onnx_direct->SetLineWidth(1.0);
      hc2diff_onnx_direct->SetLineColor(kYellow+2);
      hc2diff_onnx_direct_grad->SetLineWidth(1.0);
      hc2diff_onnx_direct_grad->SetLineColor(kYellow+2);
      hc2diff_onnx_direct_grad->SetLineStyle(kDashed);	
      hc2diff_onnx_mse->SetLineWidth(1.0);
      hc2diff_onnx_mse->SetLineColor(4);
      hc2diff_onnx_mse_grad->SetLineWidth(1.0);
      hc2diff_onnx_mse_grad->SetLineColor(4);
      hc2diff_onnx_mse_grad->SetLineStyle(kDashed);      
      hc2diff_onnx_msemodified->SetLineWidth(1.0);
      hc2diff_onnx_msemodified->SetLineColor(kMagenta+1);
      hc2diff_onnx_msemodified_grad->SetLineWidth(1.0);
      hc2diff_onnx_msemodified_grad->SetLineColor(kMagenta+1);
      hc2diff_onnx_msemodified_grad->SetLineStyle(kDashed);      
      hc2diff_onnx_bdt->SetLineWidth(1.0);
      hc2diff_onnx_bdt->SetLineColor(kOrange);
      hc2diff_onnx_bdt_grad->SetLineWidth(1.0);
      hc2diff_onnx_bdt_grad->SetLineColor(kOrange);
      hc2diff_onnx_bdt_grad->SetLineStyle(kDashed);      
      hc2diff_onnx_efficiencytruth->SetLineWidth(1.0);
      hc2diff_onnx_efficiencytruth->SetLineColor(7);
      hc2diff_onnx_efficiencytruth_grad->SetLineWidth(1.0);
      hc2diff_onnx_efficiencytruth_grad->SetLineColor(7);
      hc2diff_onnx_efficiencytruth_grad->SetLineStyle(kDashed);      
      hc2diff_bdt->SetLineWidth(1.0);
      hc2diff_bdt->SetLineColor(kGreen+2);
      
      hc2diff_onnx_groundtruth_grad->Draw("hist");
      hc2diff_onnx_groundtruth->Draw("histsame");
      hc2diff_onnx_efficiencytruth->Draw("histsame");
      hc2diff_onnx_efficiencytruth_grad->Draw("histsame");      
      hc2diff_onnx_direct->Draw("histsame");
      hc2diff_onnx_direct_grad->Draw("histsame");      
      hc2diff_onnx_mse->Draw("histsame");
      hc2diff_onnx_mse_grad->Draw("histsame");      
      hc2diff_onnx_msemodified->Draw("histsame");
      hc2diff_onnx_msemodified_grad->Draw("histsame");      
      //hc2diff_onnx_bdt->Draw("histsame");
      //hc2diff_onnx_bdt_grad->Draw("histsame");      
      hc2diff_bdt->Draw("histsame");
      
      //hc2diff_onnx_groundtruth->Draw("histsame");
      //hc2diff_onnx_groundtruth_grad->Draw("histsame");      

      leg2->Draw();
      c1_->cd(2)->Print("diffs_c2.eps", "eps");
      c1_->cd(2)->Print("diffs_c2.root", "root");
      c1_->Print("diffs.eps", "eps");
      c1_->Print("diffs.root", "root");
#endif


      return 0;
    }

  return 0;
}
