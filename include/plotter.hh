/**
 * @file plotter.hh
 * @author Christoph Langenbruch
 * @date 2025-01-21
 *
 */

#ifndef PLOTTER_H
#define PLOTTER_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <array>
#include <set>
#include <math.h>
#include <eigen3/Eigen/Dense>

#ifdef WITH_ROOT
#include "TCanvas.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TLine.h"
#include "TStyle.h"
#include "TGaxis.h"
#endif

#include "graph.hh"
#include "eventvector.hh"
#include "parametervector.hh"
#include "dimensionvector.hh"
#include "pdf.hh"
#include "physicspdfs.hh"
#include "compute.hh"
#include "utils.hh"

namespace morefit {

  struct plotter_options {
    enum plotter_type {Root, MatPlotLib};
    plotter_type plotter;
    int pdf_bin_multiplier;
    std::string file_format;
    bool optimize_parameters;
    float buffering_cost_threshold;    
    int print_level;
    int plot_width;
    float aspect_ratio;
    bool plot_pulls;
    float pull_fraction;
    float max_pull;
    plotter_options():
      plotter(plotter_type::Root),
      pdf_bin_multiplier(10),
      file_format("pdf"),
      optimize_parameters(false),
      buffering_cost_threshold(2.0),
      print_level(2),
      plot_width(1200),
      aspect_ratio(4.0/3.0),
      plot_pulls(false),
      pull_fraction(0.20),
      max_pull(3.0)
    {}
    void print()
    {
      const unsigned int width = 40;      
      std::cout << "PLOTTER OPTIONS" << std::endl;
      std::cout << std::setw(width) << std::left << "  plotter type ";
      switch (plotter) {
      case plotter_type::Root: std::cout << "Root" << std::endl; break;
      case plotter_type::MatPlotLib: std::cout << "MatPlotLib" << std::endl; break;
      default: std::cout << "UNKNOWN" << std::endl;
      }
      std::cout << std::setw(width) << std::left << "  pdf bin multiplier " << pdf_bin_multiplier << std::endl;
      std::cout << std::setw(width) << std::left << "  file format " << file_format << std::endl;
      std::cout << std::setw(width) << std::left << "  optimize parameters " << (optimize_parameters ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  buffering cost threshold " << buffering_cost_threshold << std::endl;
      std::cout << std::setw(width) << std::left << "  print level " << print_level << std::endl;
      std::cout << std::setw(width) << std::left << "  plot width " << plot_width << std::endl;
      std::cout << std::setw(width) << std::left << "  aspect ratio " << aspect_ratio << std::endl;
      std::cout << std::setw(width) << std::left << "  plot pulls " << (plot_pulls ? "YES" : "NO") << std::endl;
      std::cout << std::setw(width) << std::left << "  pull fraction " << pull_fraction << std::endl;
      std::cout << std::setw(width) << std::left << "  maximum pull range " << max_pull << std::endl;
    }
  };

  
  //plotting class
  //integration for pdfs is done centrally, parallelizing over bins
  //due to the parallelisation we also here use the different compute blocks/backends
  template<typename kernelT, typename evalT, typename backendT, typename computeT>
  class plotter {
  private:
    plotter_options* opts_;
    computeT block_;
    backendT* backend_;
    dimension<evalT> res_dim_;
    dimension<evalT> from_dim_;
    dimension<evalT> to_dim_;
    EventVector<kernelT, evalT> res_buffer_;
    EventVector<kernelT, evalT> ranges_buffer_;
    std::vector<parameter<evalT>*> params_;
    PDF<kernelT, evalT>* pdf_;
    EventVector<kernelT, evalT>* data_;
    std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> buffer_expressions_;
    std::vector<std::string> buffer_names_;
    std::string variable_;
    std::string replace_pound_signs(const std::string& str) const
    {
      std::string res="";
      for(const char& c : str)
	{
	  if (c=='#')
	    res += '\\';
	  else
	    res += c;
	}
      return res;
    }
  public:
    plotter(plotter_options* options, backendT* backend):
      opts_(options),
      block_(computeT(backend)),
      backend_(backend),
      res_dim_("integral", 0.0, 1.0),
      from_dim_("morefit_from", 0.0, 1.0),
      to_dim_("morefit_to", 0.0, 1.0),
      res_buffer_({&res_dim_}),
      ranges_buffer_({&from_dim_, &to_dim_})
    {
    }
    bool make_kernels(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data)
    {
      std::vector<std::string> floating_params;
      for (auto param : params)
	if (!param->is_constant())
	  floating_params.push_back(param->get_name());
      
      //all params are fixed for plotting
      std::vector<std::string> param_names;
      std::vector<evalT> param_values;
      for (auto param : params)
	{
	  param_names.push_back(param->get_name());
	  param_values.push_back(param->get_value());      
	}

      const std::vector<dimension<evalT>*>& dims = pdf->dimensions();
      std::vector<std::string> dimensions_names;
      std::vector<evalT> dimensions_values;
      std::string variable_from=variable_, variable_to=variable_;
      for (unsigned int i=0; i<dims.size(); i++)
	if (dims.at(i)->get_name() != variable_)
	  {
	    dimensions_names.push_back(dims.at(i)->get_from_name());
	    dimensions_values.push_back(dims.at(i)->get_min());
	    dimensions_names.push_back(dims.at(i)->get_to_name());
	    dimensions_values.push_back(dims.at(i)->get_max());	    
	  }
	else
	  {
	    variable_from = dims.at(i)->get_from_name();
	    variable_to = dims.at(i)->get_to_name();
	  }

      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> definite_integral = pdf->definite_integral()->substitute(param_names, param_values)->simplify();
      std::unique_ptr<ComputeGraphNode<kernelT, evalT>> integral = definite_integral->substitute(dimensions_names, dimensions_values)->simplify();
      integral->rename_variable(variable_from, "morefit_from");
      integral->rename_variable(variable_to, "morefit_to");
      
      //potential optimisations kernel on expressions depending only on parameters
      buffer_expressions_.clear();
      buffer_names_.clear();
      std::vector<std::unique_ptr<ComputeGraphNode<kernelT, evalT>>> graphs;      
      if (opts_->optimize_parameters)
	{
	  graphs.emplace_back(std::move(integral->optimize_buffering_constant_terms(buffer_names_, buffer_expressions_, floating_params, "morefit_parambuffer_", opts_->buffering_cost_threshold)));	  
	  //output check
	  if (opts_->print_level > 1)	    
	    {
	      std::cout << std::endl;
	      for (unsigned int i=0; i<graphs.size(); i++)
		std::cout << "GRAPHS: " << graphs.at(i)->get_kernel() << std::endl;
	      for (unsigned int i=0; i< buffer_expressions_.size(); i++)
		std::cout << "BUFFER " << buffer_names_.at(i) << " = " << buffer_expressions_.at(i)->get_kernel() << std::endl;
	      std::cout << std::endl;
	    }
	}
      else
	graphs.emplace_back(std::move(integral->copy()));

      //set up buffers
      block_.SetupInputBuffer(ranges_buffer_.buffer_size());
      block_.SetupOutputBuffer(res_buffer_.buffer_size());
      block_.SetupParameterBuffer((floating_params.size()+buffer_expressions_.size())*sizeof(kernelT));
	  
      std::vector<std::string> paramnames(floating_params);
      for (auto buffer_name : buffer_names_)
	paramnames.push_back(buffer_name);

      //make kernel
      auto t_before_kernel = std::chrono::high_resolution_clock::now();
      block_.MakeComputeKernel("integral_kernel", ranges_buffer_.nevents(), ranges_buffer_.copy_dimensions(), res_buffer_.copy_dimensions(), paramnames, graphs, false);
      block_.Finish();
      auto t_after_kernel = std::chrono::high_resolution_clock::now();

      if (opts_->print_level > 1)
	std::cout << "kernel (lh) takes " << std::chrono::duration<double, std::milli>(t_after_kernel-t_before_kernel).count() << " ms in total" << std::endl;
      return true;
    }
    //plot
    bool plot(PDF<kernelT, evalT>* pdf, std::vector<parameter<evalT>*> params, EventVector<kernelT, evalT>* data, std::string variable, std::string filename, std::string output_type, unsigned int nbins=100, evalT range_min=0.0, evalT range_max=0.0, bool replot = false)
    {
      auto t_before_plot = std::chrono::high_resolution_clock::now();

      pdf_ = pdf;
      params_ = params;
      data_ = data;
      variable_ = variable;

      int varidx = -1;
      dimension<evalT>* dim = nullptr;
      std::vector<dimension<evalT>*> dims = pdf->dimensions();
      for (unsigned int i=0; i<dims.size(); i++)
	if (dims.at(i)->get_name() == variable)
	  {
	    varidx = i;
	    dim = dims.at(i);
	    break;
	  }    
      if (varidx == -1)
	{
	  std::cout << "Variable " << variable << " not found, unable to plot." << std::endl;
	  assert(0);
	}      
      if (range_min == range_max)
	{
	  range_min = dim->get_min();
	  range_max = dim->get_max();
	}
      unsigned int npdf_bins = nbins * opts_->pdf_bin_multiplier;
      bool padded = backend_->required_padding() != 0;
      ranges_buffer_.set_padding(padded, backend_->required_padding());
      res_buffer_.set_padding(padded, backend_->required_padding());
      res_buffer_.resize(npdf_bins);
      ranges_buffer_.resize(npdf_bins);
      evalT dx = (range_max-range_min)/evalT(npdf_bins);
      for (unsigned int i=0; i<npdf_bins; i++)
	{
	  evalT from = range_min + i*dx;
	  evalT to = range_min + (i+1)*dx;
	  ranges_buffer_(i,0) = from;
	  ranges_buffer_(i,1) = to;
	}
      //make the kernel only if necessary 
      if (!replot)
	make_kernels(pdf, params, data);

      block_.SetupInputBuffer(ranges_buffer_.buffer_size());      
      block_.SetupOutputBuffer(res_buffer_.buffer_size());
      block_.SetNevents(ranges_buffer_.nevents(), ranges_buffer_.nevents_padded());
      block_.CopyToInputBuffer(ranges_buffer_);

      std::vector<std::string> parameter_names;
      std::vector<evalT> parameter_values;
      std::vector<kernelT> parameter_buffer;
      unsigned int nfloating_parameters=0;
      auto t_before_paramcalc = std::chrono::high_resolution_clock::now();
      for (const auto& param: params)
	if (!param->is_constant())
	  {
	    parameter_names.push_back(param->get_name());
	    parameter_buffer.push_back(param->get_value());
	    parameter_values.push_back(param->get_value());
	    nfloating_parameters++;
	  }
      for (const auto& expression : buffer_expressions_)
	parameter_buffer.push_back(expression->eval(parameter_names, parameter_values));
      auto t_after_paramcalc = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "param calc (plotter) takes " << std::chrono::duration<double, std::milli>(t_after_paramcalc-t_before_paramcalc).count() << " ms in total" << std::endl;      
      block_.CopyToParameterBuffer(parameter_buffer);
      
      //run, calculate integrals
      block_.SubmitKernel();
      block_.Finish();
      block_.CopyFromOutputBuffer(res_buffer_);

      //pdf
      std::vector<evalT> pdf_y(npdf_bins, 0.0);
      for (unsigned int i=0; i<npdf_bins;i++)
	pdf_y.at(i) = res_buffer_(i, 0);
      std::vector<evalT> pdf_dy(npdf_bins, 0.0);
      //data
      std::vector<evalT> y(nbins, 0.0);
      std::vector<evalT> dy(nbins, 0.0);
      if (data->has_event_weight())
	{
	  int weightidx = data->event_weight_idx();
	  for (unsigned int i=0; i<data->nevents(); i++)
	    {
	      evalT x = data->operator()(i, varidx);
	      int idx = floor((x-range_min)/(range_max-range_min)*nbins);
	      evalT w = data->operator()(i, weightidx);
	      y.at(idx) += w;
	      dy.at(idx) += w*w;
	    }
	  for (unsigned int i=0; i<nbins; i++)
	    dy.at(i) = sqrt(dy.at(i));
	}
      else
	{
	  for (unsigned int i=0; i<data->nevents(); i++)
	    {
	      evalT x = data->operator()(i, varidx);
	      int idx = floor((x-range_min)/(range_max-range_min)*nbins);
	      y.at(idx) += 1.0;
	    }
	  for (unsigned int i=0; i<nbins; i++)
	    dy.at(i) = sqrt(y.at(i));
	}
      //pulls
      double hist_int = 0.0;
      for (unsigned int i=0; i<nbins; i++)
	hist_int += y.at(i);
      double pdf_int = 0.0;
      for (unsigned int i=0; i<npdf_bins; i++)
	pdf_int += pdf_y.at(i);      
      double pdf_scale = hist_int/pdf_int*npdf_bins/double(nbins);
      std::vector<evalT> pulls(nbins, 0.0);
      std::vector<evalT> residuals_y(nbins, 0.0);
      std::vector<evalT> residuals_dy(nbins, 0.0);
      for (unsigned int i=0; i<npdf_bins; i++)
	{
	  int ibin = i/opts_->pdf_bin_multiplier;
	  residuals_y.at(ibin) += (y.at(ibin)-pdf_y.at(i)*pdf_scale)/opts_->pdf_bin_multiplier;
	}
      //pulls
      for (unsigned int i=0; i<nbins; i++)
	{
	  residuals_dy.at(i) = dy.at(i);
	  double pull = residuals_y.at(i)/dy.at(i);
	  pulls.at(i) = pull;	  
	}

      double max_pull = 0.0;
      if (opts_->max_pull > 0.0)
	max_pull = opts_->max_pull;
      else
	{
	  for (unsigned int i=0; i<nbins; i++)
	    {
	      double pull = pulls.at(i);
	      if (fabs(pull) > max_pull)
		max_pull = fabs(pull);
	    }
	  max_pull *= 1.05;
	}
      
      //actual plotting
      if (opts_->plotter == plotter_options::plotter_type::MatPlotLib)
	{
	  if (std::filesystem::path(filename).extension() == ".py")
	    {	      
	      std::ofstream outfile;
	      outfile.open(filename);
	      outfile << "import matplotlib.pyplot as plt\n"
		      << "import numpy as np\n"
		      << "plt.rcParams['text.usetex'] = True\n";
	      outfile << "x = np.linspace(" << range_min+0.5*(range_max-range_min)/double(nbins) << ", " << range_max-0.5*(range_max-range_min)/double(nbins) << ", " << nbins << ")\n";
	      outfile << "dx = [" << 0.5*(range_max-range_min)/double(nbins) << " for x in range("<< nbins <<")]\n";
	      outfile << "y = [";
	      for (unsigned int i=0; i<y.size(); i++)
		outfile << y.at(i) << (i<y.size()-1 ? ", " : "]\n");
	      outfile << "dy = [";
	      for (unsigned int i=0; i<dy.size(); i++)
		outfile << 0.5*dy.at(i) << (i<dy.size()-1 ? ", " : "]\n");
	      if (opts_->plot_pulls)
		{
		  outfile << "y_pull = [";
		  for (unsigned int i=0; i<pulls.size(); i++)
		    outfile << pulls.at(i) << (i<pulls.size()-1 ? ", " : "]\n");
		}
	      
	      double hist_max = y.at(0);
	      for (unsigned int i=0; i<nbins; i++)
		if (y.at(i) > hist_max)
		  hist_max = y.at(i);
	      
	      outfile << "x_pdf = np.linspace(" << range_min+0.5*(range_max-range_min)/double(npdf_bins) << ", " << range_max-0.5*(range_max-range_min)/double(npdf_bins) << ", " << npdf_bins << ")\n";
	      outfile << "y_pdf = [";
	      for (unsigned int i=0; i<pdf_y.size(); i++)
		outfile << pdf_y.at(i)*hist_int/pdf_int*npdf_bins/double(nbins) << (i<pdf_y.size()-1 ? ", " : "]\n");

	      if (opts_->plot_pulls)
		{
		  outfile << "fig, ax = plt.subplots(nrows=2, ncols=1,  layout='constrained', height_ratios=[" << 1.0-opts_->pull_fraction << ", " << opts_->pull_fraction << "])\n";
		  outfile << "ax[0].plot(x_pdf, y_pdf, linestyle='solid', color='red')\n";
		  outfile << "ax[0].errorbar(x, y, xerr=dx, yerr=dy, linestyle='none', marker='o', markersize=2, color='black')\n";
		  outfile << "ax[0].set_xlabel('$" << replace_pound_signs(dim->get_tex()) << "$')\n";
		  outfile << "ax[0].set_ylabel('$\\mathrm{events}$')\n";
		  outfile << "ax[0].set_xlim(" << range_min << ", " << range_max << ")\n";
		  outfile << "ax[0].set_ylim(" << 0.0 << ", " << hist_max*1.15 << ")\n";
		  outfile << "ax[1].bar(x=x, height=y_pull, width=" << (range_max-range_min)/double(nbins) << ", facecolor='gray', linestyle='solid', edgecolor='black')\n";
		  outfile << "#ax[1].set_xlabel('$" << replace_pound_signs(dim->get_tex()) << "$')\n";
		  outfile << "ax[1].set_ylabel('$\\mathrm{pull}$')\n";
		  outfile << "ax[1].set_xlim(" << range_min << ", " << range_max << ")\n";
		  outfile << "ax[1].set_ylim(-" << max_pull << ", " << max_pull << ")\n";
		  outfile << "plt.show()\n";
		  outfile << "fig.savefig(\"" << (std::filesystem::path(filename).stem().string()+".pdf") << "\", bbox_inches='tight')\n";
		}
	      else
		{
		  outfile << "fig, ax = plt.subplots()\n";
		  outfile << "ax.plot(x_pdf, y_pdf, linestyle='solid', color='red')\n";
		  outfile << "ax.errorbar(x, y, xerr=dx, yerr=dy, linestyle='none', marker='o', markersize=2, color='black')\n";
		  outfile << "ax.set_xlabel('$" << replace_pound_signs(dim->get_tex()) << "$')\n";
		  outfile << "ax.set_ylabel('$\\mathrm{events}$')\n";
		  outfile << "plt.xlim(" << range_min << ", " << range_max << ")\n";
		  outfile << "plt.ylim(" << 0.0 << ", " << hist_max*1.15 << ")\n";
		  outfile << "plt.show()\n";
		  outfile << "fig.savefig(\"" << (std::filesystem::path(filename).stem().string()+".pdf") << "\", bbox_inches='tight')\n";
		}
	      outfile.close();
	      if (opts_->print_level > 1)
		std::cout << "Written plot of variable " << variable << " to file " << filename << std::endl;
	    }
	  else
	    std::cout << "Unknown file extension: " << filename << std::endl;
	}
      else if (opts_->plotter == plotter_options::plotter_type::Root)
	{
	  if (std::filesystem::path(filename).extension() == ".C")
	    {	      
	      std::ofstream outfile;
	      outfile.open(filename);
	      outfile << "int " << std::filesystem::path(filename).stem().string() << "() {\n";
	      outfile << "gROOT->SetStyle(\"Plain\");\n"
		      << "gStyle->SetOptFit(0);\n"
		      << "gStyle->SetOptStat(0);\n"
		      << "gStyle->SetTextFont(132);\n"
		      << "gStyle->SetTextSize(0.06);\n"
		      << "gStyle->SetTitleFont(132,\"xyz\");\n"
		      << "gStyle->SetLabelFont(132,\"xyz\");\n"
		      << "gStyle->SetLabelSize(0.05,\"xyz\");\n"
		      << "gStyle->SetTitleSize(0.06,\"xyz\");\n";
	      outfile << "TCanvas* c0 = new TCanvas(\"c0\", \"c0\", " << opts_->plot_width << ", " << opts_->plot_width/evalT(opts_->aspect_ratio) << ");\n";
	      outfile << "TH1D* hist = new TH1D(\"hist\", \";" << dim->get_tex() << ";events\", " << nbins << ", " << range_min << ", " << range_max << ");\n";
	      outfile << "double hist_values[" << y.size() << "] = {";
	      for (unsigned int i=0; i<y.size(); i++)
		outfile << y.at(i) << (i<y.size()-1 ? ", " : "};\n");
	      outfile << "double hist_errors[" << dy.size() << "] = {";
	      for (unsigned int i=0; i<dy.size(); i++)
		outfile << dy.at(i) << (i<dy.size()-1 ? ", " : "};\n");
	      outfile << "for (unsigned int i=0; i<" << y.size() << "; i++)\n"
		      << "{\n"
		      << "  hist->SetBinContent(i+1, hist_values[i]);\n"
		      << "  hist->SetBinError(i+1, hist_errors[i]);\n"
		      <<"}\n";
	      outfile << "TH1D* hist_pdf = new TH1D(\"hist_pdf\", \";" << dim->get_tex() << ";events\", " << npdf_bins << ", " << range_min << ", " << range_max << ");\n";
	      outfile << "double pdf_values[" << pdf_y.size() << "] = {";
	      for (unsigned int i=0; i<pdf_y.size(); i++)
		outfile << pdf_y.at(i) << (i<pdf_y.size()-1 ? ", " : "};\n");
	      outfile << "double pdf_errors[" << pdf_dy.size() << "] = {";
	      for (unsigned int i=0; i<pdf_dy.size(); i++)
		outfile << pdf_dy.at(i) << (i<pdf_dy.size()-1 ? ", " : "};\n");
	      outfile << "for (unsigned int i=0; i<" << pdf_y.size() << "; i++)\n"
		      << "{\n"
		      << "  hist_pdf->SetBinContent(i+1, pdf_values[i]);\n"
		      << "  hist_pdf->SetBinError(i+1, pdf_errors[i]);\n"
		      <<"}\n";
	      outfile << "hist_pdf->Scale(hist->Integral()/hist_pdf->Integral()*" << npdf_bins << "/double(" << nbins << "));\n";
	      if (opts_->plot_pulls)
		{
		  outfile << "TH1D* pull_hist = new TH1D(\"pull_hist\", \";" << dim->get_tex() << ";pull\", " << nbins << ", " << range_min << ", " << range_max << ");\n";
		  outfile << "double pull_values[" << pulls.size() << "] = {";
		  for (unsigned int i=0; i<pulls.size(); i++)
		    outfile << pulls.at(i) << (i<pulls.size()-1 ? ", " : "};\n");
		  outfile << "for (unsigned int i=0; i<" << pulls.size() << "; i++)\n"
			  << "{\n"
			  << "  pull_hist->SetBinContent(i+1, pull_values[i]);\n"
			  << "  pull_hist->SetBinError(i+1, 0.0);\n"
			  <<"}\n"
			  << "c0->cd(0);\n"
			  << "TPad *hist_pad = new TPad(\"hist_pad\",\"\",0.0, " << opts_->pull_fraction << ", 1.0, 1.0);\n"
			  << "hist_pad->SetMargin(0.125,0.05,0.125,0.05);\n"
			  << "hist_pad->Draw();\n"
			  << "hist_pad->cd();\n"
			  << "hist->SetMinimum(0.0);\n"
			  << "hist->Draw(\"e\");\n"
			  << "hist_pdf->SetLineColor(2);\n"
			  << "hist_pdf->SetLineWidth(2.0);\n"
			  << "hist_pdf->Draw(\"lcsame\");\n"
			  << "c0->cd(0);\n"
			  << "TPad *pull_pad = new TPad(\"pull_pad\",\"\",0.0, 0.0, 1.0, " << opts_->pull_fraction << ");\n"
			  << "pull_pad->SetMargin(0.125,0.05,0.125,0.125);\n"
			  << "pull_pad->Draw();\n"
			  << "pull_pad->cd();\n"
			  << "pull_hist->SetFillColor(kGray);\n"
			  << "pull_hist->GetYaxis()->SetTitleOffset(1.0*" << opts_->pull_fraction/(1.0-opts_->pull_fraction) << ");\n"
			  << "pull_hist->GetYaxis()->SetLabelSize(0.05*" << (1.0-opts_->pull_fraction)/opts_->pull_fraction << ");\n"
			  << "pull_hist->GetYaxis()->SetTitleSize(0.06*" << (1.0-opts_->pull_fraction)/opts_->pull_fraction << ");\n"
			  << "pull_hist->GetXaxis()->SetLabelSize(0);\n"
			  << "pull_hist->GetXaxis()->SetTitleSize(0);\n"
			  << "pull_hist->SetMinimum(-" << max_pull << ");\n"
			  << "pull_hist->SetMaximum(+" << max_pull << ");\n"
			  << "pull_hist->Draw(\"hist\");\n"
			  << "TGaxis *axis = new TGaxis(pull_hist->GetXaxis()->GetXmin(), pull_hist->GetMaximum(), pull_hist->GetXaxis()->GetXmax(), pull_hist->GetMaximum(), pull_hist->GetXaxis()->GetXmin(), pull_hist->GetXaxis()->GetXmax(), 510, \"-U\");\n"
			  << "axis->Draw();\n"
			  << "c0->Print(\"" << std::filesystem::path(filename).stem().string() << ".eps\", \"eps\");\n"
			  << "delete hist;\n"
			  << "delete hist_pdf;\n"
			  << "delete hist_pad;\n"
			  << "delete pull_hist;\n"
			  << "delete pull_pad;\n"		  
			  << "delete c0;\n";
		}
	      else
		{
		  outfile << "c0->cd()->SetMargin(0.125,0.05,0.125,0.05);\n"
			  << "hist->SetMinimum(0.0);\n"
			  << "hist->Draw(\"e\");\n"
			  << "hist_pdf->SetLineColor(2);\n"
			  << "hist_pdf->SetLineWidth(2.0);\n"
			  << "hist_pdf->Draw(\"lcsame\");\n"
			  << "c0->Print(\"" << std::filesystem::path(filename).stem().string() << ".eps\", \"eps\");\n"
			  << "delete hist;\n"
			  << "delete hist_pdf;\n"
			  << "delete c0;\n";
		}
	      outfile << "return 1;\n"
		      << "}\n";
	      outfile.close();
	      if (opts_->print_level > 1)
		std::cout << "Written plot of variable " << variable << " to file " << filename << std::endl;
	    }
#ifdef WITH_ROOT
	  else if (std::filesystem::path(filename).extension() == ".eps")
	    {
	      gROOT->SetStyle("Plain");
	      gStyle->SetOptFit(0);
	      gStyle->SetOptStat(0);
	      gStyle->SetTextFont(132);
	      gStyle->SetTextSize(0.06);
	      gStyle->SetTitleFont(132,"xyz");
	      gStyle->SetLabelFont(132,"xyz");
	      gStyle->SetLabelSize(0.05,"xyz");
	      gStyle->SetTitleSize(0.06,"xyz");
	  
	      TCanvas* c0 = new TCanvas("c0", "c0", opts_->plot_width, opts_->plot_width/opts_->aspect_ratio);
	      TH1D* hist = new TH1D("hist", (";"+dim->get_tex()+";events").c_str(), nbins, range_min, range_max);
	      hist->Sumw2(true);
	      if (data->has_event_weight())
		{
		  int weightidx = data->event_weight_idx();
		  for (unsigned int i=0; i<data->nevents(); i++)
		    hist->Fill(data->operator()(i, varidx), data->operator()(i, weightidx));
		}
	      else
		{	     
		  for (unsigned int i=0; i<data->nevents(); i++)
		    hist->Fill(data->operator()(i, varidx));
		}
	      TH1D* hist_pdf = new TH1D("hist_pdf", (";"+dim->get_tex()+";events").c_str(), npdf_bins, range_min, range_max);
	      for (unsigned int i=0; i<npdf_bins; i++)
		{
		  hist_pdf->SetBinContent(i+1,res_buffer_(i,0));
		  hist_pdf->SetBinError(i+1,0.0);
		}
	      hist_pdf->Scale(hist->Integral()/hist_pdf->Integral()*npdf_bins/evalT(nbins));
	      //draw
	      c0->cd();
	      hist->SetMinimum(0.0);
	      hist_pdf->SetLineColor(2);
	      hist_pdf->SetLineWidth(2.0);
	      if (opts_->plot_pulls)
		{
		  c0->cd(0);
		  TPad *hist_pad = new TPad("hist_pad","",0.0, opts_->pull_fraction, 1.0, 1.0);
		  hist_pad->SetMargin(0.125,0.05,0.125,0.05);
		  hist_pad->Draw();
		  hist_pad->cd();
		  hist->Draw("e");
		  hist_pdf->Draw("lcsame");
		  c0->cd(0);		  
		  TPad *pull_pad = new TPad("pull_pad","",0.0, 0.0, 1.0, opts_->pull_fraction);
		  pull_pad->SetMargin(0.125,0.05,0.125,0.125);
		  pull_pad->Draw();
		  pull_pad->cd();
		  TH1D* pull_hist = new TH1D("pull_hist", (";"+dim->get_tex()+";pull").c_str(), nbins, range_min, range_max);
		  for (unsigned int i=0; i<nbins; i++)
		    {
		      pull_hist->SetBinContent(i+1, pulls.at(i));
		      pull_hist->SetBinError(i+1, 0.0);
		    }
		  pull_hist->SetFillColor(kGray);
		  pull_hist->GetYaxis()->SetTitleOffset(1.0*opts_->pull_fraction/(1.0-opts_->pull_fraction));
		  pull_hist->GetYaxis()->SetLabelSize(0.05*(1.0-opts_->pull_fraction)/opts_->pull_fraction);
		  pull_hist->GetYaxis()->SetTitleSize(0.06*(1.0-opts_->pull_fraction)/opts_->pull_fraction);
		  pull_hist->GetXaxis()->SetLabelSize(0);		  
		  pull_hist->GetXaxis()->SetTitleSize(0);
		  pull_hist->SetMinimum(-max_pull);
		  pull_hist->SetMaximum(+max_pull);
		  pull_hist->Draw("hist");
		  TGaxis *axis = new TGaxis(pull_hist->GetXaxis()->GetXmin(), pull_hist->GetMaximum(), pull_hist->GetXaxis()->GetXmax(), pull_hist->GetMaximum(), pull_hist->GetXaxis()->GetXmin(), pull_hist->GetXaxis()->GetXmax(), 510, "-U");
		  axis->Draw();
		  c0->Print(filename.c_str(), output_type.c_str());
		  delete pull_hist;
		  delete pull_pad;
		  delete hist;
		  delete hist_pdf;
		  delete hist_pad;
		  delete c0;
		}
	      else
		{
		  c0->cd()->SetMargin(0.125,0.05,0.125,0.05);
		  hist->Draw("e");
		  hist_pdf->Draw("lcsame");
		  c0->Print(filename.c_str(), output_type.c_str());
		  delete hist;
		  delete hist_pdf;
		  delete c0;
		}
	    }
#endif
	  else
	    std::cout << "Unknown file extension: " << filename << std::endl;
	}
      //plot
      if (opts_->print_level > 1)
	std::cout << "plotting procedure finished" << std::endl;
      auto t_after_plot = std::chrono::high_resolution_clock::now();
      if (opts_->print_level > 1)
	std::cout << "plotting of " << data->nevents() << " events took " << std::chrono::duration<double, std::milli>(t_after_plot-t_before_plot).count() << " ms in total" << std::endl;

      return true;
    }
  };

}

#endif
