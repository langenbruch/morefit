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
#include "TMath.h"


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

class kstarmumu_pdf: public RooAbsPdf {
public:
  kstarmumu_pdf(const char *name, const char *title,
		RooAbsReal& costhetal, RooAbsReal& costhetak, RooAbsReal& phi,
		RooAbsReal& Fl, RooAbsReal& S3, RooAbsReal& S4, RooAbsReal& S5,
		RooAbsReal& Afb, RooAbsReal& S7, RooAbsReal& S8,
		RooAbsReal& S9):
    RooAbsPdf(name, title),
    costhetal_("costhetal", "costhetal", this, costhetal),
    costhetak_("costhetak", "costhetak", this, costhetak),
    phi_("phi", "phi", this, phi),
    Fl_("Fl", "Fl", this, Fl),
    S3_("S3", "S3", this, S3),
    S4_("S4", "S4", this, S4),
    S5_("S5", "S5", this, S5),
    Afb_("Afb", "Afb", this, Afb),
    S7_("S7", "S7", this, S7),
    S8_("S8", "S8", this, S8),
    S9_("S9", "S9", this, S9)
  {}
  kstarmumu_pdf(kstarmumu_pdf const &other, const char *name=nullptr):
    RooAbsPdf(other, name),
    costhetal_("costhetal", this, other.costhetal_),
    costhetak_("costhetak", this, other.costhetak_),
    phi_("phi", this, other.phi_),
    Fl_("Fl", this, other.Fl_),
    S3_("S3", this, other.S3_),
    S4_("S4", this, other.S4_),
    S5_("S5", this, other.S5_),
    Afb_("Afb", this, other.Afb_),
    S7_("S7", this, other.S7_),
    S8_("S8", this, other.S8_),
    S9_("S9", this, other.S9_)
  {}
  TObject* clone(const char *newname) const override
  {
    return new kstarmumu_pdf(*this, newname);
  }
protected: 
  RooRealProxy costhetal_;
  RooRealProxy costhetak_;
  RooRealProxy phi_;
  RooRealProxy Fl_;
  RooRealProxy S3_;
  RooRealProxy S4_;
  RooRealProxy S5_;
  RooRealProxy Afb_;
  RooRealProxy S7_;
  RooRealProxy S8_;
  RooRealProxy S9_;

  inline double evaluate_prob(double costhetal, double costhetak, double phi, double Fl, double S3, double S4, double S5, double Afb, double S7, double S8, double S9) const
  {
    const double c = 9.0/32.0/TMath::Pi();
    const double costhetal2 = costhetal*costhetal;
    const double costhetak2 = costhetak*costhetak;
    const double cos2thetal = 2.0*costhetal2 - 1.0;
    //const double cos2thetak = 2.0*costhetak2 - 1.0;
    const double sinthetak2 = 1.0 - costhetak2;
    const double sinthetal2 = 1.0 - costhetal2;
    const double sinthetal = sqrt(sinthetal2);
    const double sinthetak = sqrt(sinthetak2);
    
    const double sin2thetal = 2.0*sinthetal*costhetal;
    const double sin2thetak = 2.0*sinthetak*costhetak;
    return c* (
	       sinthetak2 * 3.0/4.0*(1.0-Fl)
	       +costhetak2 * Fl
	       +sinthetak2*cos2thetal * 1.0/4.0*(1-Fl)
	       +costhetak2*cos2thetal * (-Fl)
	       +sinthetak2*sinthetal2*cos(2.0*phi) * S3
	       +sin2thetak*sin2thetal*cos(phi) * S4
	       +sin2thetak*sinthetal*cos(phi) * S5
	       +sinthetak2*costhetal * 4.0/3.0*Afb
	       +sin2thetak*sinthetal*sin(phi) * S7
	       +sin2thetak*sin2thetal*sin(phi) * S8
	       +sinthetak2*sinthetal2*sin(2.0*phi) * S9
	       );
  }
public:
  double evaluate() const override
  {
    return evaluate_prob(costhetal_, costhetak_, phi_, Fl_, S3_, S4_, S5_, Afb_, S7_, S8_, S9_);
  }
  void doEval(RooFit::EvalContext &ctx) const override
  {
    std::span<const double> costhetalSpan = ctx.at(costhetal_);
    std::span<const double> costhetakSpan = ctx.at(costhetak_);
    std::span<const double> phiSpan = ctx.at(phi_);

    std::span<const double> FlSpan = ctx.at(Fl_);
    std::span<const double> S3Span = ctx.at(S3_);
    std::span<const double> S4Span = ctx.at(S4_);
    std::span<const double> S5Span = ctx.at(S5_);
    std::span<const double> AfbSpan = ctx.at(Afb_);
    std::span<const double> S7Span = ctx.at(S7_);
    std::span<const double> S8Span = ctx.at(S8_);
    std::span<const double> S9Span = ctx.at(S9_);
    //std::cout << "used?" << std::endl;

    
    std::size_t n = ctx.output().size();
    for (std::size_t i = 0; i < n; ++i) {
      ctx.output()[i] = evaluate_prob(costhetalSpan.size() > 1 ? costhetalSpan[i] : costhetalSpan[0],
				      costhetakSpan.size() > 1 ? costhetakSpan[i] : costhetakSpan[0],
				      phiSpan.size() > 1 ? phiSpan[i] : phiSpan[0],
				      FlSpan.size() > 1 ? FlSpan[i] : FlSpan[0],				      
				      S3Span.size() > 1 ? S3Span[i] : S3Span[0],
				      S4Span.size() > 1 ? S4Span[i] : S4Span[0],
				      S5Span.size() > 1 ? S5Span[i] : S5Span[0],
				      AfbSpan.size() > 1 ? AfbSpan[i] : AfbSpan[0],
				      S7Span.size() > 1 ? S7Span[i] : S7Span[0],
				      S8Span.size() > 1 ? S8Span[i] : S8Span[0],
				      S9Span.size() > 1 ? S9Span[i] : S9Span[0]
				      );
    }
  }
  void translate(RooFit::Detail::CodeSquashContext &ctx) const override
  {
    ctx.addResult(this, ctx.buildCall("kstarmumu_pdf", costhetal_, costhetak_, phi_, Fl_, S3_, S4_, S5_, Afb_, S7_, S8_, S9_));
  }
  int getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char */*rangeName*/) const override
  {
    // Support also using the imaginary unit
    //using namespace std::complex_literals;
    // To be able to also comile C code, we define a variable that behaves like the "I" macro from C.
    //constexpr auto I = 1i;
    
    // LIST HERE OVER WHICH VARIABLES ANALYTICAL INTEGRATION IS SUPPORTED,
    // ASSIGN A NUMERIC CODE FOR EACH SUPPORTED (SET OF) PARAMETERS. THE EXAMPLE
    // BELOW ASSIGNS CODE 1 TO INTEGRATION OVER VARIABLE X YOU CAN ALSO
    // IMPLEMENT MORE THAN ONE ANALYTICAL INTEGRAL BY REPEATING THE matchArgs
    // EXPRESSION MULTIPLE TIMES.
    if (matchArgs(allVars,analVars,costhetal_,costhetak_,phi_)) return 1 ;
    return 0 ;
  }
  double analyticalIntegral(int code, const char *rangeName) const
  {
    // RETURN ANALYTICAL INTEGRAL DEFINED BY RETURN CODE ASSIGNED BY
    // getAnalyticalIntegral(). THE MEMBER FUNCTION x.min(rangeName) AND
    // x.max(rangeName) WILL RETURN THE INTEGRATION BOUNDARIES FOR EACH
    // OBSERVABLE x.
    if (code==1) {
      return 1.0;
    }
    return 0 ;
  }
};

int main()
{

  RooRealVar roofit_costhetal("costhetal", "costhetal", -1.0, 1.0);
  RooRealVar roofit_costhetak("costhetak", "costhetak", -1.0, 1.0);
  RooRealVar roofit_phi("phi", "phi", -TMath::Pi(), +TMath::Pi());

  RooRealVar roofit_Fl("Fl", "Fl", 0.6, 0.0, 1.0);
  RooRealVar roofit_S3("S3", "S3", 0.0, -1.0, 1.0);
  RooRealVar roofit_S4("S4", "S4", 0.0, -1.0, 1.0);
  RooRealVar roofit_S5("S5", "S5", 0.0, -1.0, 1.0);
  RooRealVar roofit_Afb("Afb", "Afb", 0.0, -1.0, 1.0);
  RooRealVar roofit_S7("S7", "S7", 0.0, -1.0, 1.0);
  RooRealVar roofit_S8("S8", "S8", 0.0, -1.0, 1.0);
  RooRealVar roofit_S9("S9", "S9", 0.0, -1.0, 1.0);

  roofit_Fl.setError(0.01);
  roofit_S3.setError(0.01);
  roofit_S4.setError(0.01);
  roofit_S5.setError(0.01);
  roofit_Afb.setError(0.01);
  roofit_S7.setError(0.01);
  roofit_S8.setError(0.01);
  roofit_S9.setError(0.01);
  
  kstarmumu_pdf roofit_model("model","model", roofit_costhetal, roofit_costhetak, roofit_phi, roofit_Fl, roofit_S3, roofit_S4, roofit_S5, roofit_Afb, roofit_S7, roofit_S8, roofit_S9);

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
  morefit::KstarmumuAngularPDF<kernelT, evalT> kstarmumu(&ctl, &ctk, &phi,
							 &Fl, &S3, &S4, &S5, &Afb, 
							 &S7, &S8, &S9);
  std::vector<morefit::parameter<evalT>*> params({&Fl, &S3, &S4, &S5, &Afb, 
      &S7, &S8, &S9});
  std::vector<std::string> param_names;
  std::vector<evalT> param_values;
  for (auto param : params)
    {
      std::cout << "param name: " << param->get_name() << " param value: " << param->get_value() << std::endl;
      param_names.push_back(param->get_name());
      param_values.push_back(param->get_value());      
    }

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

		  morefit::EventVector<kernelT, evalT> result({&ctl, &ctk, &phi}, ngen);	  
		  gen.generate(ngen, &kstarmumu, params, result);

		  morefit::fitter_options opts;
		  opts.minuit_printlevel = 0;
		  opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
		  //opts.minimizer = morefit::fitter_options::minimizer_type::TMinuit;
		  opts.analytic_gradient = false;
		  opts.analytic_hessian = false;
		  opts.analytic_fisher = false;
		  //opts.correct_weighted_fit = false;
		  //opts.analytic_gradient = false;
		  opts.optimize_parameters = true;
		  opts.optimize_dimensions = false;
		  opts.kahan_on_accelerator = false;
		  opts.print_level = 0;
		  morefit::fitter<kernelT, evalT, backendT, blockT> fit(&opts, &backend);
		  fit.fit(&kstarmumu, params, &result);
		  for (unsigned int j=0; j<params.size(); j++)
		    if (!params.at(j)->is_constant())
		      pulls.at(j).push_back((params.at(j)->get_value()-params.at(j)->get_start_value())/params.at(j)->get_error());		  

#ifdef WITH_ROOT
		  RooDataSet roofit_data("roofit_data","roofit_data",RooArgSet(roofit_costhetal, roofit_costhetak, roofit_phi));
		  for (unsigned int j=0; j<ngen; j++)
		    {
		      roofit_costhetal = result(j,0);
		      roofit_costhetak = result(j,1);
		      roofit_phi = result(j,2);
		      roofit_data.add(RooArgSet(roofit_costhetal, roofit_costhetak, roofit_phi));
		    }
		  roofit_Fl.setVal(Fl.get_start_value());
		  roofit_S3.setVal(S3.get_start_value());
		  roofit_S4.setVal(S4.get_start_value());
		  roofit_S5.setVal(S5.get_start_value());
		  roofit_Afb.setVal(Afb.get_start_value());
		  roofit_S7.setVal(S7.get_start_value());
		  roofit_S8.setVal(S8.get_start_value());
		  roofit_S9.setVal(S9.get_start_value());

		  roofit_Fl.setError(Fl.get_step_size());
		  roofit_S3.setError(S3.get_step_size());
		  roofit_S4.setError(S4.get_step_size());
		  roofit_S5.setError(S5.get_step_size());
		  roofit_Afb.setError(Afb.get_step_size());
		  roofit_S7.setError(S7.get_step_size());
		  roofit_S8.setError(S8.get_step_size());
		  roofit_S9.setError(S9.get_step_size());
		  
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
		  c0->Divide(3,3);
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
		  c1->Divide(3,3);
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
		  c2->Divide(3,3);
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
