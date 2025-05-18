#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <span>
#include <algorithm>

#include "TRandom3.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TMath.h"
#include "TStyle.h"

#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooFitResult.h"
#include "RooDataSet.h"
#include "RooAddPdf.h"
#include "RooMCStudy.h"
#include "RooPlot.h"

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
		RooAbsReal& Afb, RooAbsReal& S7, RooAbsReal& S8, RooAbsReal& S9):
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
  RooRealVar costhetal("costhetal", "costhetal", -1.0, 1.0);
  RooRealVar costhetak("costhetak", "costhetak", -1.0, 1.0);
  RooRealVar phi("phi", "phi", -TMath::Pi(), +TMath::Pi());

  RooRealVar Fl("Fl", "Fl", 0.6, 0.0, 1.0);
  RooRealVar S3("S3", "S3", 0.0, -1.0, 1.0);
  RooRealVar S4("S4", "S4", 0.0, -1.0, 1.0);
  RooRealVar S5("S5", "S5", 0.0, -1.0, 1.0);
  RooRealVar Afb("Afb", "Afb", 0.0, -1.0, 1.0);
  RooRealVar S7("S7", "S7", 0.0, -1.0, 1.0);
  RooRealVar S8("S8", "S8", 0.0, -1.0, 1.0);
  RooRealVar S9("S9", "S9", 0.0, -1.0, 1.0);

  Fl.setError(0.01);
  S3.setError(0.01);
  S4.setError(0.01);
  S5.setError(0.01);
  Afb.setError(0.01);
  S7.setError(0.01);
  S8.setError(0.01);
  S9.setError(0.01);
  
  kstarmumu_pdf model("model","model", costhetal, costhetak, phi, Fl, S3, S4, S5, Afb, S7, S8, S9);

  unsigned int nrepeats = 10;
  const unsigned int npoints = 4;
  unsigned int nstats[npoints] = {1000, 10000, 100000, 1000000};
  std::vector<double> means, rmss;
  for (unsigned int n=0; n<npoints; n++)
    {
      std::vector<double> runtimes;
      for (unsigned int q=0; q<nrepeats; q++)
	{
	  Fl.setVal(0.60);
	  S3.setVal(0.0);
	  S4.setVal(0.0);
	  S5.setVal(0.0);
	  Afb.setVal(0.0);
	  S7.setVal(0.0);
	  S8.setVal(0.0);
	  S9.setVal(0.0);
	  
	  Fl.setError(0.01);
	  S3.setError(0.01);
	  S4.setError(0.01);
	  S5.setError(0.01);
	  Afb.setError(0.01);
	  S7.setError(0.01);
	  S8.setError(0.01);
	  S9.setError(0.01);
  
	  auto t_before_toystudy = std::chrono::high_resolution_clock::now();

	  unsigned int ngen = nstats[n];
	  unsigned int nruns = 100;

	  RooMCStudy *mcstudy =
	    //new RooMCStudy(model, RooArgSet(costhetal, costhetak, phi), Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("cuda")));
	    //new RooMCStudy(model, RooArgSet(costhetal, costhetak, phi), Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("cpu")));
	    new RooMCStudy(model, RooArgSet(costhetal, costhetak, phi), Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("legacy"), NumCPU(16)));

	  mcstudy->generateAndFit(nruns, ngen);
   
	  auto t_after_toystudy = std::chrono::high_resolution_clock::now();
	  std::cout << "toystudy takes " << std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count() << " ms in total" << std::endl;
	  runtimes.push_back(std::chrono::duration<double, std::milli>(t_after_toystudy-t_before_toystudy).count());
	}
      double mean, rms;
      mean_rms<double, double>(runtimes, mean, rms);
      std::cout << "Toy study with " << nstats[n] << " nevents, ms runtime mean " << mean << " rms " << rms << std::endl;
      means.push_back(mean);
      rmss.push_back(rms);
    }

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
