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

int main()
{
  RooRealVar m("m", "m", 5.0, 7.0);
  RooRealVar mb("mb", "mb", 5.28, 5.0, 6.0);
  RooRealVar sigma("sigma", "sigma", 0.06, 0.005, 0.130);
  RooRealVar fsig("fsig", "fsig", 0.3, 0.0, 1.0);
  RooRealVar alpha("alpha", "alpha", -1.0, -10.0, 10.0);
  
  RooGaussian gauss("gauss", "gauss", m, mb, sigma);
  RooExponential expo("expo", "expo", m, alpha);
  RooAddPdf model("model","model", RooArgList(gauss, expo), fsig);

  unsigned int nrepeats = 10;
  const unsigned int npoints = 4;
  unsigned int nstats[npoints] = {1000, 10000, 100000, 1000000};
  std::vector<double> means, rmss;
  for (unsigned int n=0; n<npoints; n++)
    {
      std::vector<double> runtimes;
      for (unsigned int q=0; q<nrepeats; q++)
	{
	  mb.setVal(5.28);
	  sigma.setVal(0.06);
	  fsig.setVal(0.3);
	  alpha.setVal(-1.0);

	  auto t_before_toystudy = std::chrono::high_resolution_clock::now();

	  unsigned int ngen = nstats[n];
	  unsigned int nruns = 100;
	  RooMCStudy *mcstudy =
	    //new RooMCStudy(model, m, Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("cuda")));
	    //new RooMCStudy(model, m, Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("cpu")));
	    new RooMCStudy(model, m, Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("legacy"), NumCPU(16)));
	    //new RooMCStudy(model, m, Binned(false), FitOptions(Strategy(2), Hesse(true), Save(true), PrintEvalErrors(0), EvalBackend("codegen")));
 
	  mcstudy->generateAndFit(nruns, ngen);
	  if (false)
	    {
	      RooPlot *frame1 = mcstudy->plotPull(mb, Bins(50), FitGauss(true));
	      RooPlot *frame2 = mcstudy->plotPull(sigma, Bins(50), FitGauss(true));
	      RooPlot *frame3 = mcstudy->plotPull(fsig, Bins(50), FitGauss(true));
	      RooPlot *frame4 = mcstudy->plotPull(alpha, Bins(50), FitGauss(true));

	      TCanvas* c0 = new TCanvas("c0", "c0", 1600, 1200);
	      c0->Divide(2,2);
	      c0->cd(1);
	      frame1->Draw();
	      c0->cd(2);
	      frame2->Draw();
	      c0->cd(3);
	      frame3->Draw();
	      c0->cd(4);
	      frame4->Draw();
	      c0->Print("roopulls.eps", "eps");
	    }

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
