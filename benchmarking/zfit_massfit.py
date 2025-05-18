import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '0'

import numpy as np
import zfit

import tensorflow as tf
print("TENSORFLOW VERSION",tf.__version__)
#tf.config.set_visible_devices([], 'GPU')

import hist
from scipy import stats

zfit.run.set_n_cpu(n_cpu=1, strict=True)
zfit.settings.set_seed(1337)
minimiser = zfit.minimize.Minuit(minimize_strategy=2)

#dimension
obs_m = zfit.Space('Mass', limits=(5.0, 7.0))
# parameters for signal and background shapes
mu_signal = zfit.Parameter('mu_signal', 5.28, 5.0, 6.0)
sigma_signal = zfit.Parameter('sigma_signal', 0.06, 0.005, 0.130)
fsig = zfit.Parameter('f_sig', 0.3, 0.0, 1.0)
slope_bkg = zfit.Parameter('slope_bkg', -1.0, -10.0, 10.0)
#model
gaussian = zfit.pdf.Gauss(obs=obs_m, mu=mu_signal, sigma=sigma_signal, name='Signal')
exponential = zfit.pdf.Exponential(obs=obs_m, lam=slope_bkg, name='Background')
model = zfit.pdf.SumPDF([gaussian, exponential], fracs=fsig)

import datetime

stats = [1000, 10000, 100000, 1000000]
nrepeats = 10
ntoysperstudy = 100
print("nrepeats",nrepeats,"ntoysperstudy",ntoysperstudy)

results_mean = []
results_rms = []

#benchmarking
for s in stats:
    deltas = []
    for j in range(0,nrepeats):
        print("stats",s,"repeat",j)
        before = datetime.datetime.now()
        sampler = model.create_sampler(n=s) 
        toy_nll = zfit.loss.UnbinnedNLL(model=model, data=sampler)
        for i in range(0,ntoysperstudy):
            print("toy no.",i)
            mu_signal.set_value(5.28)
            sigma_signal.set_value(0.06)
            fsig.set_value(0.3)
            slope_bkg.set_value(-1.0)
            sampler.resample()
            toy_result = minimiser.minimize(toy_nll)
            toy_result.hesse(name='minuit_hesse')
            print(toy_result)
            mu_signal.set_value(5.28)
            sigma_signal.set_value(0.06)
            fsig.set_value(0.3)
            slope_bkg.set_value(-1.0)
        after = datetime.datetime.now()
        delta = after - before
        print("runtime in ms",delta.total_seconds() * 1000.0)
        deltas.append(delta.total_seconds() * 1000.0)
    print("mean", np.mean(np.array(deltas)))
    print("rms", np.std(np.array(deltas)))
    results_mean.append(np.mean(np.array(deltas)))
    results_rms.append(np.std(np.array(deltas)))
print("results_mean",results_mean)
print("results_rms",results_rms)
