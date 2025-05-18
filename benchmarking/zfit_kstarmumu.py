import os
os.environ['ZFIT_DISABLE_TF_WARNINGS'] = '0'

import numpy as np
import zfit
from zfit.core.space import ANY_LOWER, ANY_UPPER, Space, supports

import tensorflow as tf
print("TENSORFLOW VERSION",tf.__version__)
#tf.config.set_visible_devices([], 'GPU')

import hist
from scipy import stats

#zfit.run.set_n_cpu(n_cpu=16, strict=False)
zfit.settings.set_seed(1337)
minimiser = zfit.minimize.Minuit(minimize_strategy=2)


#implementation following zfit documentation
class KstarmumuPDF(zfit.pdf.ZPDF):
    _PARAMS = ['FL', 'S3', 'S4', 'S5', 'AFB', 'S7', 'S8', 'S9']
    _N_OBS = 3
    def _unnormalized_pdf(self, x):
        FL = self.params['FL']
        S3 = self.params['S3']
        S4 = self.params['S4']
        S5 = self.params['S5']
        AFB = self.params['AFB']
        S7 = self.params['S7']
        S8 = self.params['S8']
        S9 = self.params['S9']
        costheta_l, costheta_k, phi = zfit.z.zextension.unstack_x(x)
        sintheta_k = tf.sqrt(1.0 - costheta_k * costheta_k)
        sintheta_l = tf.sqrt(1.0 - costheta_l * costheta_l)
        sintheta_2k = (1.0 - costheta_k * costheta_k)
        sintheta_2l = (1.0 - costheta_l * costheta_l)
        sin2theta_k = (2.0 * sintheta_k * costheta_k)
        cos2theta_l = (2.0 * costheta_l * costheta_l - 1.0)
        sin2theta_l = (2.0 * sintheta_l * costheta_l)
        pdf = 9.0/32.0/np.pi*((3.0 / 4.0) * (1.0 - FL) * sintheta_2k +
                              FL * costheta_k * costheta_k +
                              (1.0 / 4.0) * (1.0 - FL) * sintheta_2k * cos2theta_l +
                              -1.0 * FL * costheta_k * costheta_k * cos2theta_l +
                              S3 * sintheta_2k * sintheta_2l * tf.cos(2.0 * phi) +
                              S4 * sin2theta_k * sin2theta_l * tf.cos(phi) +
                              S5 * sin2theta_k * sintheta_l * tf.cos(phi) +
                              (4.0 / 3.0) * AFB * sintheta_2k * costheta_l +
                              S7 * sin2theta_k * sintheta_l * tf.sin(phi) +
                              S8 * sin2theta_k * sin2theta_l * tf.sin(phi) +
                              S9 * sintheta_2k * sintheta_2l * tf.sin(2.0 * phi))
        return pdf
    #
    @supports()
    def _integrate(self, limits, norm_range, options):
        print("returning norm 1.0")
        return 1.0 #integral over full range


#dimension
costhetha_k = zfit.Space('costheta_k', (-1, 1))
costhetha_l = zfit.Space('costheta_l', (-1, 1))
phi = zfit.Space('phi', (-np.pi, np.pi))
angular_obs = costhetha_k * costhetha_l * phi

params_init = {'FL':  0.6, 'S3': 0.0, 'S4': 0.0, 'S5': 0.0, 'AFB': 0.0, 'S7': 0.0, 'S8': 0.0, 'S9': 0.0}
params = {name: zfit.Parameter(name, val, -1, 1) for name, val in params_init.items()}
model = KstarmumuPDF(obs=angular_obs, **params)

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
            for name, val in params_init.items():
                params[name].set_value(val)
            sampler.resample()
            toy_result = minimiser.minimize(toy_nll)
            toy_result.hesse(name='minuit_hesse')
            print(toy_result)
            for name, val in params_init.items():
                params[name].set_value(val)
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
